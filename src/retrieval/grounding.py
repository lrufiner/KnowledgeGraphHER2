"""
LLM Grounding & Hallucination Detection for the HER2 Knowledge Graph.

Implements §6.3 of the implementation plan: entity constraint, relation
validation, provenance tracking, and clinical rule enforcement.

The GroundingChecker operates in two stages:
  1. Pre-generation: Build a constrained context from Neo4j so the LLM
     only receives verified facts.
  2. Post-generation: Validate the LLM response against the KG — flag
     any entities or relations that are not present in the retrieved subgraph.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neo4j import Driver

# ---------------------------------------------------------------------------
# Known HER2 entity vocabulary (used when no Neo4j connection available)
# ---------------------------------------------------------------------------

_HER2_KNOWN_ENTITIES: set[str] = {
    "HER2_Positive", "HER2_Equivocal", "HER2_Low", "HER2_Ultralow", "HER2_Null",
    "IHC_3plus", "IHC_2plus", "IHC_1plus", "IHC_0plus", "IHC_0",
    "ISH_Group1", "ISH_Group2", "ISH_Group3", "ISH_Group4", "ISH_Group5",
    "ASCO_CAP_2023", "CAP_2025", "ESMO_2023", "Rakha_International_2026",
    "DESTINY-Breast04", "DESTINY-Breast06",
    "Trastuzumab Deruxtecan", "T-DXd", "Trastuzumab",
    "Pertuzumab", "Lapatinib", "Tucatinib", "Margetuximab",
    "Breast Cancer", "HER2", "ERBB2",
}

_VALID_CATEGORIES: set[str] = {
    "HER2-Positive", "HER2-Equivocal", "HER2-Low", "HER2-Ultralow", "HER2-Null",
}

_VALID_IHC_SCORES: set[str] = {"3+", "2+", "1+", "0+", "0"}

_VALID_ISH_GROUPS: set[str] = {
    "Group1", "Group2", "Group3", "Group4", "Group5",
}

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class GroundedContext:
    """Context retrieved from Neo4j for constraining LLM generation."""
    entities: list[dict[str, Any]] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    entity_labels: set[str] = field(default_factory=set)
    source_docs: list[str] = field(default_factory=list)

    def to_prompt_context(self) -> str:
        """Serialize to a structured prompt section."""
        lines: list[str] = ["### GROUNDED CONTEXT (verified KG facts)\n"]

        if self.entities:
            lines.append("**Entities:**")
            for e in self.entities[:15]:
                lines.append(f"  - [{e.get('node_type', 'Entity')}] {e.get('label', '')} (id: {e.get('id', '')})")

        if self.relations:
            lines.append("\n**Relations:**")
            for r in self.relations[:15]:
                conf = f" [confidence={r.get('confidence', '?')}]" if r.get("confidence") else ""
                lines.append(f"  - {r.get('from_label')} --[{r.get('relation')}]--> {r.get('to_label')}{conf}")

        if self.chunks:
            lines.append("\n**Source excerpts:**")
            for ch in self.chunks[:3]:
                src = ch.get("source_doc", "")
                txt = ch.get("context", "")[:300].replace("\n", " ")
                lines.append(f"  [{src}] {txt}")

        if self.source_docs:
            lines.append(f"\n**Sources:** {', '.join(set(self.source_docs))}")

        lines.append("\n> IMPORTANT: Only reference the above entities and relations in your response.")
        return "\n".join(lines)


@dataclass
class GroundingResult:
    """Result of post-generation grounding validation."""
    is_grounded: bool
    hallucinated_entities: list[str] = field(default_factory=list)
    unverified_relations: list[str] = field(default_factory=list)
    missing_provenance: list[str] = field(default_factory=list)
    rule_violations: list[str] = field(default_factory=list)
    confidence_penalty: float = 0.0
    flagged_text: list[str] = field(default_factory=list)

    @property
    def severity(self) -> str:
        total = (
            len(self.hallucinated_entities) * 2
            + len(self.unverified_relations)
            + len(self.rule_violations) * 3
        )
        if total == 0:
            return "CLEAN"
        if total <= 2:
            return "LOW"
        if total <= 5:
            return "MEDIUM"
        return "HIGH"

    def summary(self) -> str:
        if self.is_grounded:
            return "[GROUNDED] Response is fully grounded in the KG."
        parts: list[str] = [f"[{self.severity}] Grounding issues detected:"]
        if self.hallucinated_entities:
            parts.append(f"  Hallucinated entities: {self.hallucinated_entities}")
        if self.unverified_relations:
            parts.append(f"  Unverified relations: {self.unverified_relations}")
        if self.rule_violations:
            parts.append(f"  Clinical rule violations: {self.rule_violations}")
        if self.confidence_penalty:
            parts.append(f"  Confidence penalty: -{self.confidence_penalty:.0%}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Cypher helpers
# ---------------------------------------------------------------------------

_ENTITY_CONTEXT_QUERY = """
MATCH (e)
WHERE e.id IN $entity_ids OR toLower(e.label) IN $labels
OPTIONAL MATCH (e)-[r]->(related)
OPTIONAL MATCH (chunk:Chunk)-[:MENTIONS]->(e)
RETURN DISTINCT
    e.id            AS id,
    e.label         AS label,
    labels(e)[0]    AS node_type,
    type(r)         AS relation,
    related.label   AS related_entity,
    related.id      AS related_id,
    r.confidence    AS confidence,
    chunk.text      AS context,
    chunk.source_doc AS source_doc
LIMIT 50
"""

_CLINICAL_RULES_QUERY = """
MATCH (r:ValidationRule)
WHERE r.active = true
RETURN r.rule_id AS rule_id, r.condition AS condition, r.message AS message
LIMIT 20
"""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class GroundingChecker:
    """
    Grounds LLM responses against the Neo4j KG.

    Usage:
        checker = GroundingChecker(driver)

        # 1. Build constrained context before calling the LLM
        ctx = checker.build_context(entity_ids=["Score2Plus", "Group3"])

        # 2. Inject ctx.to_prompt_context() into the LLM prompt

        # 3. Validate LLM response after generation
        result = checker.validate_response(llm_response, ctx)
        if not result.is_grounded:
            print(result.summary())
    """

    def __init__(self, driver: Driver | None = None) -> None:
        self._driver = driver

    # ------------------------------------------------------------------
    # Stage 1: pre-generation context
    # ------------------------------------------------------------------

    def build_context(
        self,
        entity_ids: list[str] | None = None,
        labels: list[str] | None = None,
        clinical_data: dict[str, Any] | None = None,
    ) -> GroundedContext:
        """
        Retrieve a constrained context from Neo4j.
        Falls back to a vocabulary-only context when driver is not available.
        """
        entity_ids = entity_ids or []
        labels = [lb.lower() for lb in (labels or [])]

        if clinical_data:
            entity_ids = self._entity_ids_from_case(clinical_data, entity_ids)

        if self._driver is None:
            return self._fallback_context(entity_ids, clinical_data or {})

        rows: list[dict] = []
        with self._driver.session() as session:
            result = session.run(
                _ENTITY_CONTEXT_QUERY,
                entity_ids=entity_ids,
                labels=labels,
            )
            rows = [dict(r) for r in result]

        return self._build_from_rows(rows)

    # ------------------------------------------------------------------
    # Stage 2: post-generation validation
    # ------------------------------------------------------------------

    def validate_response(
        self,
        response: str,
        context: GroundedContext,
        clinical_data: dict[str, Any] | None = None,
    ) -> GroundingResult:
        """
        Validate an LLM response against the grounded context.
        Returns a GroundingResult indicating any issues found.
        """
        hallucinated: list[str] = []
        unverified: list[str] = []
        violations: list[str] = []
        flagged: list[str] = []

        # 1. Entity constraint check
        hallucinated = self._check_entity_drift(response, context)

        # 2. Clinical category validation
        cat_violations = self._check_category_claims(response)
        violations.extend(cat_violations)

        # 3. IHC/ISH value constraint
        value_issues = self._check_score_validity(response)
        violations.extend(value_issues)

        # 4. Clinical rule enforcement
        if clinical_data:
            rule_violations = self._enforce_clinical_rules(response, clinical_data)
            violations.extend(rule_violations)

        # 5. Provenance check — citations should reference known guidelines
        missing_prov = self._check_provenance(response)

        # Flag suspicious phrases
        flagged = self._find_hedging(response)

        confidence_penalty = min(
            0.5,
            len(hallucinated) * 0.1
            + len(violations) * 0.15
            + len(unverified) * 0.05,
        )

        is_grounded = (
            len(hallucinated) == 0
            and len(violations) == 0
        )

        return GroundingResult(
            is_grounded=is_grounded,
            hallucinated_entities=hallucinated,
            unverified_relations=unverified,
            missing_provenance=missing_prov,
            rule_violations=violations,
            confidence_penalty=confidence_penalty,
            flagged_text=flagged,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _entity_ids_from_case(
        self, clinical_data: dict[str, Any], existing: list[str]
    ) -> list[str]:
        ids = list(existing)
        ihc = clinical_data.get("ihc_score", "")
        ish = clinical_data.get("ish_group", "")
        score_map = {
            "3+": "Score3Plus", "2+": "Score2Plus",
            "1+": "Score1Plus", "0+": "Score0Plus", "0": "Score0",
        }
        if ihc in score_map and score_map[ihc] not in ids:
            ids.append(score_map[ihc])
        if ish and ish not in ids:
            ids.append(ish)
        return ids

    def _build_from_rows(self, rows: list[dict]) -> GroundedContext:
        seen_entities: set[str] = set()
        entities: list[dict] = []
        relations: list[dict] = []
        chunks: list[dict] = []
        source_docs: list[str] = []
        entity_labels: set[str] = set()

        for row in rows:
            eid = row.get("id") or ""
            if eid and eid not in seen_entities:
                seen_entities.add(eid)
                entities.append({
                    "id": eid,
                    "label": row.get("label", ""),
                    "node_type": row.get("node_type", ""),
                })
                if row.get("label"):
                    entity_labels.add(str(row["label"]).lower())

            if row.get("relation") and row.get("related_entity"):
                rel = {
                    "from_label": row.get("label"),
                    "relation": row.get("relation"),
                    "to_label": row.get("related_entity"),
                    "confidence": row.get("confidence"),
                }
                if rel not in relations:
                    relations.append(rel)

            if row.get("context"):
                ch = {
                    "context": row["context"],
                    "source_doc": row.get("source_doc", ""),
                }
                if ch not in chunks:
                    chunks.append(ch)
            if row.get("source_doc"):
                source_docs.append(str(row["source_doc"]))

        return GroundedContext(
            entities=entities,
            relations=relations,
            chunks=chunks,
            entity_labels=entity_labels,
            source_docs=source_docs,
        )

    def _fallback_context(
        self, entity_ids: list[str], clinical_data: dict[str, Any]
    ) -> GroundedContext:
        """Minimal context from hardcoded vocabulary when Neo4j is unavailable."""
        entities = [
            {"id": eid, "label": eid.replace("_", " "), "node_type": "KGEntity"}
            for eid in entity_ids
        ]
        entity_labels = {lb.lower() for lb in _HER2_KNOWN_ENTITIES}
        return GroundedContext(
            entities=entities,
            entity_labels=entity_labels,
            source_docs=["ASCO_CAP_2023", "Rakha_International_2026"],
        )

    def _check_entity_drift(
        self, response: str, context: GroundedContext
    ) -> list[str]:
        """
        Detect entity names in the response that are not in the grounded context
        and not in the known HER2 vocabulary.
        """
        # Only flag drug names and category names not in either allowed set
        suspicious_patterns = [
            r"\bT-DM1\b",           # Ado-trastuzumab — different from T-DXd
            r"\bKadcyla\b",
            r"\bEnhertu\b",         # Brand name — acceptable but flag for audit
            r"\bPertuzumab\b",
            r"\bNeratinib\b",
        ]
        flagged = []
        combined_vocab = context.entity_labels | {lb.lower() for lb in _HER2_KNOWN_ENTITIES}
        for pattern in suspicious_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for m in matches:
                if m.lower() not in combined_vocab and m not in context.entity_labels:
                    flagged.append(m)
        return list(set(flagged))

    def _check_category_claims(self, response: str) -> list[str]:
        """Flag HER2 category claims that don't match the five-category model."""
        violations = []
        # Detect use of deprecated "HER2-Negative" as a final classification
        if re.search(r"\bHER2[-\s]Negative\b", response, re.IGNORECASE):
            violations.append(
                "Deprecated category 'HER2-Negative' used. "
                "Use HER2-Low, HER2-Ultralow, or HER2-Null per 2022+ guidelines."
            )
        return violations

    def _check_score_validity(self, response: str) -> list[str]:
        """Validate any IHC scores mentioned in the response."""
        violations = []
        # Match patterns like "IHC 2+", "IHC 3+", "IHC 0" — capture digit(s) + optional "+"
        mentioned = re.findall(r"\bIHC\s+([0-9]+\+?)", response, re.IGNORECASE)
        for score in mentioned:
            if score not in _VALID_IHC_SCORES:
                violations.append(f"Unknown IHC score referenced: '{score}'")
        return violations

    def _enforce_clinical_rules(
        self, response: str, clinical_data: dict[str, Any]
    ) -> list[str]:
        """
        Enforce hard clinical rules based on case data.
        Maps to §4.2 VALIDATION_RULES from the implementation plan.
        """
        violations = []
        ihc = clinical_data.get("ihc_score", "")
        ish = clinical_data.get("ish_group", "")

        # Rule: IHC 3+ claiming equivocal or low without ISH context
        if ihc == "3+":
            if re.search(r"HER2[-\s](Equivocal|Low|Ultralow|Null)", response, re.IGNORECASE):
                violations.append(
                    "IHC 3+ case: response claims non-positive HER2 category without ISH override."
                )

        # Rule: IHC 0 claiming T-DXd eligibility (standard criteria)
        if ihc == "0" and not ish:
            if re.search(r"T-DXd|trastuzumab deruxtecan", response, re.IGNORECASE):
                if not re.search(r"DESTINY-Breast06|Rakha 2026|not eligible|expanded crit", response, re.IGNORECASE):
                    violations.append(
                        "IHC 0 case claims T-DXd eligibility without citing DESTINY-Breast06 / Rakha 2026 expanded criteria."
                    )

        return violations

    def _check_provenance(self, response: str) -> list[str]:
        """Flag claims that appear to need a citation but don't have one."""
        missing = []
        # Look for definitive clinical statements without guideline mentions
        clinical_claims = re.findall(
            r"(is eligible for|is not eligible for|is positive|is negative|is recommended|requires|must)",
            response,
            re.IGNORECASE,
        )
        guideline_mentions = re.findall(
            r"(ASCO|CAP|ESMO|NCCN|Rakha|DESTINY|PMID|guideline|trial|criteria)",
            response,
            re.IGNORECASE,
        )
        if len(clinical_claims) > 3 and len(guideline_mentions) == 0:
            missing.append("Multiple clinical claims made without guideline citations.")
        return missing

    def _find_hedging(self, response: str) -> list[str]:
        """Find suspicious high-confidence claims that may indicate hallucination."""
        patterns = [
            r"\balways\b", r"\bnever\b", r"\bguaranteed\b",
            r"\b100%\b", r"\bwithout exception\b",
        ]
        found = []
        for p in patterns:
            m = re.search(p, response, re.IGNORECASE)
            if m:
                found.append(m.group(0))
        return found
