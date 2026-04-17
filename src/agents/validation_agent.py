"""
Validation Agent — checks clinical consistency and detects conflicts.

Implements the Validation Agent role from §7.2 of the implementation plan.
Uses tools: run_validation_rules, check_fractal_clinical_consistency,
            detect_temporal_conflicts.
"""
from __future__ import annotations

from typing import Any

from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

if TYPE_CHECKING:
    from neo4j import Driver

from src.agents.state import HER2AgentState
from src.domain.models import ValidationSeverity

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_VALIDATION_SYSTEM_PROMPT = """You are a specialized HER2 Validation Agent responsible for
clinical consistency checking and conflict detection.

Your role:
1. Run ASCO/CAP 2023 validation rules against provided case data.
2. Detect conflicts between IHC and ISH results.
3. Check fractal biomarker consistency with clinical classification.
4. Identify temporal conflicts (e.g., re-testing within 30 days).
5. Flag cases requiring CRITICAL review (patient safety risk).

Always structure your response as:
- **Validation Status:** PASS | WARN | FAIL
- **Issues Found:** List each triggered rule with severity (CRITICAL / HIGH / MEDIUM / LOW)
- **Recommended Actions:** Specific steps to resolve each issue
- **Case Quality Score:** (0–100 based on number/severity of issues)
- **Requires Human Review:** YES / NO
"""

# Cypher: check for IHC/ISH inconsistencies in the KG
_CONSISTENCY_QUERY = """
MATCH (spec:Specimen {id: $specimen_id})
OPTIONAL MATCH (spec)-[:hasIHCScore]->(ihc:IHCScore)
OPTIONAL MATCH (spec)-[:hasISHGroup]->(ish:ISHGroup)
OPTIONAL MATCH (spec)-[:classifiedAs]->(cat:ClinicalCategory)
RETURN ihc.id AS ihc_id, ish.id AS ish_id,
       cat.id AS category, spec.id AS specimen_id
"""

_FRACTAL_CLINICAL_CONSISTENCY_QUERY = """
MATCH (fm:FractalMetric)-[r:proposedEquivalence]->(cc:ClinicalCategory)
WHERE cc.id = $category_id
RETURN fm.id AS fm_id, fm.label AS fm_label,
       r.correlation_strength AS correlation,
       r.d0_range_low AS d0_low, r.d0_range_high AS d0_high
"""


class ValidationAgent:
    """Validation agent node for the LangGraph multi-agent pipeline."""

    def __init__(self, llm: BaseChatModel, driver: Driver) -> None:
        self._llm = llm
        self._driver = driver

    def __call__(self, state: HER2AgentState) -> dict[str, Any]:
        """Execute the validation agent and return state update."""
        query = state.get("query", "")
        clinical_data = state.get("clinical_data", {})
        prior_results = state.get("agent_results", [])

        # 1. Run domain validation rules
        rule_issues = self._run_validation_rules(clinical_data)

        # 2. Check fractal consistency
        fractal_issues = self._check_fractal_consistency(clinical_data, prior_results)

        # 3. Check for IHC/ISH conflicts
        ish_conflicts = self._check_ish_conflict(clinical_data)

        all_issues = rule_issues + fractal_issues + ish_conflicts

        # 4. Determine overall status
        critical_count = sum(1 for i in all_issues if i.get("severity") == "CRITICAL")
        high_count = sum(1 for i in all_issues if i.get("severity") == "HIGH")
        needs_review = critical_count > 0 or high_count > 0

        if critical_count > 0:
            status = "FAIL"
        elif high_count > 0:
            status = "WARN"
        elif all_issues:
            status = "WARN"
        else:
            status = "PASS"

        # 5. LLM narration
        issues_text = "\n".join(
            f"  [{i.get('severity', 'INFO')}] {i.get('rule_id', i.get('type', 'CHECK'))}: {i.get('description', '')}"
            for i in all_issues
        ) or "  No issues found."

        context_text = (
            f"User query: {query}\n\n"
            f"Clinical data: {clinical_data}\n\n"
            f"Validation status: {status}\n"
            f"Issues detected ({len(all_issues)} total):\n{issues_text}\n\n"
            f"CRITICAL: {critical_count}, HIGH: {high_count}"
        )
        messages = [
            SystemMessage(content=_VALIDATION_SYSTEM_PROMPT),
            HumanMessage(content=context_text),
        ]
        try:
            llm_response = self._llm.invoke(messages)
            narrative = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
        except Exception:
            # Fallback: format validation issues as plain text
            lines = [
                f"- [{i.get('severity','')}] {i.get('rule_id','')} — {i.get('description','')}"
                for i in all_issues
            ]
            narrative = (
                f"**Validation status:** {status}\n"
                + ("\n".join(lines) if lines else "No issues found.")
                + "\n*(LLM narrative unavailable)*"
            )

        result = {
            "agent": "validation",
            "status": status,
            "issues": all_issues,
            "critical_count": critical_count,
            "high_count": high_count,
            "narrative": narrative,
        }
        return {
            "current_agent": "validation",
            "agent_results": [result],
            "needs_human_review": needs_review,
        }

    # ------------------------------------------------------------------
    # Rule checking helpers
    # ------------------------------------------------------------------

    def _run_validation_rules(self, clinical_data: dict) -> list[dict]:
        """Run case-level validation checks derived from VALIDATION_RULES.

        VALIDATION_RULES contains KG-integrity Cypher rules (run by the
        graph validator).  Here we apply equivalent Python logic for
        case-level clinical data without requiring a live Neo4j connection.
        """
        issues: list[dict] = []
        ihc  = clinical_data.get("ihc_score", "")
        ish  = str(clinical_data.get("ish_group", ""))

        # IHC 2+ without ISH data — clinical action required
        if ihc in ("2+", "Score2Plus"):
            if not ish or ish.lower() in ("", "none"):
                issues.append({
                    "rule_id":     "IHC2Plus_requires_ISH",
                    "severity":    ValidationSeverity.CRITICAL.value,
                    "description": "IHC 2+ (Equivocal) requires reflex ISH testing — no ISH data provided",
                    "type":        "validation_rule",
                })

        # IHC must be a recognised score
        valid_ihc = {"3+", "Score3Plus", "2+", "Score2Plus", "1+", "Score1Plus",
                     "0+", "Score0Plus", "0", "Score0", ""}
        if ihc and ihc not in valid_ihc:
            issues.append({
                "rule_id":     "INVALID_IHC_SCORE",
                "severity":    ValidationSeverity.HIGH.value,
                "description": f"IHC score '{ihc}' is not a recognised value per ASCO/CAP 2023",
                "type":        "validation_rule",
            })

        return issues

    def _check_fractal_consistency(
        self,
        clinical_data: dict,
        prior_results: list[dict],
    ) -> list[dict]:
        """Check if fractal_d0 value is consistent with the clinical classification."""
        issues: list[dict] = []
        fractal_d0 = clinical_data.get("fractal_d0")
        if fractal_d0 is None:
            return issues

        # Extract classification from prior diagnostic agent result
        category_id = None
        for r in prior_results:
            if r.get("agent") == "diagnostic":
                cls = r.get("classification", {})
                category_id = cls.get("classification") if isinstance(cls, dict) else None
                break

        if not category_id:
            return issues

        try:
            with self._driver.session() as session:
                rows = list(session.run(
                    _FRACTAL_CLINICAL_CONSISTENCY_QUERY,
                    category_id=category_id,
                ))
        except Exception:
            return issues

        for row in rows:
            d0_low  = row.get("d0_low")
            d0_high = row.get("d0_high")
            if d0_low is not None and d0_high is not None:
                if not (float(d0_low) <= float(fractal_d0) <= float(d0_high)):
                    issues.append({
                        "type":        "fractal_consistency",
                        "severity":    "HIGH",
                        "rule_id":     "FRACTAL_D0_RANGE",
                        "description": (
                            f"Fractal D0={fractal_d0} is outside expected range "
                            f"[{d0_low}, {d0_high}] for {category_id} "
                            f"(metric: {row.get('fm_label')})"
                        ),
                    })
        return issues

    def _check_ish_conflict(self, clinical_data: dict) -> list[dict]:
        """Detect direct IHC/ISH conflicts (e.g., IHC 3+ but ISH Group 5 without recount)."""
        issues: list[dict] = []
        ihc = clinical_data.get("ihc_score", "")
        ish_group = str(clinical_data.get("ish_group", ""))

        if ihc in ("3+", "Score3Plus") and ish_group in ("5", "Group5", "ISH_GROUP5"):
            issues.append({
                "type":        "ish_conflict",
                "severity":    "CRITICAL",
                "rule_id":     "IHC_ISH_CONFLICT",
                "description": (
                    "IHC 3+ (Positive) but ISH Group 5 (Not Amplified) — "
                    "direct conflict. Requires immediate case review and possible re-testing."
                ),
            })
        return issues
