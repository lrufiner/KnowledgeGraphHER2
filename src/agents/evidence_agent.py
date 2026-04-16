"""
Evidence Agent — retrieves guideline recommendations and trial data.

Implements the Evidence Agent role from §7.2 of the implementation plan.
Uses tools: vector_search_guidelines, get_guideline_recommendations.
"""
from __future__ import annotations

from typing import Any

from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

if TYPE_CHECKING:
    from neo4j import Driver

from src.agents.state import HER2AgentState
from src.retrieval.entity_retriever import EntityRetriever

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_EVIDENCE_SYSTEM_PROMPT = """You are a specialized HER2 Evidence Agent with expertise in
clinical trial data, guideline recommendations, and evidence-based oncology.

Your role:
1. Answer questions about guideline recommendations (ASCO/CAP 2023, ESMO 2023, NCCN, Rakha 2026).
2. Summarise key clinical trial data (DESTINY-Breast04, DESTINY-Breast06, ToGA, etc.).
3. Assess treatment eligibility for HER2-targeted therapies (T-DXd, pertuzumab, trastuzumab).
4. Provide source citations with guideline section references.

Always structure your response as:
- **Answer:** Direct answer to the evidence question
- **Supporting Evidence:** Key data points (trial names, endpoints, p-values, ORR/PFS if known)
- **Applicable Guidelines:** Which guidelines address this
- **Limitations / Caveats:** Important caveats, subgroup analyses, pending approvals
- **Citations:** [GuidelineID § section] or [TrialName, PMID]
"""

# ---------------------------------------------------------------------------
# Cypher queries for evidence retrieval
# ---------------------------------------------------------------------------

_EVIDENCE_QUERY = """
MATCH (chunk:Chunk)
WHERE chunk.content_type IN ['guideline', 'clinical_trial', 'evidence']
  AND toLower(chunk.text) CONTAINS toLower($keyword)
RETURN chunk.text        AS context,
       chunk.source_doc  AS source_doc,
       chunk.content_type AS content_type
ORDER BY chunk.source_doc
LIMIT 10
"""

_THERAPEUTIC_ELIGIBILITY_QUERY = """
MATCH (cat:ClinicalCategory)-[r:eligibleFor|notEligibleFor]->(drug:TherapeuticAgent)
WHERE cat.id IN $category_ids OR $category_ids = []
RETURN cat.label      AS category,
       type(r)        AS eligibility,
       drug.label     AS drug,
       r.clinical_context AS context,
       r.trial_evidence AS trial
ORDER BY cat.label, drug.label
"""

_GUIDELINE_CITATIONS_QUERY = """
MATCH (g:Guideline)-[:covers]->(topic)
WHERE toLower(g.label) CONTAINS toLower($keyword)
   OR toLower(topic.label) CONTAINS toLower($keyword)
RETURN g.id AS guideline_id, g.label AS guideline_label, g.year AS year,
       topic.label AS topic
ORDER BY g.year DESC
LIMIT 10
"""


class EvidenceAgent:
    """Evidence retrieval agent node for the LangGraph multi-agent pipeline."""

    def __init__(self, llm: BaseChatModel, driver: Driver) -> None:
        self._llm = llm
        self._driver = driver
        self._entity_retriever = EntityRetriever(driver)

    def __call__(self, state: HER2AgentState) -> dict[str, Any]:
        """Execute the evidence agent and return state update."""
        query = state.get("query", "")
        clinical_data = state.get("clinical_data", {})

        # 1. Extract keywords from the query for KG lookup
        keywords = self._extract_keywords(query, clinical_data)

        # 2. Retrieve relevant evidence chunks and entity context
        evidence_rows: list[dict] = []
        eligibility_rows: list[dict] = []
        for keyword in keywords[:3]:  # limit to top 3 to avoid noise
            rows = self._query_evidence(keyword)
            evidence_rows.extend(rows)

        # 3. Retrieve therapeutic eligibility info if classification is known
        prior_results = state.get("agent_results", [])
        category_ids = self._extract_categories(prior_results, clinical_data)
        if category_ids:
            eligibility_rows = self._query_eligibility(category_ids)

        # 4. LLM narration with grounded context
        context_text = (
            f"User query: {query}\n\n"
            f"Clinical context: {clinical_data}\n\n"
            f"Retrieved evidence from KG ({len(evidence_rows)} chunks):\n"
            + "\n---\n".join(
                f"[{r.get('source_doc', 'Unknown')}] {r.get('context', '')[:400]}"
                for r in evidence_rows[:5]
            )
            + "\n\nTreatment eligibility from KG:\n"
            + "\n".join(
                f"  {r.get('category')} {r.get('eligibility')} {r.get('drug')} — {r.get('context', '')}"
                for r in eligibility_rows[:5]
            )
        )
        messages = [
            SystemMessage(content=_EVIDENCE_SYSTEM_PROMPT),
            HumanMessage(content=context_text),
        ]
        llm_response = self._llm.invoke(messages)
        narrative = llm_response.content if hasattr(llm_response, "content") else str(llm_response)

        # Build citations list
        citations = [
            {"source": r.get("source_doc", ""), "type": r.get("content_type", ""), "text": r.get("context", "")[:150]}
            for r in evidence_rows[:5]
        ]

        result = {
            "agent": "evidence",
            "evidence_rows": evidence_rows[:5],
            "eligibility_rows": eligibility_rows[:5],
            "narrative": narrative,
            "citations": citations,
        }
        return {
            "current_agent": "evidence",
            "agent_results": [result],
            "citations": citations,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_keywords(self, query: str, clinical_data: dict) -> list[str]:
        """Extract meaningful keywords from the query and clinical data."""
        keywords: list[str] = []
        # Known clinical terms to look for (case-insensitive)
        for term in [
            "T-DXd", "trastuzumab deruxtecan", "pertuzumab", "trastuzumab",
            "DESTINY", "HER2-low", "HER2-ultralow", "IHC", "ISH",
            "ASCO", "CAP", "ESMO", "NCCN", "Rakha",
        ]:
            if term.lower() in query.lower():
                keywords.append(term)
        # Add the query itself as a fallback
        keywords.append(query[:80])
        return keywords

    def _extract_categories(self, prior_results: list[dict], clinical_data: dict) -> list[str]:
        """Extract ClinicalCategory IDs from prior agent results."""
        category_ids: list[str] = []
        for r in prior_results:
            cls_info = r.get("classification", {})
            cls = cls_info.get("classification") if isinstance(cls_info, dict) else None
            if cls:
                category_ids.append(cls)
        # Also check clinical_data directly
        if "category" in clinical_data:
            category_ids.append(clinical_data["category"])
        return list(set(category_ids))

    def _query_evidence(self, keyword: str) -> list[dict]:
        try:
            with self._driver.session() as session:
                result = session.run(_EVIDENCE_QUERY, keyword=keyword)
                return [dict(r) for r in result]
        except Exception:
            return []

    def _query_eligibility(self, category_ids: list[str]) -> list[dict]:
        try:
            with self._driver.session() as session:
                result = session.run(_THERAPEUTIC_ELIGIBILITY_QUERY, category_ids=category_ids)
                return [dict(r) for r in result]
        except Exception:
            return []
