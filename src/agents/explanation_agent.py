"""
Explanation Agent — generates human-readable explanations of diagnostic reasoning.

Implements the Explanation Agent role from §7.2 of the implementation plan.
Uses tools: get_diagnostic_pathway, get_scoring_criteria, explain_fractal_correlation.
"""
from __future__ import annotations

from typing import Any

from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

if TYPE_CHECKING:
    from neo4j import Driver

from src.agents.state import HER2AgentState

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_EXPLANATION_SYSTEM_PROMPT = """You are a specialized HER2 Explanation Agent.
Your role is to provide clear, step-by-step explanations of HER2 diagnostic reasoning —
suitable for pathologists, oncologists, and informed patients.

You translate technical algorithm traversals and KG facts into plain clinical narrative.

When explaining a diagnostic pathway:
1. State the initial test result and its clinical significance.
2. Walk through each decision point in the algorithm (IHC → ISH if applicable).
3. Explain *why* each decision threshold exists (scientific rationale).
4. Identify the final classification and its therapeutic implications.
5. Note any fractal biomarker correlations if provided.

When explaining a specific concept (e.g., "Why is IHC 2+ equivocal?"):
1. Define the concept in clinical terms.
2. Explain the biological basis.
3. Describe the algorithmic consequence.
4. Give a clinical example.

Always end with a "Key Takeaways" section (3–5 bullet points).
"""

# Cypher to fetch scoring criteria nodes
_SCORING_CRITERIA_QUERY = """
MATCH (s:IHCScore)
RETURN s.id AS id, s.label AS label, s.score AS score,
       s.staining_pattern AS staining_pattern,
       s.percentage_threshold AS percentage_threshold
ORDER BY s.score DESC
"""

_FRACTAL_CORRELATION_QUERY = """
MATCH (fm:FractalMetric)-[r:proposedEquivalence]->(cc:ClinicalCategory)
RETURN fm.label AS fractal_metric, fm.id AS fm_id,
       r.correlation_strength AS correlation, r.p_value AS p_value,
       cc.id AS category, cc.label AS category_label
ORDER BY r.correlation_strength DESC
"""


class ExplanationAgent:
    """Explanation agent node for the LangGraph multi-agent pipeline."""

    def __init__(self, llm: BaseChatModel, driver: Driver) -> None:
        self._llm = llm
        self._driver = driver

    def __call__(self, state: HER2AgentState) -> dict[str, Any]:
        """Execute the explanation agent and return state update."""
        query = state.get("query", "")
        clinical_data = state.get("clinical_data", {})
        prior_results = state.get("agent_results", [])

        # 1. Fetch scoring criteria and fractal correlations from KG
        scoring_criteria = self._get_scoring_criteria()
        fractal_correlations = self._get_fractal_correlations()

        # 2. Extract diagnostic pathway from prior Diagnostic Agent result (if any)
        pathway_text = ""
        for r in prior_results:
            if r.get("agent") == "diagnostic":
                cls = r.get("classification", {})
                steps = cls.get("pathway_steps", [])
                if steps:
                    pathway_text = " → ".join(steps)
                break

        # 3. Build grounded context
        criteria_text = "\n".join(
            f"  {c.get('label', c.get('id', ''))}: score={c.get('score')}, "
            f"pattern={c.get('staining_pattern')}, threshold={c.get('percentage_threshold')}"
            for c in scoring_criteria
        ) or "  (KG not yet populated)"

        fractal_text = "\n".join(
            f"  {c.get('fractal_metric')} ↔ {c.get('category_label')}: "
            f"r={c.get('correlation')}, p={c.get('p_value')}"
            for c in fractal_correlations[:5]
        ) or "  (No fractal correlations in KG)"

        context_text = (
            f"User query: {query}\n\n"
            f"Clinical data: {clinical_data}\n\n"
            f"Diagnostic pathway (from prior agent): {pathway_text or 'N/A'}\n\n"
            f"IHC Scoring Criteria from KG:\n{criteria_text}\n\n"
            f"Fractal–Clinical Correlations from KG:\n{fractal_text}\n\n"
            f"Prior agent outputs: {[r.get('agent') for r in prior_results]}"
        )
        messages = [
            SystemMessage(content=_EXPLANATION_SYSTEM_PROMPT),
            HumanMessage(content=context_text),
        ]
        llm_response = self._llm.invoke(messages)
        narrative = llm_response.content if hasattr(llm_response, "content") else str(llm_response)

        result = {
            "agent": "explanation",
            "scoring_criteria": scoring_criteria,
            "fractal_correlations": fractal_correlations[:5],
            "pathway_text": pathway_text,
            "narrative": narrative,
        }
        return {
            "current_agent": "explanation",
            "agent_results": [result],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_scoring_criteria(self) -> list[dict]:
        try:
            with self._driver.session() as session:
                return [dict(r) for r in session.run(_SCORING_CRITERIA_QUERY)]
        except Exception:
            return []

    def _get_fractal_correlations(self) -> list[dict]:
        try:
            with self._driver.session() as session:
                return [dict(r) for r in session.run(_FRACTAL_CORRELATION_QUERY)]
        except Exception:
            return []
