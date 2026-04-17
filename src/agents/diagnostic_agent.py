"""
Diagnostic Agent — traverses IHC/ISH decision trees to classify HER2 status.

Implements the Diagnostic Agent role from §7.2 of the implementation plan.
Uses tools: execute_cypher, traverse_decision_tree, get_ihc_algorithm, get_ish_algorithm.
"""
from __future__ import annotations

from typing import Any

from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

if TYPE_CHECKING:
    from neo4j import Driver

from src.agents.state import HER2AgentState
from src.retrieval.pathway_retriever import PathwayRetriever

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_DIAGNOSTIC_SYSTEM_PROMPT = """You are a specialized HER2 Diagnostic Agent with expertise in
ASCO/CAP 2023 guidelines, the Rakha International 2026 scoring matrix, and ISH group interpretation.

Your role:
1. Given clinical test data (IHC score, ISH group, ratio, signals/cell), determine the HER2 classification.
2. Report the full decision pathway (which nodes were traversed).
3. Cite the applicable guideline.
4. Flag any cases requiring reflex ISH testing.
5. Identify HER2-Low (IHC 1+ or IHC 2+/ISH-) and HER2-Ultralow (IHC 0+) subcategories.

Always structure your response as:
- **Classification:** [HER2-Positive | HER2-Equivocal | HER2-Low | HER2-Ultralow | HER2-Null]
- **IHC Score:** [0 | 0+ | 1+ | 2+ | 3+]
- **ISH Interpretation:** (if applicable)
- **Decision Pathway:** Step-by-step listing of algorithm nodes traversed
- **Applicable Guideline:** ASCO/CAP 2023 / Rakha 2026 / ESMO 2023
- **Action Required:** (e.g., reflex ISH if IHC 2+)
- **Confidence:** HIGH | MEDIUM | LOW

If data is insufficient for a definitive classification, state what additional testing is needed.
"""


def _classify_from_data(clinical_data: dict[str, Any]) -> dict[str, Any]:
    """
    Rule-based classification from structured clinical data.

    This provides a fast, deterministic classification that the LLM
    then narrates.  Returns a dict with classification, pathway_steps,
    action_required, confidence, and applicable_guideline.
    """
    ihc = clinical_data.get("ihc_score", "")
    ish_group = clinical_data.get("ish_group")
    ish_ratio = clinical_data.get("ish_ratio")
    signals = clinical_data.get("signals_per_cell")

    pathway: list[str] = []

    # ── IHC 3+ ────────────────────────────────────────────────────────────
    if ihc in ("3+", "Score3Plus", "IHC3Plus"):
        pathway = ["IHC_ENTRY", "IHC_NODE1 → YES", "Score3Plus"]
        return {
            "classification": "HER2_Positive",
            "ihc_score": "3+",
            "pathway_steps": pathway,
            "action_required": None,
            "confidence": "HIGH",
            "applicable_guideline": "ASCO_CAP_2023",
        }

    # ── IHC 2+ (Equivocal) ────────────────────────────────────────────────
    if ihc in ("2+", "Score2Plus", "IHC2Plus"):
        pathway = ["IHC_ENTRY", "IHC_NODE1 → NO", "IHC_NODE2 → YES", "Score2Plus"]
        if ish_group is None:
            return {
                "classification": "HER2_Equivocal",
                "ihc_score": "2+",
                "pathway_steps": pathway,
                "action_required": "Reflex ISH testing required (ASCO/CAP 2023 mandate)",
                "confidence": "MEDIUM",
                "applicable_guideline": "ASCO_CAP_2023",
            }
        # ISH provided — classify by group
        return _classify_ish(ish_group, ish_ratio, signals, pathway)

    # ── IHC 1+ ────────────────────────────────────────────────────────────
    if ihc in ("1+", "Score1Plus", "IHC1Plus"):
        pathway = ["IHC_ENTRY", "IHC_NODE1 → NO", "IHC_NODE2 → NO", "IHC_NODE3 → YES", "Score1Plus"]
        return {
            "classification": "HER2_Low",
            "ihc_score": "1+",
            "pathway_steps": pathway,
            "action_required": "Consider T-DXd eligibility assessment (DESTINY-Breast06)",
            "confidence": "HIGH",
            "applicable_guideline": "ASCO_CAP_2023",
        }

    # ── IHC 0+ (Ultralow — Rakha 2026) ────────────────────────────────────
    if ihc in ("0+", "Score0Plus", "IHC0Plus"):
        pathway = ["IHC_ENTRY", "IHC_NODE1 → NO", "IHC_NODE2 → NO", "IHC_NODE3 → NO",
                   "IHC_NODE4 → YES", "Score0Plus"]
        return {
            "classification": "HER2_Ultralow",
            "ihc_score": "0+",
            "pathway_steps": pathway,
            "action_required": "See Rakha 2026 / DESTINY-Breast06 expanded criteria",
            "confidence": "HIGH",
            "applicable_guideline": "Rakha_International_2026",
        }

    # ── IHC 0 (Null) ──────────────────────────────────────────────────────
    if ihc in ("0", "Score0", "IHC0"):
        pathway = ["IHC_ENTRY", "IHC_NODE1 → NO", "IHC_NODE2 → NO", "IHC_NODE3 → NO",
                   "IHC_NODE4 → NO", "IHC_NODE5 → terminal", "Score0"]
        return {
            "classification": "HER2_Null",
            "ihc_score": "0",
            "pathway_steps": pathway,
            "action_required": None,
            "confidence": "HIGH",
            "applicable_guideline": "ASCO_CAP_2023",
        }

    # Unknown input
    return {
        "classification": "UNKNOWN",
        "ihc_score": ihc,
        "pathway_steps": [],
        "action_required": "Insufficient data — provide IHC score (0, 0+, 1+, 2+, 3+)",
        "confidence": "LOW",
        "applicable_guideline": None,
    }


def _classify_ish(
    ish_group: str,
    ish_ratio: float | None,
    signals: float | None,
    prior_pathway: list[str],
) -> dict[str, Any]:
    """Apply ISH group algorithm; continues from a prior IHC 2+ pathway."""
    g = str(ish_group).replace("Group", "").strip()

    # ISH Group 1: Positive
    if g == "1":
        return {
            "classification": "HER2_Positive",
            "ihc_score": "2+",
            "ish_group": ish_group,
            "pathway_steps": prior_pathway + ["ISH_GROUP1_RESULT → HER2_Positive"],
            "action_required": None,
            "confidence": "HIGH",
            "applicable_guideline": "ASCO_CAP_2023",
        }
    # ISH Group 5: Negative
    if g == "5":
        return {
            "classification": "HER2_Low",
            "ihc_score": "2+",
            "ish_group": ish_group,
            "pathway_steps": prior_pathway + ["ISH_GROUP5_RESULT → HER2_Negative → HER2_Low"],
            "action_required": "Consider T-DXd eligibility (IHC 2+/ISH-)",
            "confidence": "HIGH",
            "applicable_guideline": "ASCO_CAP_2023",
        }
    # Groups 2–4: indeterminate, depends on recount/IHC correlation
    return {
        "classification": "HER2_Equivocal",
        "ihc_score": "2+",
        "ish_group": ish_group,
        "pathway_steps": prior_pathway + [f"ISH_NODE_G{g}_ENTRY → workup required"],
        "action_required": f"Group {g} workup: correlate with IHC and recount per ASCO/CAP Comment-A",
        "confidence": "MEDIUM",
        "applicable_guideline": "ASCO_CAP_2023",
    }


# ---------------------------------------------------------------------------
# LangGraph node function
# ---------------------------------------------------------------------------

class DiagnosticAgent:
    """Diagnostic agent node for the LangGraph multi-agent pipeline."""

    def __init__(self, llm: BaseChatModel, driver: Driver) -> None:
        self._llm = llm
        self._retriever = PathwayRetriever(driver)

    def __call__(self, state: HER2AgentState) -> dict[str, Any]:
        """Execute the diagnostic agent and return state update."""
        clinical_data = state.get("clinical_data", {})
        query = state.get("query", "")

        # 1. Deterministic rule-based classification
        classification = _classify_from_data(clinical_data)

        # 2. Attempt KG-backed pathway retrieval
        ihc_score = classification.get("ihc_score", "")
        ish_group = clinical_data.get("ish_group")
        ihc_id_map = {
            "3+": "Score3Plus", "2+": "Score2Plus", "1+": "Score1Plus",
            "0+": "Score0Plus", "0": "Score0",
        }
        ihc_node_id = ihc_id_map.get(ihc_score)
        kg_pathway: dict[str, Any] = {}
        if ihc_node_id:
            try:
                kg_pathway = self._retriever.get_pathway(ihc_node_id, ish_group)
            except Exception:  # Neo4j not available — continue with rule-based
                pass

        # 3. LLM narration
        context_text = (
            f"Clinical data: {clinical_data}\n"
            f"Rule-based classification: {classification}\n"
            f"KG pathway: {kg_pathway}\n"
            f"User query: {query}"
        )
        messages = [
            SystemMessage(content=_DIAGNOSTIC_SYSTEM_PROMPT),
            HumanMessage(content=context_text),
        ]
        try:
            llm_response = self._llm.invoke(messages)
            narrative = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
        except Exception:
            # Fallback: deterministic text from classification dict
            cls = classification
            narrative = (
                f"**Classification:** {cls.get('classification','').replace('_','-')}\n"
                f"**IHC Score:** {cls.get('ihc_score','')}\n"
                f"**Confidence:** {cls.get('confidence','')}\n"
                f"**Guideline:** {cls.get('applicable_guideline','')}\n"
                f"**Pathway:** {' → '.join(cls.get('pathway_steps',[]))}\n"
                + (f"**Action required:** {cls.get('action_required')}\n" if cls.get('action_required') else "")
                + "*(LLM narration unavailable)*"
            )

        result = {
            "agent": "diagnostic",
            "classification": classification,
            "kg_pathway": kg_pathway,
            "narrative": narrative,
        }
        return {
            "current_agent": "diagnostic",
            "agent_results": [result],
        }
