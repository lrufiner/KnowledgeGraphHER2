"""
Supervisor Agent — orchestrates the HER2 multi-agent system.

Implements the Supervisor pattern from §7.1–7.2 of the implementation plan.
Routes incoming queries to specialized agents, accumulates results, and
synthesises the final clinical response.

LangGraph graph:
    START → supervisor → (diagnostic | evidence | explanation | validation)*
          → synthesize → END
"""
from __future__ import annotations

import json
from typing import Any, Literal

from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

if TYPE_CHECKING:
    from neo4j import Driver

from src.agents.diagnostic_agent import DiagnosticAgent
from src.agents.evidence_agent import EvidenceAgent
from src.agents.explanation_agent import ExplanationAgent
from src.agents.state import EMPTY_STATE, HER2AgentState
from src.agents.validation_agent import ValidationAgent

# ---------------------------------------------------------------------------
# Routing prompt (§7.2)
# ---------------------------------------------------------------------------

_ROUTING_PROMPT = """Given the clinical query, select the most appropriate agent(s):

1. DIAGNOSTIC: Questions about HER2 classification given test results
   (e.g., "What is the HER2 status for IHC 2+, ISH ratio 1.8, 5.2 signals/cell?")

2. EVIDENCE: Questions about guideline recommendations, trial data, treatment eligibility
   (e.g., "Is T-DXd approved for HER2-ultralow?", "What does DESTINY-Breast06 say?")

3. EXPLANATION: Requests for reasoning chains, diagnostic pathway explanations
   (e.g., "Why is IHC 2+ equivocal?", "Explain the ISH Group 3 workup")

4. VALIDATION: Consistency checks, conflict detection, QA queries
   (e.g., "Is D0=1.9 consistent with IHC 0?", "Check this case against ASCO/CAP")

You may invoke multiple agents sequentially.
Respond with a JSON array of agent names in execution order, from:
["diagnostic", "evidence", "explanation", "validation"]

Only include agents that are clearly needed. Examples:
- Pure classification question → ["diagnostic"]
- Evidence + eligibility question → ["evidence"]
- "Explain why" question → ["explanation"]
- Consistency check with data → ["diagnostic", "validation"]
- Full clinical workup → ["diagnostic", "evidence", "explanation", "validation"]

Respond ONLY with the JSON array, no other text.
"""

_SYNTHESIS_SYSTEM_PROMPT = """You are a senior HER2 clinical specialist synthesising the outputs
of multiple specialized AI agents into a cohesive, clinically actionable response.

Your response must:
1. Synthesise findings from all agents into a single coherent narrative.
2. Highlight the most clinically important findings (esp. CRITICAL/HIGH validation issues).
3. Provide a clear recommendation with confidence level.
4. List all citations in a structured format.
5. Explicitly flag any case requiring human expert review.

Structure your response as:
## Summary
[2–3 sentence clinical summary]

## Findings
[Synthesised findings from all agents]

## Recommendation
[Clear clinical recommendation]

## Citations
[List of cited guidelines and trials]

## Quality / Confidence
[Overall confidence: HIGH | MEDIUM | LOW — with rationale]
"""

# Maximum routing iterations to prevent infinite loops
_MAX_ITERATIONS = 3

# ---------------------------------------------------------------------------
# Routing helper
# ---------------------------------------------------------------------------

def _route_query(
    llm: BaseChatModel,
    query: str,
    clinical_data: dict,
) -> list[str]:
    """Ask the LLM to decide which agents to invoke."""
    context = f"Query: {query}\nClinical data: {json.dumps(clinical_data) if clinical_data else 'None'}"
    messages = [
        SystemMessage(content=_ROUTING_PROMPT),
        HumanMessage(content=context),
    ]
    try:
        response = llm.invoke(messages)
        text = response.content if hasattr(response, "content") else str(response)
        # Extract JSON array from response
        text = text.strip()
        if text.startswith("["):
            agents: list[str] = json.loads(text)
        else:
            # Fallback: find the first [...]
            start = text.find("[")
            end   = text.rfind("]") + 1
            if start >= 0 and end > start:
                agents = json.loads(text[start:end])
            else:
                agents = ["diagnostic"]  # safe fallback
    except Exception:
        agents = ["diagnostic"]

    # Validate — only allow known agent names
    valid = {"diagnostic", "evidence", "explanation", "validation"}
    return [a for a in agents if a in valid] or ["diagnostic"]


# ---------------------------------------------------------------------------
# LangGraph node functions
# ---------------------------------------------------------------------------

def _make_supervisor_node(llm: BaseChatModel) -> Any:
    def supervisor_node(state: HER2AgentState) -> dict[str, Any]:
        query         = state.get("query", "")
        clinical_data = state.get("clinical_data", {})
        iteration     = state.get("iteration_count", 0)

        if iteration >= _MAX_ITERATIONS:
            return {"target_agents": [], "iteration_count": iteration + 1}

        target_agents = _route_query(llm, query, clinical_data)
        return {
            "target_agents":    target_agents,
            "iteration_count":  iteration + 1,
        }
    return supervisor_node


def _make_synthesize_node(llm: BaseChatModel) -> Any:
    def synthesize_node(state: HER2AgentState) -> dict[str, Any]:
        query         = state.get("query", "")
        agent_results = state.get("agent_results", [])
        citations     = state.get("citations", [])

        summaries = "\n\n".join(
            f"=== {r.get('agent', 'AGENT').upper()} ===\n{r.get('narrative', '')}"
            for r in agent_results
        ) or "(no agent results)"

        context_text = (
            f"Original query: {query}\n\n"
            f"Agent outputs:\n{summaries}\n\n"
            f"Citations: {citations}"
        )
        messages = [
            SystemMessage(content=_SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(content=context_text),
        ]
        response = llm.invoke(messages)
        final_response = response.content if hasattr(response, "content") else str(response)

        # Overall confidence: lowest of individual agent confidences
        confidences = []
        for r in agent_results:
            cls = r.get("classification", {})
            if isinstance(cls, dict) and cls.get("confidence"):
                conf_map = {"HIGH": 0.9, "MEDIUM": 0.6, "LOW": 0.3}
                confidences.append(conf_map.get(cls["confidence"], 0.5))
        confidence = min(confidences) if confidences else 0.5

        needs_review = state.get("needs_human_review", False)
        # Auto-flag if any validation agent found CRITICAL issues
        for r in agent_results:
            if r.get("agent") == "validation" and r.get("critical_count", 0) > 0:
                needs_review = True
                break

        return {
            "final_response":     final_response,
            "citations":          citations,
            "confidence":         confidence,
            "needs_human_review": needs_review,
        }
    return synthesize_node


# ---------------------------------------------------------------------------
# Routing edge
# ---------------------------------------------------------------------------

def _make_route_after_supervisor(
    diagnostic: DiagnosticAgent,
    evidence: EvidenceAgent,
    explanation: ExplanationAgent,
    validation: ValidationAgent,
) -> Any:
    """Return the conditional edge function for post-supervisor routing."""

    def _route(state: HER2AgentState) -> Literal[
        "diagnostic", "evidence", "explanation", "validation", "synthesize"
    ]:
        remaining = list(state.get("target_agents", []))
        # The next agent to call is the first in the list that hasn't run yet
        ran = {r["agent"] for r in state.get("agent_results", []) if "agent" in r}
        for agent_name in remaining:
            if agent_name not in ran:
                return agent_name  # type: ignore[return-value]
        return "synthesize"

    return _route


# ---------------------------------------------------------------------------
# Public: build the multi-agent LangGraph
# ---------------------------------------------------------------------------

def build_agent_graph(llm: BaseChatModel, driver: Driver) -> Any:
    """
    Build and compile the HER2 multi-agent LangGraph.

    Returns a compiled LangGraph `App` (callable with invoke / stream).

    Usage::

        graph = build_agent_graph(llm, driver)
        result = graph.invoke({
            "query": "What is the HER2 status for IHC 2+, ratio 1.8, 4.2 signals/cell?",
            "clinical_data": {"ihc_score": "2+", "ish_group": "Group3",
                              "ish_ratio": 1.8, "signals_per_cell": 4.2},
            **EMPTY_STATE,
        })
        print(result["final_response"])
    """
    diagnostic  = DiagnosticAgent(llm, driver)
    evidence    = EvidenceAgent(llm, driver)
    explanation = ExplanationAgent(llm, driver)
    validation  = ValidationAgent(llm, driver)

    supervisor  = _make_supervisor_node(llm)
    synthesize  = _make_synthesize_node(llm)
    route_fn    = _make_route_after_supervisor(diagnostic, evidence, explanation, validation)

    builder: StateGraph = StateGraph(HER2AgentState)

    # Add nodes
    builder.add_node("supervisor",   supervisor)
    builder.add_node("diagnostic",   diagnostic)
    builder.add_node("evidence",     evidence)
    builder.add_node("explanation",  explanation)
    builder.add_node("validation",   validation)
    builder.add_node("synthesize",   synthesize)

    # Edges
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_fn,
        {
            "diagnostic":  "diagnostic",
            "evidence":    "evidence",
            "explanation": "explanation",
            "validation":  "validation",
            "synthesize":  "synthesize",
        },
    )
    # After each agent, route back through supervisor to decide next step
    for agent_name in ("diagnostic", "evidence", "explanation", "validation"):
        builder.add_edge(agent_name, "supervisor")

    builder.add_edge("synthesize", END)

    return builder.compile()
