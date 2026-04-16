"""
LangGraph state definition for the HER2 multi-agent system.

All agents in the supervisor pattern share this TypedDict as their
graph state.  Fields marked with Annotated[list, add] are accumulated
(not replaced) when multiple agents write to them.
"""
from __future__ import annotations

from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


def _add(a: list, b: list) -> list:  # noqa: D401
    """Minimal list-accumulator for Annotated fields."""
    return a + b


class HER2AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    query: str
    # Optional structured clinical data provided by the caller
    # Keys: ihc_score, ish_group, ish_ratio, signals_per_cell, fractal_d0, ...
    clinical_data: dict

    # ── Routing ────────────────────────────────────────────────────────────
    # Which agents the supervisor has decided to invoke
    target_agents: list[str]
    # Which agent is currently executing
    current_agent: str

    # ── Accumulated outputs (one entry per invoked agent) ──────────────────
    agent_results: Annotated[list[dict], _add]

    # ── Synthesis ──────────────────────────────────────────────────────────
    final_response: str
    citations: list[dict]
    confidence: float

    # ── Control ────────────────────────────────────────────────────────────
    iteration_count: int
    needs_human_review: bool


# Sentinel value for unset optional slots
EMPTY_STATE: HER2AgentState = {
    "query": "",
    "clinical_data": {},
    "target_agents": [],
    "current_agent": "",
    "agent_results": [],
    "final_response": "",
    "citations": [],
    "confidence": 0.0,
    "iteration_count": 0,
    "needs_human_review": False,
}
