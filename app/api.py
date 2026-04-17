"""
HER2 Knowledge Graph — FastAPI REST Endpoints

Implements the API layer from Sprint 4 (§app/api.py, implementation_plan.md).

Endpoints:
    POST /diagnose             — IHC/ISH → HER2 classification + pathway
    POST /query                — Natural language → multi-agent response
    POST /validate             — Case consistency validation
    GET  /evidence/{category}  — Therapeutic eligibility by HER2 category
    GET  /stats                — KG health metrics
    GET  /health               — Liveness probe
    GET  /docs                 — Interactive API docs (auto by FastAPI)

Run:
    uvicorn app.api:app --reload --port 8000
"""
from __future__ import annotations

import os
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="HER2 Knowledge Graph API",
    description=(
        "REST API for HER2 diagnostic classification, validation, "
        "evidence retrieval, and multi-agent clinical reasoning. "
        "Powered by LangGraph, Neo4j, and Ollama."
    ),
    version="1.0.0",
    contact={
        "name": "DigPatho",
        "url": "https://github.com/your-org/KnowledgeGraphHER2",
    },
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------
_llm_singleton = None
_driver_singleton = None


def _get_llm():
    global _llm_singleton
    if _llm_singleton is None:
        model = os.getenv("LLM_MODEL", "qwen3:8b")
        try:
            from langchain_ollama import ChatOllama
            _llm_singleton = ChatOllama(model=model, temperature=0)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"LLM unavailable: {exc}") from exc
    return _llm_singleton


def _get_driver():
    global _driver_singleton
    if _driver_singleton is None:
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        try:
            from neo4j import GraphDatabase
            _driver_singleton = GraphDatabase.driver(uri, auth=(user, password))
            _driver_singleton.verify_connectivity()
        except Exception:
            _driver_singleton = None  # Allow degraded mode
    return _driver_singleton


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class DiagnoseRequest(BaseModel):
    ihc_score: str = Field(..., examples=["2+"], description="IHC score: 3+, 2+, 1+, 0+, or 0")
    ish_group: str | None = Field(None, examples=["Group3"], description="ISH group: Group1–Group5")
    ish_ratio: float | None = Field(None, examples=[1.7], description="HER2/CEP17 ratio")
    signals_per_cell: float | None = Field(None, examples=[5.4], description="Mean HER2 signals per cell")
    narrate: bool = Field(False, description="Include LLM narrative (requires Ollama)")


class DiagnoseResponse(BaseModel):
    classification: str
    ihc_score: str
    ish_group: str | None
    confidence: str
    guideline: str
    action: str
    pathway: str
    narrative: str | None = None


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language clinical question", min_length=5)
    ihc_score: str | None = Field(None, examples=["2+"])
    ish_group: str | None = Field(None, examples=["Group3"])
    ish_ratio: float | None = Field(None)
    agents: list[str] | None = Field(
        None,
        description="Optional list of agents to invoke: diagnostic, evidence, explanation, validation",
    )


class QueryResponse(BaseModel):
    query: str
    agents_invoked: list[str]
    confidence: float
    needs_human_review: bool
    synthesis: str
    agent_results: list[dict[str, Any]]


class ValidateRequest(BaseModel):
    ihc_score: str = Field(..., examples=["3+"])
    ish_group: str | None = Field(None, examples=["Group5"])
    fractal_d0: float | None = Field(None, examples=[1.9])


class ValidateResponse(BaseModel):
    status: str          # "PASS" | "FAIL"
    issues: list[dict[str, Any]]
    needs_human_review: bool
    narrative: str | None = None


class EvidenceResponse(BaseModel):
    category: str
    eligible_agents: list[dict[str, Any]]
    not_eligible_agents: list[dict[str, Any]]
    guidelines: list[str]
    summary: str | None = None


class StatsResponse(BaseModel):
    total_nodes: int
    total_relations: int
    node_counts: dict[str, int]
    neo4j_connected: bool


# ---------------------------------------------------------------------------
# Static evidence reference (guidelines-based, no Neo4j required)
# ---------------------------------------------------------------------------

_STATIC_EVIDENCE: dict[str, dict] = {
    "HER2_Positive": {
        "eligible": [
            {"agent": "Trastuzumab", "context": "Standard first-line (ASCO/CAP 2023)", "trial": "ToGA"},
            {"agent": "Pertuzumab", "context": "Early + mBC (ASCO/CAP 2023)", "trial": "CLEOPATRA"},
            {"agent": "T-DXd", "context": "2L+ metastatic", "trial": "DESTINY-Breast03"},
            {"agent": "Tucatinib", "context": "2L+ metastatic", "trial": "HER2CLIMB"},
        ],
        "not_eligible": [],
        "guidelines": ["ASCO/CAP 2023", "ESMO 2023", "NCCN 2023"],
    },
    "HER2_Equivocal": {
        "eligible": [
            {"agent": "Trastuzumab", "context": "When reflexed ISH confirms positivity", "trial": "—"},
            {"agent": "T-DXd", "context": "When ISH+ confirmed", "trial": "DESTINY-Breast03"},
        ],
        "not_eligible": [
            {"agent": "T-DXd", "context": "Not eligible pending ISH result", "trial": "—"},
        ],
        "guidelines": ["ASCO/CAP 2023"],
    },
    "HER2_Low": {
        "eligible": [
            {"agent": "T-DXd", "context": "First-line metastatic (HR+/HR-)", "trial": "DESTINY-Breast04"},
        ],
        "not_eligible": [
            {"agent": "Trastuzumab", "context": "Not eligible — IHC 1+ or 2+/ISH- is not amplified", "trial": "—"},
            {"agent": "Pertuzumab", "context": "Not eligible", "trial": "—"},
        ],
        "guidelines": ["ASCO/CAP 2023", "DESTINY-Breast04 (NEJM 2022)"],
    },
    "HER2_Ultralow": {
        "eligible": [
            {"agent": "T-DXd", "context": "Expanded criteria — HR+, metastatic, endocrine-pretreated", "trial": "DESTINY-Breast06"},
        ],
        "not_eligible": [
            {"agent": "Trastuzumab", "context": "Not eligible", "trial": "—"},
            {"agent": "Pertuzumab", "context": "Not eligible", "trial": "—"},
        ],
        "guidelines": ["Rakha_International_2026", "DESTINY-Breast06"],
    },
    "HER2_Null": {
        "eligible": [],
        "not_eligible": [
            {"agent": "T-DXd", "context": "Not eligible — no HER2 expression (standard criteria)", "trial": "—"},
            {"agent": "Trastuzumab", "context": "Not eligible", "trial": "—"},
        ],
        "guidelines": ["ASCO/CAP 2023"],
    },
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", summary="Liveness probe")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "1.0.0"}


@app.post(
    "/diagnose",
    response_model=DiagnoseResponse,
    summary="HER2 classification from IHC/ISH values",
    description=(
        "Deterministic classification using ASCO/CAP 2023 and Rakha 2026 algorithms. "
        "Set `narrate=true` to include an LLM-generated clinical interpretation (requires Ollama)."
    ),
)
def diagnose(request: DiagnoseRequest) -> DiagnoseResponse:
    from src.agents.diagnostic_agent import _classify_from_data

    clinical_data: dict[str, Any] = {"ihc_score": request.ihc_score}
    if request.ish_group:
        clinical_data["ish_group"] = request.ish_group
    if request.ish_ratio is not None:
        clinical_data["ish_ratio"] = request.ish_ratio
    if request.signals_per_cell is not None:
        clinical_data["signals_per_cell"] = request.signals_per_cell

    result = _classify_from_data(clinical_data)

    narrative = None
    if request.narrate:
        llm = _get_llm()
        driver = _get_driver()
        from src.agents.diagnostic_agent import DiagnosticAgent
        from src.agents.state import EMPTY_STATE
        agent = DiagnosticAgent(llm=llm, driver=driver)
        state = {
            **EMPTY_STATE,
            "clinical_data": clinical_data,
            "query": f"Classify IHC {request.ihc_score}",
        }
        new_state = agent(state)
        for r in new_state.get("agent_results", []):
            if r.get("agent") == "diagnostic":
                narrative = r.get("narrative")
                break

    pathway_steps = result.get("pathway_steps", [])
    return DiagnoseResponse(
        classification=result.get("classification", "Unknown"),
        ihc_score=request.ihc_score,
        ish_group=request.ish_group,
        confidence=result.get("confidence", "UNKNOWN"),
        guideline=result.get("applicable_guideline", result.get("guideline", "")),
        action=result.get("action_required", result.get("action", "")),
        pathway=" → ".join(pathway_steps) if pathway_steps else result.get("pathway", ""),
        narrative=narrative,
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Natural language clinical query (multi-agent)",
    description=(
        "Routes the query through the LangGraph supervisor. "
        "Requires Ollama running locally."
    ),
)
def query_endpoint(request: QueryRequest) -> QueryResponse:
    llm = _get_llm()
    driver = _get_driver()

    clinical_data: dict[str, Any] = {}
    if request.ihc_score:
        clinical_data["ihc_score"] = request.ihc_score
    if request.ish_group:
        clinical_data["ish_group"] = request.ish_group
    if request.ish_ratio is not None:
        clinical_data["ish_ratio"] = request.ish_ratio

    from src.agents.supervisor import build_agent_graph
    from src.agents.state import EMPTY_STATE
    graph = build_agent_graph(llm=llm, driver=driver)
    initial_state = {
        **EMPTY_STATE,
        "query": request.query,
        "clinical_data": clinical_data,
    }
    if request.agents:
        initial_state["target_agents"] = list(request.agents)

    final = graph.invoke(initial_state)

    agents_invoked = [r["agent"] for r in final.get("agent_results", []) if "agent" in r]
    return QueryResponse(
        query=request.query,
        agents_invoked=agents_invoked,
        confidence=float(final.get("confidence", 0.0)),
        needs_human_review=bool(final.get("needs_human_review", False)),
        synthesis=final.get("final_response", final.get("synthesis", "")),
        agent_results=final.get("agent_results", []),
    )


@app.post(
    "/validate",
    response_model=ValidateResponse,
    summary="Clinical consistency validation",
    description="Checks IHC/ISH values for rule violations and conflicts.",
)
def validate(request: ValidateRequest) -> ValidateResponse:
    from src.agents.validation_agent import ValidationAgent
    from src.agents.state import EMPTY_STATE

    clinical_data: dict[str, Any] = {"ihc_score": request.ihc_score}
    if request.ish_group:
        clinical_data["ish_group"] = request.ish_group
    if request.fractal_d0 is not None:
        clinical_data["fractal_d0"] = request.fractal_d0

    llm = None
    try:
        llm = _get_llm()
    except HTTPException:
        pass  # Degraded mode — deterministic only

    driver = _get_driver()
    agent = ValidationAgent(llm=llm, driver=driver)
    state = {**EMPTY_STATE, "clinical_data": clinical_data, "query": "Validate"}
    new_state = agent(state)

    issues: list[dict] = []
    narrative: str | None = None
    status = "PASS"
    for r in new_state.get("agent_results", []):
        if r.get("agent") == "validation":
            issues = r.get("issues", [])
            status = r.get("status", "PASS")
            narrative = r.get("narrative")
            break

    return ValidateResponse(
        status=status,
        issues=issues,
        needs_human_review=bool(new_state.get("needs_human_review", False)),
        narrative=narrative,
    )


@app.get(
    "/evidence/{category}",
    response_model=EvidenceResponse,
    summary="Therapeutic eligibility by HER2 category",
    description="Returns evidence-based eligibility for HER2-targeted agents. Uses static guideline data.",
)
def evidence(
    category: str,
    summarize: bool = Query(False, description="Generate LLM summary (requires Ollama)"),
) -> EvidenceResponse:
    # Normalise key
    norm = category.replace("-", "_")
    if norm not in _STATIC_EVIDENCE:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown category '{category}'. Valid: {list(_STATIC_EVIDENCE.keys())}",
        )
    data = _STATIC_EVIDENCE[norm]

    summary: str | None = None
    if summarize:
        try:
            llm = _get_llm()
            eligible_str = ", ".join(e["agent"] for e in data["eligible"]) or "none"
            prompt = (
                f"For HER2 category {category.replace('_', '-')}, eligible agents are: {eligible_str}. "
                f"Guidelines: {', '.join(data['guidelines'])}. "
                f"Write a 2-sentence clinical summary."
            )
            summary = llm.invoke(prompt).content
        except Exception:
            pass

    return EvidenceResponse(
        category=norm,
        eligible_agents=data["eligible"],
        not_eligible_agents=data["not_eligible"],
        guidelines=data["guidelines"],
        summary=summary,
    )


@app.get(
    "/stats",
    response_model=StatsResponse,
    summary="Knowledge Graph statistics",
    description="Returns node/relation counts from Neo4j. Returns zeros if not connected.",
)
def stats() -> StatsResponse:
    driver = _get_driver()
    if driver is None:
        return StatsResponse(
            total_nodes=0,
            total_relations=0,
            node_counts={},
            neo4j_connected=False,
        )
    try:
        from src.graph.neo4j_builder import get_graph_stats
        s = get_graph_stats(driver)
        return StatsResponse(
            total_nodes=s.get("total_nodes", 0),
            total_relations=s.get("total_relations", 0),
            node_counts=s.get("node_counts", {}),
            neo4j_connected=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
