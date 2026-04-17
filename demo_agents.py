"""
Demo interactivo del sistema multi-agente HER2 KG.

Usa Ollama (qwen3:8b) para la parte LLM y simula el driver Neo4j
con respuestas vacías (el clasificador diagnóstico es 100% local).

Ejecutar:
    python demo_agents.py
    python demo_agents.py --query "Why is IHC 2+ equivocal?" --agent explanation
"""
from __future__ import annotations

import argparse
import json
import sys
from unittest.mock import MagicMock

from langchain_ollama import ChatOllama

# ── Mock Neo4j driver (no server needed) ─────────────────────────────────────

def _mock_driver():
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.run = MagicMock(return_value=iter([]))
    mock_driver = MagicMock()
    mock_driver.session = MagicMock(return_value=mock_session)
    return mock_driver

# ── Casos de prueba ───────────────────────────────────────────────────────────

CASES = {
    "diagnostic_3plus": {
        "query": "What is the HER2 status for this biopsy?",
        "clinical_data": {"ihc_score": "3+"},
        "agent": "diagnostic",
    },
    "diagnostic_2plus_ish3": {
        "query": "IHC 2+, ISH Group 3 — classify HER2 status",
        "clinical_data": {"ihc_score": "2+", "ish_group": "Group3", "ish_ratio": 1.7, "signals_per_cell": 5.4},
        "agent": "diagnostic",
    },
    "diagnostic_ultralow": {
        "query": "IHC 0+ result — is patient eligible for T-DXd?",
        "clinical_data": {"ihc_score": "0+"},
        "agent": "diagnostic",
    },
    "validation_conflict": {
        "query": "Is this case consistent? IHC 3+ but ISH Group 5",
        "clinical_data": {"ihc_score": "3+", "ish_group": "Group5"},
        "agent": "validation",
    },
    "explanation": {
        "query": "Why is IHC 2+ considered equivocal and what does it mean therapeutically?",
        "clinical_data": {},
        "agent": "explanation",
    },
    "evidence": {
        "query": "Is T-DXd (trastuzumab deruxtecan) approved for HER2-low breast cancer?",
        "clinical_data": {"category": "HER2_Low"},
        "agent": "evidence",
    },
}


def run_single_agent(llm, driver, case_name: str) -> None:
    case = CASES[case_name]
    query         = case["query"]
    clinical_data = case["clinical_data"]
    agent_name    = case["agent"]

    print(f"\n{'='*70}")
    print(f"CASO: {case_name}")
    print(f"AGENTE: {agent_name.upper()}")
    print(f"QUERY: {query}")
    if clinical_data:
        print(f"DATOS CLÍNICOS: {json.dumps(clinical_data, ensure_ascii=False)}")
    print(f"{'='*70}\n")

    from src.agents.state import EMPTY_STATE

    state = dict(EMPTY_STATE)
    state["query"]         = query
    state["clinical_data"] = clinical_data

    if agent_name == "diagnostic":
        from src.agents.diagnostic_agent import DiagnosticAgent, _classify_from_data

        # Primero mostramos la clasificación determinista (sin LLM)
        cls = _classify_from_data(clinical_data)
        print("── CLASIFICACIÓN DETERMINISTA (sin LLM) ──")
        print(f"  Clasificación: {cls['classification']}")
        print(f"  IHC Score:     {cls['ihc_score']}")
        print(f"  Guideline:     {cls['applicable_guideline']}")
        print(f"  Confianza:     {cls['confidence']}")
        if cls.get("action_required"):
            print(f"  Acción:        {cls['action_required']}")
        print(f"  Ruta:          {' → '.join(cls['pathway_steps'])}")
        print()

        # Luego la narración LLM
        print("── NARRACIÓN LLM (qwen3:8b) ──")
        agent = DiagnosticAgent(llm, driver)
        update = agent(state)
        print(update["agent_results"][0]["narrative"])

    elif agent_name == "validation":
        from src.agents.validation_agent import ValidationAgent

        agent = ValidationAgent(llm, driver)
        update = agent(state)
        result = update["agent_results"][0]

        print(f"── ESTADO: {result['status']} ──")
        if result["issues"]:
            for issue in result["issues"]:
                print(f"  [{issue['severity']}] {issue['rule_id']}: {issue['description']}")
        else:
            print("  Sin problemas detectados.")
        print()
        print("── NARRACIÓN LLM ──")
        print(result["narrative"])

    elif agent_name == "explanation":
        from src.agents.explanation_agent import ExplanationAgent

        agent = ExplanationAgent(llm, driver)
        update = agent(state)
        print(update["agent_results"][0]["narrative"])

    elif agent_name == "evidence":
        from src.agents.evidence_agent import EvidenceAgent

        agent = EvidenceAgent(llm, driver)
        update = agent(state)
        print(update["agent_results"][0]["narrative"])


def run_supervisor_demo(llm, driver, query: str, clinical_data: dict) -> None:
    from src.agents.supervisor import build_agent_graph
    from src.agents.state import EMPTY_STATE

    print(f"\n{'='*70}")
    print("DEMO: SUPERVISOR MULTI-AGENTE")
    print(f"QUERY: {query}")
    if clinical_data:
        print(f"DATOS: {json.dumps(clinical_data, ensure_ascii=False)}")
    print(f"{'='*70}\n")

    print("[1/3] Construyendo el grafo LangGraph...")
    graph = build_agent_graph(llm, driver)
    print("[2/3] Invoking...")

    state = dict(EMPTY_STATE)
    state["query"]         = query
    state["clinical_data"] = clinical_data

    result = graph.invoke(state)

    print(f"\n[3/3] Agentes invocados: {[r['agent'] for r in result.get('agent_results', [])]}")
    print(f"Confianza: {result.get('confidence', 0):.0%}")
    print(f"Requiere revisión humana: {'SÍ ⚠️' if result.get('needs_human_review') else 'NO'}")
    print(f"\n── RESPUESTA FINAL ──\n{result.get('final_response', '')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo del sistema multi-agente HER2 KG")
    parser.add_argument("--case", choices=list(CASES.keys()), default=None,
                        help="Caso predefinido a ejecutar")
    parser.add_argument("--agent", choices=["diagnostic", "evidence", "explanation", "validation"],
                        default=None)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--ihc", type=str, default=None, help="IHC score: 0, 0+, 1+, 2+, 3+")
    parser.add_argument("--ish-group", type=str, default=None)
    parser.add_argument("--supervisor", action="store_true",
                        help="Run full supervisor pipeline")
    parser.add_argument("--model", type=str, default="qwen3:8b")
    parser.add_argument("--all", action="store_true", help="Run all predefined cases")
    parser.add_argument("--neo4j", action="store_true",
                        help="Use real Neo4j driver instead of mock (requires running Neo4j)")
    args = parser.parse_args()

    print(f"Cargando modelo Ollama: {args.model} ...")
    llm = ChatOllama(model=args.model, temperature=0)
    if args.neo4j:
        from src.pipeline.config import PipelineConfig
        cfg = PipelineConfig.from_env()
        driver = cfg.get_neo4j_driver()
        print(f"Conectado a Neo4j: {cfg.neo4j_uri}")
    else:
        driver = _mock_driver()
    print("Listo.\n")

    if args.all:
        for case_name in CASES:
            run_single_agent(llm, driver, case_name)
        return

    if args.supervisor or (args.query and not args.agent):
        query         = args.query or "Classify HER2 status: IHC 2+, ISH Group 3, ratio 1.7"
        clinical_data = {}
        if args.ihc:
            clinical_data["ihc_score"] = args.ihc
        if args.ish_group:
            clinical_data["ish_group"] = args.ish_group
        run_supervisor_demo(llm, driver, query, clinical_data)
        return

    if args.case:
        run_single_agent(llm, driver, args.case)
        return

    if args.agent and args.query:
        # Ad-hoc case
        clinical_data = {}
        if args.ihc:
            clinical_data["ihc_score"] = args.ihc
        if args.ish_group:
            clinical_data["ish_group"] = args.ish_group
        CASES["_adhoc"] = {"query": args.query, "clinical_data": clinical_data, "agent": args.agent}
        run_single_agent(llm, driver, "_adhoc")
        return

    # Default: show all 3 key cases
    for case_name in ("diagnostic_2plus_ish3", "validation_conflict", "explanation"):
        run_single_agent(llm, driver, case_name)


if __name__ == "__main__":
    main()
