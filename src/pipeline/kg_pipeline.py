"""
Main KG construction pipeline — LangGraph-orchestrated 8-phase workflow.

Phases:
  1. INGEST   — Load .md documents (priority order)
  2. CHUNK    — Semantic-aware segmentation
  3. EXTRACT  — LLM-based entity/relation extraction
  4. RESOLVE  — Canonical URI resolution
  5. BUILD    — Neo4j graph construction (seed + extracted)
  6. VALIDATE — Clinical consistency checks
  7. EXPORT   — RDF/OWL export placeholder + stats
  8. DONE

Run with:
    python -m app.cli run-pipeline
    python -m app.cli run-pipeline --llm-mode ollama  (cheap/local)
    python -m app.cli run-pipeline --llm-mode claude  (production)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated, Any, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.domain.models import (
    DocumentChunk, ExtractionResult, ResolvedEntity, ResolvedRelation,
    ValidationReport,
)
from src.domain.ontology import SEED_ENTITIES
from src.extraction.entity_extractor import extract_batch
from src.extraction.resolution import resolve_all_entities
from src.extraction.algorithm_parser import parse_and_load_all_algorithms
from src.graph.neo4j_builder import (
    get_graph_stats, initialize_schema, load_seed_data,
    load_toy_fractal_specimens, upsert_chunk_node,
    upsert_entities, upsert_relations,
)
from src.graph.rdf_exporter import export_rdf
from src.graph.validator import run_validation
from src.graph.vector_indexer import create_vector_indexes
from src.ingestion.markdown_loader import load_all_markdown_docs
from src.ingestion.pdf_loader import load_all_pdf_docs
from src.pipeline.config import PipelineConfig


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

def _append(a: list, b: list) -> list:  # reducer for Annotated lists
    return a + b


class PipelineState(TypedDict):
    config:               PipelineConfig
    # Phase 1
    raw_documents:        list[dict]
    # Phase 2
    chunks:               list[DocumentChunk]
    # Phase 3
    raw_extractions:      list[ExtractionResult]
    extraction_errors:    Annotated[list[str], _append]
    # Phase 4
    resolved_entities:    list[ResolvedEntity]
    resolved_relations:   list[ResolvedRelation]
    unresolved_count:     int
    # Phase 5
    neo4j_stats:          dict
    # Phase 6
    validation_report:    Optional[ValidationReport]
    is_consistent:        bool
    # Phase 7
    export_paths:         dict
    # Control
    errors:               Annotated[list[str], _append]
    current_phase:        str
    requires_human_review: bool
    start_time:           float


# ---------------------------------------------------------------------------
# Phase nodes
# ---------------------------------------------------------------------------

def phase_ingest(state: PipelineState) -> dict:
    """Phase 1: Load all .md documents from docs_dir + PDFs from guides_dir."""
    cfg = state["config"]
    docs_dir   = Path(cfg.docs_dir)
    guides_dir = Path(cfg.guides_dir)
    print(f"\n{'='*60}")
    print(f"[Phase 1/7] INGEST — Loading from {docs_dir} and {guides_dir}")

    if not docs_dir.exists():
        return {"errors": [f"docs_dir not found: {docs_dir}"], "current_phase": "INGEST_FAILED"}

    # Technical appendices contain non-clinical tutorial content (GraphRAG/LangChain
    # examples with synthetic entities like Ana García, Ciudad, etc.) that pollute the KG.
    EXCLUDED_DOCS = {
        "apendice_frameworks_graphrag.md",
        "apendice_langchain_langgraph.md",
    }

    # Collect markdown file metadata
    raw_docs = [
        {"path": str(p), "format": "markdown", "name": p.name}
        for p in sorted(docs_dir.glob("*.md"))
        if p.name not in EXCLUDED_DOCS
    ]
    excluded = [p.name for p in sorted(docs_dir.glob("*.md")) if p.name in EXCLUDED_DOCS]
    if excluded:
        print(f"  Skipping {len(excluded)} non-clinical docs: {excluded}")
    print(f"  Found {len(raw_docs)} .md files")

    # Collect PDF file metadata
    if guides_dir.exists():
        pdf_docs = [
            {"path": str(p), "format": "pdf", "name": p.name}
            for p in sorted(guides_dir.glob("*.pdf"))
        ]
        print(f"  Found {len(pdf_docs)} PDF guideline files")
        raw_docs.extend(pdf_docs)
    else:
        print(f"  guides_dir not found: {guides_dir} — skipping PDF ingestion")

    return {
        "raw_documents": raw_docs,
        "current_phase": "INGEST_DONE",
    }


def phase_chunk(state: PipelineState) -> dict:
    """Phase 2: Semantic-aware chunking of all loaded documents (md + PDF)."""
    cfg = state["config"]
    docs_dir   = Path(cfg.docs_dir)
    guides_dir = Path(cfg.guides_dir)
    print(f"\n{'='*60}")
    print(f"[Phase 2/7] CHUNK — Semantic segmentation (size={cfg.chunk_size}, overlap={cfg.chunk_overlap})")

    # Mirror the exclusion list from phase_ingest so chunks match processed docs
    EXCLUDED_DOCS = {
        "apendice_frameworks_graphrag.md",
        "apendice_langchain_langgraph.md",
    }
    # Markdown chunks — technical appendices are tagged TECHNICAL_APPENDIX
    chunks = load_all_markdown_docs(
        docs_dir,
        chunk_size=cfg.chunk_size,
        overlap=cfg.chunk_overlap,
        exclude=EXCLUDED_DOCS,
    )
    md_count = len(chunks)
    print(f"  Markdown: {md_count} chunks")

    # PDF guideline chunks (5.2)
    if guides_dir.exists():
        pdf_chunks = load_all_pdf_docs(
            guides_dir,
            chunk_size=cfg.chunk_size,
            overlap=cfg.chunk_overlap,
        )
        chunks.extend(pdf_chunks)
        print(f"  PDFs:     {len(pdf_chunks)} chunks from {guides_dir.name}/")
    else:
        print(f"  PDFs:     guides_dir not found ({guides_dir}) — skipping")

    print(f"  Total:    {len(chunks)} chunks")

    # Print content type distribution
    from collections import Counter
    ct_counts = Counter(c.content_type.value for c in chunks)
    for ct, cnt in ct_counts.most_common():
        print(f"    {ct:25s}: {cnt}")

    return {
        "chunks": chunks,
        "current_phase": "CHUNK_DONE",
    }


def phase_extract(state: PipelineState) -> dict:
    """Phase 3: LLM-based entity/relation extraction from chunks."""
    cfg = state["config"]
    chunks = state["chunks"]
    print(f"\n{'='*60}")
    print(f"[Phase 3/7] EXTRACT — LLM: {cfg.llm_mode}/{cfg._active_model()}")

    llm = cfg.get_llm()
    results = extract_batch(chunks, llm, verbose=True)

    errors = [r.error for r in results if r.error]
    total_entities = sum(len(r.entities) for r in results)
    total_relations = sum(len(r.relations) for r in results)

    print(f"\n  Extracted {total_entities} entities, {total_relations} relations")
    print(f"  Errors: {len(errors)}")

    return {
        "raw_extractions": results,
        "extraction_errors": errors,
        "current_phase": "EXTRACT_DONE",
    }


def phase_resolve(state: PipelineState) -> dict:
    """Phase 4: URI resolution of extracted entities."""
    print(f"\n{'='*60}")
    print(f"[Phase 4/7] RESOLVE — Canonical URI resolution")

    raw_extractions = state["raw_extractions"]
    all_resolved: list[ResolvedEntity] = []
    all_relations: list[ResolvedRelation] = []
    unresolved = 0

    for extraction in raw_extractions:
        if extraction.error or not extraction.entities:
            continue

        resolved = resolve_all_entities(extraction.entities, source_doc=extraction.chunk_id)

        # Build local extraction ID → resolved entity ID map (per chunk)
        local_id_map: dict[str, str] = {}
        for orig_ent, res_ent in zip(extraction.entities, resolved):
            local_id_map[orig_ent.id] = res_ent.id

        # Count local (her2:) URIs
        for r in resolved:
            if r.resolved_uri.startswith("her2:") and not r.ncit_uri:
                unresolved += 1

        all_resolved.extend(resolved)

        # Translate relation IDs
        for rel in extraction.relations:
            subj_id = local_id_map.get(rel.subject_id, rel.subject_id)
            obj_id  = local_id_map.get(rel.object_id,  rel.object_id)
            from src.domain.models import ResolvedRelation
            all_relations.append(ResolvedRelation(
                subject_id=subj_id,
                predicate=rel.predicate,
                object_id=obj_id,
                confidence=rel.confidence,
                evidence=rel.evidence,
                source_chunk=extraction.chunk_id,
                guideline_version=rel.guideline_version,
                conditions=rel.conditions,
            ))

    print(f"  Resolved {len(all_resolved)} entities ({unresolved} using local her2: URIs)")
    print(f"  Resolved {len(all_relations)} relations")

    return {
        "resolved_entities": all_resolved,
        "resolved_relations": all_relations,
        "unresolved_count": unresolved,
        "current_phase": "RESOLVE_DONE",
    }


def phase_build(state: PipelineState) -> dict:
    """Phase 5: Neo4j graph construction (seed + extracted + toy fractals)."""
    cfg = state["config"]
    print(f"\n{'='*60}")
    print(f"[Phase 5/7] BUILD — Neo4j at {cfg.neo4j_uri}")

    driver = cfg.get_neo4j_driver()
    try:
        # Schema
        initialize_schema(driver)

        # Seed entities from ontology
        seed_stats = load_seed_data(driver)

        # Toy fractal specimens
        load_toy_fractal_specimens(driver)

        # Extracted entities
        resolved_entities = state.get("resolved_entities", [])
        n_ents = upsert_entities(driver, resolved_entities)
        print(f"  Upserted {n_ents} extracted entities")

        # Extracted relations
        resolved_relations = state.get("resolved_relations", [])
        n_rels = upsert_relations(driver, resolved_relations)
        print(f"  Upserted {n_rels} extracted relations")

        # Store chunk provenance nodes
        for chunk in state.get("chunks", []):
            upsert_chunk_node(
                driver, chunk.chunk_id, chunk.source_doc,
                chunk.section, chunk.content, chunk.content_type.value,
            )

        # Wire MENTIONS edges (Chunk → Entity) for GraphRAG retrieval
        from src.graph.neo4j_builder import create_mentions_edges
        n_mentions = create_mentions_edges(driver, resolved_entities)
        print(f"  Created {n_mentions} MENTIONS edges (Chunk->Entity)")

        stats = get_graph_stats(driver)
        print(f"  Graph stats: {stats['total_nodes']} nodes, {stats['total_relations']} relations")
        for label, cnt in stats["node_counts"].items():
            if cnt > 0:
                print(f"    {label:30s}: {cnt}")

    finally:
        driver.close()

    return {
        "neo4j_stats": stats,
        "current_phase": "BUILD_DONE",
    }


def phase_parse_algorithms(state: PipelineState) -> dict:
    """Phase 6: Parse IHC/ISH decision tree algorithms into Neo4j nodes."""
    cfg = state["config"]
    print(f"\n{'='*60}")
    print(f"[Phase 6/8] PARSE_ALGORITHMS — IHC/ISH decision trees")

    driver = cfg.get_neo4j_driver()
    try:
        stats = parse_and_load_all_algorithms(driver, verbose=True)
    finally:
        driver.close()

    return {
        "neo4j_stats": {**state.get("neo4j_stats", {}), "algorithm_nodes": stats["nodes"],
                        "algorithm_edges": stats["edges"]},
        "current_phase": "PARSE_ALGORITHMS_DONE",
    }


def phase_validate(state: PipelineState) -> dict:
    """Phase 7: Run clinical validation rules."""
    cfg = state["config"]
    print(f"\n{'='*60}")
    print(f"[Phase 7/8] VALIDATE — Clinical consistency checks")

    driver = cfg.get_neo4j_driver()
    try:
        report = run_validation(driver, verbose=True)
    finally:
        driver.close()

    return {
        "validation_report": report,
        "is_consistent": report.is_consistent,
        "current_phase": "VALIDATE_DONE",
    }


def phase_export(state: PipelineState) -> dict:
    """Phase 8: RDF/OWL export, vector indexes, and pipeline log."""
    cfg = state["config"]
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    elapsed = time.time() - state.get("start_time", time.time())

    import datetime as _dt
    run_ts = _dt.datetime.utcnow()
    ts_suffix = run_ts.strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*60}")
    print(f"[Phase 8/8] EXPORT — RDF export + vector indexes + pipeline log")

    export_paths: dict = {}

    # RDF/OWL export — timestamped filename to avoid overwriting previous runs
    driver = cfg.get_neo4j_driver()
    try:
        rdf_paths = export_rdf(driver, output_dir=output_dir, verbose=True, timestamp=ts_suffix)
        export_paths.update(rdf_paths)
    except Exception as exc:
        print(f"  [EXPORT] RDF export skipped: {exc!s:.100}")
    finally:
        driver.close()

    # Vector indexes (structure only — embeddings require embedder call)
    driver2 = cfg.get_neo4j_driver()
    try:
        create_vector_indexes(driver2, dim=cfg.embedding_dim, verbose=True)
    except Exception as exc:
        print(f"  [EXPORT] Vector index creation skipped: {exc!s:.100}")
    finally:
        driver2.close()

    # Compute embeddings for entities and chunks (batch, idempotent)
    driver3 = cfg.get_neo4j_driver()
    try:
        embedder = cfg.get_embedder()
        from src.graph.vector_indexer import embed_all_entities, embed_all_chunks
        n_ent = embed_all_entities(driver3, embedder, verbose=True)
        n_chk = embed_all_chunks(driver3, embedder, verbose=True)
        print(f"  [EXPORT] Embeddings: {n_ent} entities + {n_chk} chunks")
    except Exception as exc:
        print(f"  [EXPORT] Embeddings skipped: {exc!s:.120}")
    finally:
        driver3.close()

    report = state.get("validation_report")
    report_data = report.summary() if report else {}

    log = {
        "run_timestamp":    run_ts.isoformat(),
        "llm_mode":         cfg.llm_mode,
        "active_model":     cfg._active_model(),
        "docs_dir":         cfg.docs_dir,
        "chunks_processed": len(state.get("chunks", [])),
        "entities_resolved": len(state.get("resolved_entities", [])),
        "relations_resolved": len(state.get("resolved_relations", [])),
        "neo4j_stats":      state.get("neo4j_stats", {}),
        "validation":       report_data,
        "is_consistent":    state.get("is_consistent", False),
        "errors":           state.get("errors", []),
        "elapsed_seconds":  round(elapsed, 1),
    }

    log_path = output_dir / f"pipeline_log_{ts_suffix}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, default=str)

    print(f"  Pipeline log written to: {log_path}")
    print(f"  Total elapsed: {elapsed:.1f}s")
    print(f"\n{'='*60}")
    print("  KG CONSTRUCTION COMPLETE")
    print(f"  Neo4j Browser: http://localhost:7474")
    print(f"  Explore: MATCH (n) RETURN n LIMIT 50")
    print(f"{'='*60}\n")

    export_paths["log"] = str(log_path)
    return {
        "export_paths": export_paths,
        "current_phase": "DONE",
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def should_skip_extraction(state: PipelineState) -> str:
    """If no chunks, skip extraction."""
    return "extract" if state.get("chunks") else "build"


def check_validation(state: PipelineState) -> str:
    """After validation, always proceed to export."""
    return "export"


# ---------------------------------------------------------------------------
# Build the LangGraph pipeline
# ---------------------------------------------------------------------------

def build_pipeline() -> Any:
    """Construct, compile, and return the LangGraph pipeline."""
    workflow = StateGraph(PipelineState)

    workflow.add_node("ingest",           phase_ingest)
    workflow.add_node("chunk",             phase_chunk)
    workflow.add_node("extract",           phase_extract)
    workflow.add_node("resolve",           phase_resolve)
    workflow.add_node("build",             phase_build)
    workflow.add_node("parse_algorithms",  phase_parse_algorithms)
    workflow.add_node("validate",          phase_validate)
    workflow.add_node("export",            phase_export)

    workflow.add_edge(START,               "ingest")
    workflow.add_edge("ingest",            "chunk")
    workflow.add_conditional_edges("chunk", should_skip_extraction,
                                   {"extract": "extract", "build": "build"})
    workflow.add_edge("extract",           "resolve")
    workflow.add_edge("resolve",           "build")
    workflow.add_edge("build",             "parse_algorithms")
    workflow.add_edge("parse_algorithms",  "validate")
    workflow.add_edge("validate",          "export")
    workflow.add_edge("export",            END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_pipeline(llm_mode: str = "ollama") -> dict:
    """
    Run the full KG construction pipeline.

    Args:
        llm_mode: "ollama" (default, free/local), "openai" (cheap), "claude" (production)
    """
    import os
    os.environ["HER2_KG_LLM_MODE"] = llm_mode

    cfg = PipelineConfig.from_env()
    cfg.configure_langsmith()

    print(f"\nHER2 Knowledge Graph — Pipeline v2.0")
    print(f"LLM: {cfg.llm_mode} / {cfg._active_model()}")
    print(f"Neo4j: {cfg.neo4j_uri}")

    app = build_pipeline()
    initial_state: PipelineState = {
        "config":           cfg,
        "raw_documents":    [],
        "chunks":           [],
        "raw_extractions":  [],
        "extraction_errors": [],
        "resolved_entities": [],
        "resolved_relations": [],
        "unresolved_count": 0,
        "neo4j_stats":      {},
        "validation_report": None,
        "is_consistent":    False,
        "export_paths":     {},
        "errors":           [],
        "current_phase":    "START",
        "requires_human_review": False,
        "start_time":       time.time(),
    }

    final_state = app.invoke(initial_state)
    return final_state
