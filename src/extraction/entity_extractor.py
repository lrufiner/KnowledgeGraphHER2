"""
LLM-based entity and relation extractor.

Supports Claude, OpenAI, and Ollama via LangChain BaseChatModel.
Uses structured JSON output with schema constraints from the HER2 ontology.
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.domain.models import (
    ContentType, DocumentChunk, EdgeType, EntityModel,
    ExtractionResult, NodeType, RelationModel,
)

# ---------------------------------------------------------------------------
# System prompt — schema-constrained extraction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert in oncology pathology and biomedical ontologies.
Your task is to extract entities and semantic relations from clinical document fragments about the HER2 receptor in breast cancer.

VALID ENTITY TYPES:
- ClinicalCategory: HER2 clinical status (Positive, Negative, Low, Ultralow, Null, Equivocal)
- IHCScore: IHC staining score (Score0, Score0Plus, Score1Plus, Score2Plus, Score3Plus)
- ISHGroup: ISH group 1–5 with numeric criteria
- StainingPattern: Staining characteristics (intensity, circumferentiality, percentage)
- TherapeuticAgent: Drug or treatment (Trastuzumab, TrastuzumabDeruxtecan, Pertuzumab, TDM1)
- ClinicalTrial: Clinical trial (DESTINY_Breast04, DESTINY_Breast06, DAISY_Trial)
- Biomarker: Molecular marker (HER2/ERBB2, CEP17, ER, PgR, Ki67)
- Guideline: Clinical guideline (ASCO_CAP_2023, CAP_Biomarker_2025, ESMO_2023, Rakha_International_2026)
- QualityMeasure: QA requirement (fixation time, section age, controls, EQA)
- FractalMetric: Fractal measurement (FractalDimension_D0, FractalDimension_D1, Lacunarity, MultifractalSpread)
- PathologicalFeature: Pathological characteristic (ArchitecturalComplexity, IntratumoralHeterogeneity)
- Assay: Diagnostic test (IHC, ISH, FISH, VENTANA_HER2_4B5, HercepTest)
- DiagnosticDecision: Decision node in an algorithm tree
- Threshold: Numeric threshold (ratio, percentage, signals/cell)

VALID PREDICATES:
- implies: IHCScore/ISHGroup → ClinicalCategory (diagnostic implication)
- requiresReflexTest: IHCScore → Assay (mandatory follow-up test)
- eligibleFor: ClinicalCategory → TherapeuticAgent
- notEligibleFor: ClinicalCategory → TherapeuticAgent
- definedIn: Entity → Guideline or ClinicalTrial
- hasQualityRequirement: Assay → QualityMeasure
- associatedWith: FractalMetric → PathologicalFeature
- proposedEquivalence: FractalMetric → ClinicalCategory (HYPOTHESIS only)
- inconsistentWith: for fractal-clinical inconsistency alerts
- hasThreshold: ISHGroup or DiagnosticDecision → Threshold
- leadsTo: DiagnosticDecision → DiagnosticDecision (algorithm flow)
- overrides: newer Guideline → older Guideline
- supportedByEvidence: Entity → ClinicalTrial
- hasStainingPattern: IHCScore → StainingPattern
- refinesCategory: ClinicalCategory → parent ClinicalCategory
- contradictsIfConcurrent: for conflicting concurrent findings

CRITICAL RULES:
1. Return ONLY a valid JSON object — no markdown, no preamble, no trailing text.
2. Every entity must have: id, label, type, definition, candidate_uri (or null), confidence (0–1).
3. Every relation must have: subject_id, predicate, object_id, confidence, evidence.
4. Confidence = 1.0 only for explicit guideline statements. Use 0.7–0.9 for strong inference, 0.5–0.7 for uncertain.
5. Include source_quote (exact text) for every entity definition.
6. Mark fractal proposedEquivalence with lower confidence (≤0.7).
7. Do NOT invent entities not present in the fragment."""

# ---------------------------------------------------------------------------
# Few-shot example (hardcoded HER2 clinical example)
# ---------------------------------------------------------------------------

_FEW_SHOT_EXAMPLE = {
    "input": (
        "HER2-ultralow was defined as faint or barely perceptible, incomplete membrane staining "
        "in >0% to ≤10% of tumor cells (IHC score 0+/with membrane staining). "
        "T-DXd has shown benefit in HR-positive HER2-ultralow metastatic breast cancer "
        "in the DESTINY-Breast06 trial."
    ),
    "output": {
        "entities": [
            {"id": "e1", "label": "HER2-Ultralow", "type": "ClinicalCategory",
             "definition": "IHC 0+ (faint/barely perceptible, incomplete membrane staining in >0% to ≤10% of tumor cells)",
             "source_quote": "faint or barely perceptible, incomplete membrane staining in >0% to ≤10% of tumor cells",
             "candidate_uri": "NCIt:C173792", "confidence": 1.0},
            {"id": "e2", "label": "IHC Score 0+", "type": "IHCScore",
             "definition": "Faint/barely perceptible incomplete membrane staining in >0–≤10% tumor cells",
             "source_quote": "IHC score 0+/with membrane staining",
             "candidate_uri": "NCIt:C173787", "confidence": 1.0},
            {"id": "e3", "label": "Trastuzumab Deruxtecan", "type": "TherapeuticAgent",
             "definition": "Anti-HER2 antibody-drug conjugate",
             "source_quote": "T-DXd", "candidate_uri": "NCIt:C155379", "confidence": 1.0},
            {"id": "e4", "label": "DESTINY-Breast06", "type": "ClinicalTrial",
             "definition": "Phase 3 RCT of T-DXd in HR+ HER2-ultralow metastatic breast cancer",
             "source_quote": "DESTINY-Breast06 trial", "candidate_uri": None, "confidence": 1.0},
        ],
        "relations": [
            {"subject_id": "e2", "predicate": "implies", "object_id": "e1",
             "confidence": 1.0, "evidence": "Diagnostic definition from DESTINY-Breast06"},
            {"subject_id": "e1", "predicate": "eligibleFor", "object_id": "e3",
             "confidence": 0.95, "evidence": "T-DXd benefit shown in DESTINY-Breast06"},
            {"subject_id": "e1", "predicate": "supportedByEvidence", "object_id": "e4",
             "confidence": 1.0, "evidence": "Primary trial source"},
        ]
    }
}


def _build_messages(chunk: DocumentChunk) -> list:
    few_shot = (
        f"EXAMPLE INPUT:\n{_FEW_SHOT_EXAMPLE['input']}\n\n"
        f"EXAMPLE OUTPUT:\n{json.dumps(_FEW_SHOT_EXAMPLE['output'], indent=2, ensure_ascii=False)}"
    )
    user_content = (
        f"Source document: {chunk.source_doc}\n"
        f"Section: {chunk.section}\n"
        f"Content type: {chunk.content_type.value}\n\n"
        f"{few_shot}\n\n"
        f"NOW EXTRACT FROM THIS FRAGMENT:\n{chunk.content}\n\n"
        "Return ONLY JSON."
    )
    return [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]


def _clean_json(raw: str) -> str:
    """Strip markdown code fences and thinking blocks if LLM added them."""
    raw = raw.strip()
    # Strip <think>...</think> blocks (qwen3, gemma4 thinking models)
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    raw = raw.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    return raw.strip()


def _parse_extraction(raw: str) -> tuple[list[EntityModel], list[RelationModel], str | None]:
    """Parse raw LLM output into typed models. Returns (entities, relations, error)."""
    try:
        data = json.loads(_clean_json(raw))
    except json.JSONDecodeError as e:
        return [], [], f"JSONDecodeError: {e} | raw[:200]={raw[:200]}"

    entities: list[EntityModel] = []
    for e in data.get("entities", []):
        try:
            # Normalise type field
            raw_type = e.get("type", "")
            try:
                node_type = NodeType(raw_type)
            except ValueError:
                node_type = NodeType.CLINICAL_CATEGORY  # safe fallback
            entities.append(EntityModel(
                id=e["id"],
                label=e.get("label", ""),
                type=node_type,
                definition=e.get("definition"),
                source_quote=e.get("source_quote"),
                candidate_uri=e.get("candidate_uri"),
                confidence=float(e.get("confidence", 1.0)),
            ))
        except Exception as ex:
            pass  # skip malformed entities

    relations: list[RelationModel] = []
    for r in data.get("relations", []):
        try:
            raw_pred = r.get("predicate", "")
            try:
                edge_type = EdgeType(raw_pred)
            except ValueError:
                continue  # skip unknown predicates
            relations.append(RelationModel(
                subject_id=r["subject_id"],
                predicate=edge_type,
                object_id=r["object_id"],
                confidence=float(r.get("confidence", 1.0)),
                evidence=r.get("evidence"),
            ))
        except Exception:
            pass

    return entities, relations, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _extract_with_retry(chunk: DocumentChunk, llm, messages: list) -> tuple[str, Exception | None]:
    """Invoke the LLM; retry once on empty/None response."""
    for attempt in range(2):
        try:
            response = llm.invoke(messages)
            raw = response.content if hasattr(response, "content") else str(response)
            raw = raw.strip() if raw else ""
            if raw:
                return raw, None
        except Exception as e:
            if attempt == 1:
                return "", e
    return "", None


def extract_from_chunk(chunk: DocumentChunk, llm) -> ExtractionResult:
    """
    Extract entities and relations from a single DocumentChunk using the LLM.

    Args:
        chunk: The document chunk to process.
        llm:   A LangChain BaseChatModel (Claude / OpenAI / Ollama).

    Returns:
        ExtractionResult with entities, relations, and optional error.
    """
    messages = _build_messages(chunk)
    raw, exc = _extract_with_retry(chunk, llm, messages)
    if exc:
        return ExtractionResult(
            chunk_id=chunk.chunk_id,
            section=chunk.section,
            error=f"LLM call failed: {exc}",
        )
    if not raw:
        return ExtractionResult(
            chunk_id=chunk.chunk_id,
            section=chunk.section,
            entities=[], relations=[], error=None,
        )

    entities, relations, error = _parse_extraction(raw)
    return ExtractionResult(
        chunk_id=chunk.chunk_id,
        section=chunk.section,
        entities=entities,
        relations=relations,
        error=error,
        raw_response=raw if error else None,
    )


def extract_batch(
    chunks: list[DocumentChunk],
    llm,
    skip_types: set[ContentType] | None = None,
    verbose: bool = True,
) -> list[ExtractionResult]:
    """
    Extract from a list of chunks sequentially.
    Optionally skip chunks of certain content types (e.g., pure ontology code).
    """
    if skip_types is None:
        # Skip raw turtle/OWL code blocks — they're already in ontology.py
        skip_types = {ContentType.ONTOLOGY}

    results: list[ExtractionResult] = []
    for i, chunk in enumerate(chunks, 1):
        if chunk.content_type in skip_types:
            continue
        if verbose:
            print(f"[{i}/{len(chunks)}] Extracting: {chunk.chunk_id} ({chunk.content_type.value})")
        result = extract_from_chunk(chunk, llm)
        if result.error and verbose:
            print(f"  [!] Error: {result.error[:120]}")
        else:
            ent_count = len(result.entities)
            rel_count = len(result.relations)
            if verbose:
                print(f"  [OK] {ent_count} entities, {rel_count} relations")
        results.append(result)
    return results
