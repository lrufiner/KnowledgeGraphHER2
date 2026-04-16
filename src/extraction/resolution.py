"""
URI resolution module — three-tier strategy.

Tier 1: Curated lookup table (CANONICAL_URIS from ontology.py)
Tier 2: Fuzzy label match against the same table (edit distance)
Tier 3: Return a local her2: URI derived from the entity label

External API calls (BioPortal, NCIt SPARQL) are optional and gated
behind flags to avoid network dependency during local/cost-saving runs.
"""
from __future__ import annotations

import re
from difflib import get_close_matches

from src.domain.models import EntityModel, NodeType, ResolvedEntity
from src.domain.ontology import CANONICAL_URIS, HER2_BASE, SNOMED_URIS


def _safe_id(label: str) -> str:
    """Convert a label to a safe lowercase_underscore ID."""
    return re.sub(r'[^a-z0-9_]', '_', label.lower().strip()).strip('_')


def resolve_uri(label: str, candidate_uri: str | None = None) -> dict[str, str | None]:
    """
    Resolve a label to canonical URIs using the three-tier strategy.

    Returns:
        dict with keys: resolved_uri, ncit_uri, snomed_uri, loinc_code
    """
    result: dict[str, str | None] = {
        "resolved_uri": None,
        "ncit_uri":     None,
        "snomed_uri":   None,
        "loinc_code":   None,
    }

    # ── Tier 1: Exact match in canonical table ────────────────────────────
    if candidate_uri and (
        candidate_uri.startswith("NCIt:") or
        candidate_uri.startswith("her2:") or
        candidate_uri.startswith("snomed:")
    ):
        result["resolved_uri"] = candidate_uri
        if candidate_uri.startswith("NCIt:"):
            result["ncit_uri"] = candidate_uri
        elif candidate_uri.startswith("snomed:"):
            result["snomed_uri"] = candidate_uri
        return result

    if label in CANONICAL_URIS:
        uri = CANONICAL_URIS[label]
        result["resolved_uri"] = uri
        if uri.startswith("NCIt:"):
            result["ncit_uri"] = uri
        elif uri.startswith("snomed:"):
            result["snomed_uri"] = uri
        # Also try SNOMED
        if label in SNOMED_URIS:
            result["snomed_uri"] = SNOMED_URIS[label]
        return result

    # ── Tier 2: Fuzzy match (edit distance) ───────────────────────────────
    all_labels = list(CANONICAL_URIS.keys())
    close = get_close_matches(label, all_labels, n=1, cutoff=0.80)
    if close:
        uri = CANONICAL_URIS[close[0]]
        result["resolved_uri"] = uri
        if uri.startswith("NCIt:"):
            result["ncit_uri"] = uri
        return result

    # Case-insensitive exact match
    label_lower = label.lower()
    for known_label, uri in CANONICAL_URIS.items():
        if known_label.lower() == label_lower:
            result["resolved_uri"] = uri
            if uri.startswith("NCIt:"):
                result["ncit_uri"] = uri
            return result

    # ── Tier 3: Local her2: URI ───────────────────────────────────────────
    local_id = _safe_id(label)
    result["resolved_uri"] = f"her2:{local_id}"
    return result


def resolve_entity(entity: EntityModel, source_doc: str) -> ResolvedEntity:
    """
    Resolve a single extracted entity to a ResolvedEntity with canonical URI.
    """
    uris = resolve_uri(entity.label, entity.candidate_uri)
    return ResolvedEntity(
        id=entity.label.replace(" ", "_").replace("/", "_"),
        label=entity.label,
        type=entity.type,
        definition=entity.definition,
        source_quote=entity.source_quote,
        resolved_uri=uris["resolved_uri"] or f"her2:{_safe_id(entity.label)}",
        ncit_uri=uris["ncit_uri"],
        snomed_uri=uris["snomed_uri"],
        loinc_code=uris["loinc_code"],
        source_doc=source_doc,
        confidence=entity.confidence,
        interobserver_variability=entity.interobserver_variability,
        evidence_level=entity.evidence_level,
    )


def resolve_all_entities(
    entities: list[EntityModel],
    source_doc: str,
    deduplicate: bool = True,
) -> list[ResolvedEntity]:
    """
    Resolve a list of extracted entities.
    If deduplicate=True, entities with the same resolved_uri are merged
    (highest confidence wins, definitions appended if different).
    """
    resolved = [resolve_entity(e, source_doc) for e in entities]

    if not deduplicate:
        return resolved

    # Deduplicate by resolved_uri, keeping highest confidence
    seen: dict[str, ResolvedEntity] = {}
    for ent in resolved:
        key = ent.resolved_uri
        if key not in seen or ent.confidence > seen[key].confidence:
            seen[key] = ent
        else:
            # Merge definitions if different
            existing = seen[key]
            if (ent.definition and existing.definition and
                    ent.definition != existing.definition):
                existing.definition = (
                    existing.definition + " | " + ent.definition
                )
    return list(seen.values())


def build_id_map(resolved: list[ResolvedEntity]) -> dict[str, str]:
    """
    Build a mapping from local extraction IDs (e1, e2, ...) to resolved entity IDs.
    Needed to translate relation subject_id/object_id after resolution.

    Since extraction IDs are local (per chunk), we can only build this map
    per chunk. Returns {local_id: resolved_entity.id} based on label matching.
    """
    # This map is built by the caller with chunk-level context.
    # Here we provide a helper to create it from entities that still carry
    # their original extraction IDs.
    return {}  # Overridden in pipeline
