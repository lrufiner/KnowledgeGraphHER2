"""Tests for URI resolution."""
import pytest
from src.extraction.resolution import resolve_uri, resolve_entity, resolve_all_entities
from src.domain.models import EntityModel, NodeType


class TestResolveUri:
    def test_exact_match_clinical_category(self):
        r = resolve_uri("HER2-Positive")
        assert r["resolved_uri"] == "NCIt:C68748"
        assert r["ncit_uri"] == "NCIt:C68748"

    def test_exact_match_ihc_score(self):
        r = resolve_uri("IHC 2+")
        assert r["resolved_uri"] == "NCIt:C173789"

    def test_candidate_uri_used_when_valid(self):
        r = resolve_uri("Some Entity", candidate_uri="NCIt:C99999")
        assert r["resolved_uri"] == "NCIt:C99999"
        assert r["ncit_uri"] == "NCIt:C99999"

    def test_fuzzy_match(self):
        # Slightly different capitalisation
        r = resolve_uri("HER2 Positive")
        assert "NCIt:" in r["resolved_uri"]

    def test_fallback_local_uri(self):
        r = resolve_uri("Completely Unknown Entity XYZ123")
        assert r["resolved_uri"].startswith("her2:")

    def test_tdxd_resolves_correctly(self):
        r = resolve_uri("T-DXd")
        assert r["resolved_uri"] == "NCIt:C155379"


class TestResolveEntity:
    def test_basic_resolution(self):
        ent = EntityModel(
            id="e1", label="HER2-Low", type=NodeType.CLINICAL_CATEGORY, confidence=0.9,
        )
        resolved = resolve_entity(ent, source_doc="test_chunk")
        assert resolved.ncit_uri == "NCIt:C173791"
        assert resolved.source_doc == "test_chunk"
        assert resolved.confidence == 0.9

    def test_deduplication(self):
        entities = [
            EntityModel(id="e1", label="HER2-Positive", type=NodeType.CLINICAL_CATEGORY, confidence=0.8),
            EntityModel(id="e2", label="HER2-Positive", type=NodeType.CLINICAL_CATEGORY, confidence=0.95),
        ]
        resolved = resolve_all_entities(entities, source_doc="test", deduplicate=True)
        assert len(resolved) == 1
        assert resolved[0].confidence == 0.95  # highest confidence kept
