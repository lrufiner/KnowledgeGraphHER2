"""Tests for domain models and ontology constants."""
import pytest
from src.domain.models import (
    ContentType, DocumentChunk, EdgeType, EntityModel,
    ExtractionResult, NodeType, RelationModel, ResolvedEntity,
    ValidationReport, ValidationResult, ValidationSeverity,
)
from src.domain.ontology import (
    CANONICAL_URIS, SEED_ENTITIES, SEED_RELATIONS, TOY_FRACTAL_SPECIMENS,
)


class TestModels:
    def test_document_chunk_to_dict(self):
        chunk = DocumentChunk(
            chunk_id="test_c0001",
            source_doc="annex_guidelines.md",
            section="IHC Scoring",
            content="IHC 3+ implies HER2-Positive",
            content_type=ContentType.CRITERIA,
        )
        d = chunk.to_dict()
        assert d["chunk_id"] == "test_c0001"
        assert d["content_type"] == "criteria"

    def test_entity_model_validation(self):
        ent = EntityModel(
            id="e1",
            label="HER2-Positive",
            type=NodeType.CLINICAL_CATEGORY,
            confidence=0.95,
        )
        assert ent.confidence == 0.95
        assert ent.type == NodeType.CLINICAL_CATEGORY

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            EntityModel(id="e1", label="X", type=NodeType.IHC_SCORE, confidence=1.5)

    def test_validation_report_consistency(self):
        report = ValidationReport()
        # Critical failure → is_consistent=False
        report.add(ValidationResult(
            rule_id="test_rule",
            valid=False,
            severity=ValidationSeverity.CRITICAL,
            message="Critical failure",
            source="test",
        ))
        assert not report.is_consistent
        summary = report.summary()
        assert summary["failed"] == 1
        assert summary["critical"] == 1

    def test_validation_report_medium_stays_consistent(self):
        report = ValidationReport()
        report.add(ValidationResult(
            rule_id="medium_rule",
            valid=False,
            severity=ValidationSeverity.MEDIUM,
            message="Medium issue",
            source="test",
        ))
        # Medium failures don't break consistency
        assert report.is_consistent


class TestOntology:
    def test_canonical_uris_not_empty(self):
        assert len(CANONICAL_URIS) > 20

    def test_all_clinical_categories_in_canonical(self):
        required = ["HER2-Positive", "HER2-Negative", "HER2-Low", "HER2-Ultralow", "HER2-Null", "HER2-Equivocal"]
        for cat in required:
            assert cat in CANONICAL_URIS, f"Missing: {cat}"

    def test_seed_entities_have_required_fields(self):
        for ent in SEED_ENTITIES:
            assert "id" in ent, f"Missing id in {ent}"
            assert "label" in ent, f"Missing label in {ent}"
            assert "type" in ent, f"Missing type in {ent}"

    def test_seed_relations_have_required_fields(self):
        for rel in SEED_RELATIONS:
            assert "subject_id" in rel
            assert "predicate" in rel
            assert "object_id" in rel

    def test_toy_fractal_specimens_have_metrics(self):
        for spec in TOY_FRACTAL_SPECIMENS:
            assert "D0" in spec
            assert "D1" in spec
            assert "Lacunarity" in spec
            assert "ihc_score" in spec

    def test_inconsistent_specimen_present(self):
        """Ensure toy data includes at least one inconsistent specimen for alert testing."""
        inconsistent = [s for s in TOY_FRACTAL_SPECIMENS
                        if "INCONSISTENT" in s["specimen_id"]]
        assert len(inconsistent) >= 1

    def test_all_ihc_scores_present_in_seeds(self):
        seed_ids = {e["id"] for e in SEED_ENTITIES}
        for score_id in ["Score0", "Score0Plus", "Score1Plus", "Score2Plus", "Score3Plus"]:
            assert score_id in seed_ids, f"Missing seed IHCScore: {score_id}"

    def test_all_ish_groups_present_in_seeds(self):
        seed_ids = {e["id"] for e in SEED_ENTITIES}
        for grp in ["Group1", "Group2", "Group3", "Group4", "Group5"]:
            assert grp in seed_ids, f"Missing seed ISHGroup: {grp}"
