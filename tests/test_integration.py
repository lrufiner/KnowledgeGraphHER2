"""
Integration tests — require a running Neo4j instance.

These tests are skipped automatically when Neo4j is not reachable.
In CI they run only on `main` branch (see .github/workflows/ci.yml).

Run locally:
    python -m pytest tests/test_integration.py -v
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Schema & seed
# ---------------------------------------------------------------------------

class TestNeo4jSchema:
    """Verify schema creation is idempotent and seed data loads correctly."""

    def test_initialize_schema_runs_without_error(self, neo4j_driver):
        from src.graph.neo4j_builder import initialize_schema
        initialize_schema(neo4j_driver)  # Should not raise

    def test_load_seed_data_creates_core_nodes(self, neo4j_driver):
        from src.graph.neo4j_builder import initialize_schema, load_seed_data
        initialize_schema(neo4j_driver)
        load_seed_data(neo4j_driver)

        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (n:ClinicalCategory) RETURN count(n) AS cnt"
            )
            count = result.single()["cnt"]
        assert count >= 6, f"Expected ≥6 ClinicalCategory nodes, got {count}"

    def test_seed_relations_exist(self, neo4j_driver):
        from src.graph.neo4j_builder import initialize_schema, load_seed_data
        initialize_schema(neo4j_driver)
        load_seed_data(neo4j_driver)

        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH ()-[r:eligibleFor]->() RETURN count(r) AS cnt"
            )
            rels = result.single()["cnt"]
        assert rels >= 1, "Expected at least 1 eligibleFor relation"

    def test_idempotent_seed_double_load(self, neo4j_driver):
        """Calling load_seed_data twice should not duplicate nodes."""
        from src.graph.neo4j_builder import initialize_schema, load_seed_data
        initialize_schema(neo4j_driver)
        load_seed_data(neo4j_driver)
        load_seed_data(neo4j_driver)  # second call

        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (n:IHCScore) RETURN count(n) AS cnt"
            )
            count = result.single()["cnt"]
        assert count == 5, f"Expected exactly 5 IHCScore nodes after double seed load, got {count}"


# ---------------------------------------------------------------------------
# Entity upsert
# ---------------------------------------------------------------------------

class TestEntityUpsert:
    def test_upsert_single_entity(self, neo4j_driver):
        from src.graph.neo4j_builder import initialize_schema, upsert_entities
        from src.domain.models import NodeType
        from src.domain.models import ResolvedEntity

        initialize_schema(neo4j_driver)
        entity = ResolvedEntity(
            id="TestEntity_Integration",
            label="Test Entity",
            type=NodeType.BIOMARKER,
            definition="Integration test placeholder",
            source_quote=None,
            resolved_uri="her2:test_entity_integration",
            ncit_uri=None,
            snomed_uri=None,
            loinc_code=None,
            source_doc="test_integration.py",
            confidence=1.0,
        )
        upsert_entities(neo4j_driver, [entity])

        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (n:Biomarker {id: $id}) RETURN n.label AS label",
                id="TestEntity_Integration",
            )
            row = result.single()
        assert row is not None
        assert row["label"] == "Test Entity"

        # Cleanup
        with neo4j_driver.session() as session:
            session.run("MATCH (n:Biomarker {id: 'TestEntity_Integration'}) DETACH DELETE n")


# ---------------------------------------------------------------------------
# Validation pipeline
# ---------------------------------------------------------------------------

class TestValidationPipeline:
    def test_validation_passes_after_seed(self, neo4j_driver):
        from src.graph.neo4j_builder import initialize_schema, load_seed_data
        from src.graph.validator import run_validation

        initialize_schema(neo4j_driver)
        load_seed_data(neo4j_driver)

        report = run_validation(neo4j_driver)
        assert report is not None
        # After seed load all validation rules should pass
        from src.domain.models import ValidationSeverity
        passed = [r for r in report.results if r.valid]
        failed_critical = [
            r for r in report.results
            if not r.valid and r.severity == ValidationSeverity.CRITICAL
        ]
        assert len(failed_critical) == 0, (
            f"Critical validation failures after seed load: "
            f"{[r.rule_id for r in failed_critical]}"
        )
        assert len(passed) >= 10, f"Expected >=10 passing rules, got {len(passed)}"


# ---------------------------------------------------------------------------
# Graph stats
# ---------------------------------------------------------------------------

class TestGraphStats:
    def test_stats_returns_positive_counts(self, neo4j_driver):
        from src.graph.neo4j_builder import get_graph_stats, initialize_schema, load_seed_data

        initialize_schema(neo4j_driver)
        load_seed_data(neo4j_driver)
        stats = get_graph_stats(neo4j_driver)

        assert stats["total_nodes"] > 0
        assert stats["total_relations"] > 0
        assert "ClinicalCategory" in stats.get("node_counts", {})
