"""
Tests for extraction layer: algorithm_definitions, algorithm_parser, and entity_extractor.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Tests for algorithm_definitions
# ---------------------------------------------------------------------------

class TestAlgorithmDefinitions:
    def test_all_algorithms_present(self):
        from src.domain.algorithm_definitions import ALL_ALGORITHMS, IHC_ALGORITHM_ID, ISH_ALGORITHM_ID, RAKHA_ALGORITHM_ID
        assert IHC_ALGORITHM_ID in ALL_ALGORITHMS
        assert ISH_ALGORITHM_ID in ALL_ALGORITHMS
        assert RAKHA_ALGORITHM_ID in ALL_ALGORITHMS

    def test_ihc_algorithm_has_entry_node(self):
        from src.domain.algorithm_definitions import IHC_ALGORITHM_NODES
        node_ids = [n["id"] for n in IHC_ALGORITHM_NODES]
        assert "IHC_ENTRY" in node_ids

    def test_ihc_algorithm_has_all_scores(self):
        """IHC algorithm must have result nodes for all 5 IHC scores."""
        from src.domain.algorithm_definitions import IHC_ALGORITHM_NODES, IHC_ALGORITHM_EDGES
        # Collect all referenced IDs (from_id, to_id, if_yes result, if_no result)
        referenced_ids: set[str] = {n["id"] for n in IHC_ALGORITHM_NODES}
        for edge in IHC_ALGORITHM_EDGES:
            referenced_ids.add(edge["from_id"])
            referenced_ids.add(edge["to_id"])

        expected_scores = {"Score3Plus", "Score2Plus", "Score1Plus", "Score0Plus", "Score0"}
        for score in expected_scores:
            assert score in referenced_ids, f"IHC algorithm missing reference to {score}"

    def test_ish_algorithm_covers_all_groups(self):
        """ISH algorithm must mention all 5 ISH groups (in question text, node IDs, or result fields)."""
        from src.domain.algorithm_definitions import ISH_ALGORITHM_NODES
        # Build a single searchable corpus from all text fields
        corpus = " ".join(
            " ".join([
                n.get("id", ""),
                n.get("question", ""),
                n.get("result", "") or "",
                n.get("label", "") or "",
            ])
            for n in ISH_ALGORITHM_NODES
        )
        for i in range(1, 6):
            assert f"Group {i}" in corpus or f"Group{i}" in corpus or f"G{i}_" in corpus, \
                   f"ISH algorithm missing any reference to Group {i}"

    def test_rakha_algorithm_has_0plus_node(self):
        """Rakha 2026 algorithm must include the new 0+ / ultralow score."""
        from src.domain.algorithm_definitions import RAKHA_MATRIX_NODES
        results = [n.get("if_no", {}) for n in RAKHA_MATRIX_NODES if isinstance(n.get("if_no"), dict)]
        results += [n.get("if_yes", {}) for n in RAKHA_MATRIX_NODES if isinstance(n.get("if_yes"), dict)]
        all_results = " ".join(str(r.get("result", "")) for r in results)
        assert "Score0Plus" in all_results, "Rakha algorithm must reference Score0Plus (ultralow)"

    def test_each_node_has_required_fields(self):
        from src.domain.algorithm_definitions import ALL_ALGORITHMS
        required = {"id", "question", "type", "algorithm_id", "node_order", "guideline_source"}
        for algo_id, algo in ALL_ALGORITHMS.items():
            for node in algo["nodes"]:
                missing = required - set(node.keys())
                assert not missing, f"Algorithm '{algo_id}' node '{node.get('id')}' missing: {missing}"

    def test_no_duplicate_node_ids_within_algorithm(self):
        from src.domain.algorithm_definitions import ALL_ALGORITHMS
        for algo_id, algo in ALL_ALGORITHMS.items():
            node_ids = [n["id"] for n in algo["nodes"]]
            assert len(node_ids) == len(set(node_ids)), \
                f"Algorithm '{algo_id}' has duplicate node IDs"

    def test_edge_references_are_consistent(self):
        """All IHC edge from_id / to_id must exist in the combined node pool."""
        from src.domain.algorithm_definitions import (
            IHC_ALGORITHM_EDGES, IHC_ALGORITHM_NODES, ALL_ALGORITHMS,
        )
        from src.domain.ontology import SEED_ENTITIES

        # All node IDs from all algorithms + all seed entity IDs
        all_node_ids: set[str] = {n["id"] for n in SEED_ENTITIES}
        for algo in ALL_ALGORITHMS.values():
            all_node_ids.update(n["id"] for n in algo["nodes"])

        for edge in IHC_ALGORITHM_EDGES:
            from_id = edge["from_id"]
            to_id   = edge["to_id"]
            assert from_id in all_node_ids, f"Edge from_id '{from_id}' not in any node pool"
            assert to_id   in all_node_ids, f"Edge to_id '{to_id}' not in any node pool"


# ---------------------------------------------------------------------------
# Tests for algorithm_parser (unit tests with mocked Neo4j)
# ---------------------------------------------------------------------------

class TestAlgorithmParser:
    def _make_mock_driver(self):
        """Create a mock Neo4j driver/session that silently accepts all queries."""
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.run = MagicMock(return_value=MagicMock())

        mock_driver = MagicMock()
        mock_driver.session = MagicMock(return_value=mock_session)
        return mock_driver, mock_session

    def test_parse_ihc_algorithm_returns_stats(self):
        from src.extraction.algorithm_parser import parse_and_load_algorithm
        from src.domain.algorithm_definitions import IHC_ALGORITHM_ID

        mock_driver, mock_session = self._make_mock_driver()
        stats = parse_and_load_algorithm(mock_driver, IHC_ALGORITHM_ID, verbose=False)

        assert "nodes" in stats
        assert "edges" in stats
        assert stats["nodes"] > 0, "Should create at least one node"
        assert stats["edges"] > 0, "Should create at least one edge"

    def test_parse_all_algorithms_returns_aggregate(self):
        from src.extraction.algorithm_parser import parse_and_load_all_algorithms

        mock_driver, _ = self._make_mock_driver()
        stats = parse_and_load_all_algorithms(mock_driver, verbose=False)

        assert stats["nodes"] > 0
        assert stats["edges"] > 0

    def test_parse_unknown_algorithm_raises(self):
        from src.extraction.algorithm_parser import parse_and_load_algorithm

        mock_driver, _ = self._make_mock_driver()
        with pytest.raises(ValueError, match="not found"):
            parse_and_load_algorithm(mock_driver, "NON_EXISTENT_ALGO", verbose=False)

    def test_ihc_algorithm_creates_correct_node_count(self):
        from src.extraction.algorithm_parser import parse_and_load_algorithm
        from src.domain.algorithm_definitions import IHC_ALGORITHM_ID, IHC_ALGORITHM_NODES

        mock_driver, mock_session = self._make_mock_driver()
        stats = parse_and_load_algorithm(mock_driver, IHC_ALGORITHM_ID, verbose=False)

        # Should upsert exactly len(IHC_ALGORITHM_NODES) nodes
        assert stats["nodes"] == len(IHC_ALGORITHM_NODES)

    def test_all_algorithms_parsed_without_error(self):
        """Smoke test: parsing all algorithms must not throw exceptions."""
        from src.extraction.algorithm_parser import parse_and_load_all_algorithms
        from src.domain.algorithm_definitions import ALL_ALGORITHMS

        mock_driver, _ = self._make_mock_driver()
        # Should complete without raising
        stats = parse_and_load_all_algorithms(mock_driver, verbose=False)
        assert stats["nodes"] >= len(ALL_ALGORITHMS)  # at least 1 node per algorithm


# ---------------------------------------------------------------------------
# Tests for resolution module (supplementary coverage)
# ---------------------------------------------------------------------------

class TestResolutionEdgeCases:
    def test_none_candidate_triggers_fuzzy(self):
        from src.extraction.resolution import resolve_uri
        result = resolve_uri("IHC 3+", candidate_uri=None)
        assert result["resolved_uri"] is not None

    def test_all_seed_entity_ids_resolve_without_fallback(self):
        """Every seed entity label should resolve to a canonical (non-fallback) URI."""
        from src.domain.ontology import SEED_ENTITIES
        from src.extraction.resolution import resolve_uri

        local_count = 0
        for ent in SEED_ENTITIES:
            result = resolve_uri(ent["label"], candidate_uri=ent.get("ncit_uri"))
            if result["resolved_uri"] and result["resolved_uri"].startswith("her2:"):
                if not result.get("ncit_uri"):
                    local_count += 1

        # Allow up to 10 entities to have only local URIs (fractal, trials, etc.)
        assert local_count <= 10, \
            f"{local_count} seed entities only got local URIs — too many"

    def test_guideline_labels_resolve(self):
        from src.extraction.resolution import resolve_uri
        guidelines = ["ASCO_CAP_2023", "CAP_2025", "ESMO_2023", "Rakha_2026"]
        for gl in guidelines:
            r = resolve_uri(gl)
            assert r["resolved_uri"] is not None, f"Guideline '{gl}' failed to resolve"
