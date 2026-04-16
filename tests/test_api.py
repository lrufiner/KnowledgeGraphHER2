"""
Tests for Sprint 4: FastAPI endpoints, GroundingChecker, and LightRAGWrapper.

Coverage:
    TestGroundingChecker     (9 tests) — entity drift, category validation, rule enforcement
    TestLightRAGWrapper      (4 tests) — availability flag, fallback result, convenience methods
    TestHealthEndpoint       (1 test)  — GET /health
    TestDiagnoseEndpoint     (6 tests) — POST /diagnose — all IHC scores, validation, ISH cases
    TestValidateEndpoint     (4 tests) — POST /validate — PASS/FAIL, human review flag
    TestEvidenceEndpoint     (5 tests) — GET /evidence/{category} — all 5 categories, 404
    TestStatsEndpoint        (2 tests) — GET /stats — no Neo4j, response shape
    TestQueryEndpoint        (3 tests) — POST /query — mock LLM required

Total: 34 tests
"""
from __future__ import annotations

import sys
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Grounding tests
# ---------------------------------------------------------------------------


class TestGroundingChecker:
    """Tests for src/retrieval/grounding.py"""

    def _make_checker(self):
        from src.retrieval.grounding import GroundingChecker
        return GroundingChecker(driver=None)

    def _make_empty_context(self):
        from src.retrieval.grounding import GroundedContext
        return GroundedContext()

    def test_build_context_fallback_no_driver(self):
        checker = self._make_checker()
        ctx = checker.build_context(entity_ids=["Score2Plus"])
        assert ctx is not None
        # Fallback context should have populated entity_labels
        assert len(ctx.entity_labels) > 0

    def test_build_context_includes_entity_ids(self):
        checker = self._make_checker()
        ctx = checker.build_context(entity_ids=["Score2Plus", "Group3"])
        entity_ids = [e["id"] for e in ctx.entities]
        assert "Score2Plus" in entity_ids
        assert "Group3" in entity_ids

    def test_entity_ids_from_case_ihc(self):
        checker = self._make_checker()
        ids = checker._entity_ids_from_case({"ihc_score": "3+"}, [])
        assert "Score3Plus" in ids

    def test_entity_ids_from_case_ish(self):
        checker = self._make_checker()
        ids = checker._entity_ids_from_case({"ihc_score": "2+", "ish_group": "Group3"}, [])
        assert "Score2Plus" in ids
        assert "Group3" in ids

    def test_validate_response_clean(self):
        checker = self._make_checker()
        ctx = checker.build_context(entity_ids=["Score2Plus"])
        response = (
            "IHC 2+ with ISH Group 3 (ratio 1.7) is classified as HER2-Equivocal "
            "per ASCO/CAP 2023. Reflex workup is required."
        )
        result = checker.validate_response(response, ctx)
        assert result.is_grounded

    def test_validate_detects_deprecated_negative(self):
        checker = self._make_checker()
        ctx = self._make_empty_context()
        response = "The patient is HER2-Negative based on IHC 1+."
        result = checker.validate_response(response, ctx)
        assert not result.is_grounded
        assert any("Deprecated" in v for v in result.rule_violations)

    def test_validate_detects_invalid_ihc_score(self):
        checker = self._make_checker()
        ctx = self._make_empty_context()
        response = "The IHC 5+ score indicates strong amplification."
        result = checker.validate_response(response, ctx)
        assert not result.is_grounded or len(result.rule_violations) > 0

    def test_enforce_clinical_rules_ihc3_claims_low(self):
        checker = self._make_checker()
        ctx = self._make_empty_context()
        response = "This IHC 3+ case is HER2-Low and does not require treatment."
        result = checker.validate_response(
            response, ctx, clinical_data={"ihc_score": "3+"}
        )
        assert not result.is_grounded

    def test_grounding_result_severity_levels(self):
        from src.retrieval.grounding import GroundingResult
        clean = GroundingResult(is_grounded=True)
        assert clean.severity == "CLEAN"

        mild = GroundingResult(is_grounded=False, unverified_relations=["r1"])
        assert mild.severity in ("LOW", "MEDIUM")

        severe = GroundingResult(
            is_grounded=False,
            hallucinated_entities=["e1", "e2"],
            rule_violations=["v1", "v2"],
        )
        assert severe.severity in ("MEDIUM", "HIGH")

    def test_grounded_context_to_prompt(self):
        from src.retrieval.grounding import GroundedContext
        ctx = GroundedContext(
            entities=[{"id": "Score2Plus", "label": "IHC 2+", "node_type": "IHCScore"}],
            relations=[{"from_label": "IHC 2+", "relation": "requiresReflexTest",
                        "to_label": "ISH Assay", "confidence": 0.9}],
            source_docs=["ASCO_CAP_2023"],
        )
        prompt = ctx.to_prompt_context()
        assert "GROUNDED CONTEXT" in prompt
        assert "Score2Plus" in prompt
        assert "requiresReflexTest" in prompt


# ---------------------------------------------------------------------------
# LightRAG wrapper tests (no actual LightRAG installed)
# ---------------------------------------------------------------------------


class TestLightRAGWrapper:
    """Tests for src/retrieval/lightrag_wrapper.py"""

    def _make_wrapper(self):
        from src.retrieval.lightrag_wrapper import LightRAGWrapper
        return LightRAGWrapper()

    def test_is_available_returns_bool(self):
        wrapper = self._make_wrapper()
        assert isinstance(wrapper.is_available, bool)

    def test_query_fallback_when_not_installed(self):
        wrapper = self._make_wrapper()
        # If LightRAG isn't installed the wrapper returns a graceful fallback
        result = wrapper.query("test query", mode="hybrid")
        assert result.query == "test query"
        assert result.mode == "hybrid"
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    def test_query_her2_classification_builds_query(self):
        wrapper = self._make_wrapper()
        result = wrapper.query_her2_classification("2+", "Group3")
        assert "2+" in result.query
        assert "Group3" in result.query

    def test_query_therapeutic_eligibility(self):
        wrapper = self._make_wrapper()
        result = wrapper.query_therapeutic_eligibility("HER2-Low")
        assert "HER2-Low" in result.query


# ---------------------------------------------------------------------------
# FastAPI endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture
def api_client():
    """TestClient with stubbed LLM/driver."""
    from fastapi.testclient import TestClient
    from app import api as api_module

    # Stub driver (not connected)
    api_module._driver_singleton = None
    api_module._llm_singleton = None

    client = TestClient(api_module.app, raise_server_exceptions=True)
    yield client

    # Cleanup
    api_module._driver_singleton = None
    api_module._llm_singleton = None


class TestHealthEndpoint:
    def test_health_ok(self, api_client):
        resp = api_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestDiagnoseEndpoint:
    """POST /diagnose — deterministic mode (no LLM)."""

    def test_diagnose_3plus(self, api_client):
        resp = api_client.post("/diagnose", json={"ihc_score": "3+", "narrate": False})
        assert resp.status_code == 200
        data = resp.json()
        assert data["classification"] == "HER2_Positive"
        assert data["confidence"] == "HIGH"

    def test_diagnose_0_null(self, api_client):
        resp = api_client.post("/diagnose", json={"ihc_score": "0", "narrate": False})
        assert resp.status_code == 200
        assert resp.json()["classification"] == "HER2_Null"

    def test_diagnose_0plus_ultralow(self, api_client):
        resp = api_client.post("/diagnose", json={"ihc_score": "0+", "narrate": False})
        assert resp.status_code == 200
        assert resp.json()["classification"] == "HER2_Ultralow"

    def test_diagnose_1plus_low(self, api_client):
        resp = api_client.post("/diagnose", json={"ihc_score": "1+", "narrate": False})
        assert resp.status_code == 200
        assert resp.json()["classification"] == "HER2_Low"

    def test_diagnose_2plus_with_ish_group1(self, api_client):
        resp = api_client.post(
            "/diagnose",
            json={"ihc_score": "2+", "ish_group": "Group1", "ish_ratio": 2.5, "narrate": False},
        )
        assert resp.status_code == 200
        assert resp.json()["classification"] == "HER2_Positive"

    def test_diagnose_2plus_no_ish_equivocal(self, api_client):
        resp = api_client.post("/diagnose", json={"ihc_score": "2+", "narrate": False})
        assert resp.status_code == 200
        cls = resp.json()["classification"]
        assert cls == "HER2_Equivocal"


class TestValidateEndpoint:
    """POST /validate"""

    def test_validate_clean_case_pass(self, api_client):
        resp = api_client.post(
            "/validate", json={"ihc_score": "3+", "ish_group": "Group1"}
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "PASS"

    def test_validate_conflict_ihc3_ish_group5(self, api_client):
        resp = api_client.post(
            "/validate", json={"ihc_score": "3+", "ish_group": "Group5"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "FAIL"
        assert data["needs_human_review"] is True

    def test_validate_returns_issues_list(self, api_client):
        resp = api_client.post(
            "/validate", json={"ihc_score": "3+", "ish_group": "Group5"}
        )
        assert resp.status_code == 200
        assert isinstance(resp.json()["issues"], list)

    def test_validate_1plus_no_ish_clean(self, api_client):
        resp = api_client.post("/validate", json={"ihc_score": "1+"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "PASS"


class TestEvidenceEndpoint:
    """GET /evidence/{category}"""

    def test_evidence_her2_positive(self, api_client):
        resp = api_client.get("/evidence/HER2_Positive")
        assert resp.status_code == 200
        data = resp.json()
        assert data["category"] == "HER2_Positive"
        assert len(data["eligible_agents"]) > 0

    def test_evidence_her2_low_tdxd_eligible(self, api_client):
        resp = api_client.get("/evidence/HER2_Low")
        assert resp.status_code == 200
        eligible = [e["agent"] for e in resp.json()["eligible_agents"]]
        assert "T-DXd" in eligible

    def test_evidence_her2_null_nothing_eligible(self, api_client):
        resp = api_client.get("/evidence/HER2_Null")
        assert resp.status_code == 200
        assert resp.json()["eligible_agents"] == []

    def test_evidence_returns_guidelines(self, api_client):
        resp = api_client.get("/evidence/HER2_Ultralow")
        assert resp.status_code == 200
        assert len(resp.json()["guidelines"]) > 0

    def test_evidence_unknown_category_404(self, api_client):
        resp = api_client.get("/evidence/HER2_Unknown")
        assert resp.status_code == 404


class TestStatsEndpoint:
    """GET /stats"""

    def test_stats_no_neo4j_returns_zeros(self, api_client):
        from app import api as api_module
        with patch.object(api_module, "_get_driver", return_value=None):
            resp = api_client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["neo4j_connected"] is False
        assert data["total_nodes"] == 0

    def test_stats_response_shape(self, api_client):
        resp = api_client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_nodes" in data
        assert "total_relations" in data
        assert "node_counts" in data
        assert "neo4j_connected" in data


class TestQueryEndpoint:
    """POST /query — requires patched LLM."""

    def _mock_llm(self):
        mock = MagicMock()
        mock.invoke.return_value = MagicMock(content='["diagnostic"]')
        return mock

    def test_query_missing_body_422(self, api_client):
        resp = api_client.post("/query", json={})
        assert resp.status_code == 422

    def test_query_too_short_422(self, api_client):
        resp = api_client.post("/query", json={"query": "IHC"})
        assert resp.status_code == 422

    def test_query_with_mocked_llm(self, api_client):
        import app.api as api_module
        api_module._llm_singleton = self._mock_llm()
        try:
            resp = api_client.post(
                "/query",
                json={
                    "query": "What is the HER2 classification for IHC 2+ with ISH Group 3?",
                    "ihc_score": "2+",
                    "ish_group": "Group3",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "synthesis" in data
            assert "agents_invoked" in data
        finally:
            api_module._llm_singleton = None
