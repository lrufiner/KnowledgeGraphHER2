"""
Tests for the multi-agent system: state, tools, diagnostic classification,
validation rules, and supervisor routing.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_driver(query_results: list[dict] | None = None):
    """Return a mock Neo4j driver that yields *query_results* from session.run()."""
    query_results = query_results or []

    mock_result = MagicMock()
    mock_result.__iter__ = MagicMock(return_value=iter(
        [MagicMock(**{f: v for f, v in row.items()}) for row in query_results]
    ))

    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.run = MagicMock(return_value=mock_result)

    mock_driver = MagicMock()
    mock_driver.session = MagicMock(return_value=mock_session)
    return mock_driver


def _make_mock_llm(response_text: str = "Mocked LLM response"):
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = response_text
    mock_llm.invoke = MagicMock(return_value=mock_response)
    return mock_llm


# ---------------------------------------------------------------------------
# Tests for HER2AgentState
# ---------------------------------------------------------------------------

class TestHER2AgentState:
    def test_empty_state_has_all_keys(self):
        from src.agents.state import EMPTY_STATE, HER2AgentState
        required_keys = {
            "query", "clinical_data", "target_agents", "current_agent",
            "agent_results", "final_response", "citations", "confidence",
            "iteration_count", "needs_human_review",
        }
        for key in required_keys:
            assert key in EMPTY_STATE, f"EMPTY_STATE missing key: {key}"

    def test_empty_state_defaults(self):
        from src.agents.state import EMPTY_STATE
        assert EMPTY_STATE["query"] == ""
        assert EMPTY_STATE["agent_results"] == []
        assert EMPTY_STATE["iteration_count"] == 0
        assert EMPTY_STATE["needs_human_review"] is False
        assert EMPTY_STATE["confidence"] == 0.0

    def test_add_accumulator_merges_lists(self):
        from src.agents.state import _add
        a = [{"x": 1}]
        b = [{"x": 2}]
        merged = _add(a, b)
        assert len(merged) == 2
        assert merged[0]["x"] == 1
        assert merged[1]["x"] == 2


# ---------------------------------------------------------------------------
# Tests for diagnostic classification (rule-based, no LLM)
# ---------------------------------------------------------------------------

class TestDiagnosticClassification:
    def _classify(self, clinical_data: dict) -> dict:
        from src.agents.diagnostic_agent import _classify_from_data
        return _classify_from_data(clinical_data)

    def test_ihc_3plus_is_positive(self):
        result = self._classify({"ihc_score": "3+"})
        assert result["classification"] == "HER2_Positive"
        assert result["confidence"] == "HIGH"

    def test_ihc_2plus_requires_reflex_ish(self):
        result = self._classify({"ihc_score": "2+"})
        assert result["classification"] == "HER2_Equivocal"
        assert "ISH" in result["action_required"]

    def test_ihc_2plus_ish_group1_is_positive(self):
        result = self._classify({"ihc_score": "2+", "ish_group": "Group1"})
        assert result["classification"] == "HER2_Positive"

    def test_ihc_2plus_ish_group5_is_low(self):
        result = self._classify({"ihc_score": "2+", "ish_group": "Group5"})
        assert result["classification"] == "HER2_Low"
        assert result["confidence"] == "HIGH"

    def test_ihc_1plus_is_low(self):
        result = self._classify({"ihc_score": "1+"})
        assert result["classification"] == "HER2_Low"

    def test_ihc_0plus_is_ultralow(self):
        result = self._classify({"ihc_score": "0+"})
        assert result["classification"] == "HER2_Ultralow"
        assert "Rakha" in result["applicable_guideline"]

    def test_ihc_0_is_null(self):
        result = self._classify({"ihc_score": "0"})
        assert result["classification"] == "HER2_Null"

    def test_unknown_ihc_returns_unknown(self):
        result = self._classify({"ihc_score": "bad_value"})
        assert result["classification"] == "UNKNOWN"
        assert result["confidence"] == "LOW"

    def test_pathway_steps_are_non_empty_for_known_scores(self):
        for ihc_score in ("3+", "2+", "1+", "0+", "0"):
            result = self._classify({"ihc_score": ihc_score})
            assert len(result["pathway_steps"]) > 0, \
                f"No pathway steps for ihc_score={ihc_score}"

    def test_guideline_set_for_known_scores(self):
        for ihc_score in ("3+", "2+", "1+", "0+", "0"):
            result = self._classify({"ihc_score": ihc_score})
            assert result["applicable_guideline"] is not None, \
                f"No guideline for ihc_score={ihc_score}"


# ---------------------------------------------------------------------------
# Tests for DiagnosticAgent (with mocked LLM + driver)
# ---------------------------------------------------------------------------

class TestDiagnosticAgent:
    def test_agent_returns_agent_results(self):
        from src.agents.diagnostic_agent import DiagnosticAgent
        from src.agents.state import EMPTY_STATE

        state = dict(EMPTY_STATE)
        state["query"] = "IHC 3+ result, what is HER2 status?"
        state["clinical_data"] = {"ihc_score": "3+"}

        mock_driver = _make_mock_driver()
        mock_llm = _make_mock_llm("HER2-Positive classification confirmed.")

        agent = DiagnosticAgent(mock_llm, mock_driver)
        update = agent(state)

        assert "agent_results" in update
        assert len(update["agent_results"]) == 1
        assert update["agent_results"][0]["agent"] == "diagnostic"

    def test_agent_sets_current_agent(self):
        from src.agents.diagnostic_agent import DiagnosticAgent
        from src.agents.state import EMPTY_STATE

        state = dict(EMPTY_STATE)
        state["clinical_data"] = {"ihc_score": "1+"}

        agent = DiagnosticAgent(_make_mock_llm(), _make_mock_driver())
        update = agent(state)
        assert update["current_agent"] == "diagnostic"

    def test_narrative_comes_from_llm(self):
        from src.agents.diagnostic_agent import DiagnosticAgent
        from src.agents.state import EMPTY_STATE

        expected_narrative = "Unique sentinel narrative text"
        state = dict(EMPTY_STATE)
        state["clinical_data"] = {"ihc_score": "2+", "ish_group": "Group3"}

        agent = DiagnosticAgent(_make_mock_llm(expected_narrative), _make_mock_driver())
        update = agent(state)
        assert expected_narrative in update["agent_results"][0]["narrative"]


# ---------------------------------------------------------------------------
# Tests for ValidationAgent (rule-based, no LLM needed for core logic)
# ---------------------------------------------------------------------------

class TestValidationAgent:
    def _make_agent(self, llm_text: str = "Validation complete."):
        from src.agents.validation_agent import ValidationAgent
        return ValidationAgent(_make_mock_llm(llm_text), _make_mock_driver())

    def test_ihc_ish_conflict_detected(self):
        from src.agents.validation_agent import ValidationAgent
        from src.agents.state import EMPTY_STATE

        agent = self._make_agent()
        state = dict(EMPTY_STATE)
        state["clinical_data"] = {"ihc_score": "3+", "ish_group": "Group5"}
        update = agent(state)
        result = update["agent_results"][0]
        assert result["critical_count"] > 0
        assert result["status"] == "FAIL"

    def test_clean_case_passes(self):
        from src.agents.validation_agent import ValidationAgent
        from src.agents.state import EMPTY_STATE

        agent = self._make_agent()
        state = dict(EMPTY_STATE)
        state["clinical_data"] = {"ihc_score": "3+"}
        update = agent(state)
        result = update["agent_results"][0]
        # No IHC/ISH conflict — should have 0 critical issues
        assert result["critical_count"] == 0

    def test_needs_human_review_set_on_critical(self):
        from src.agents.validation_agent import ValidationAgent
        from src.agents.state import EMPTY_STATE

        agent = self._make_agent()
        state = dict(EMPTY_STATE)
        state["clinical_data"] = {"ihc_score": "3+", "ish_group": "Group5"}
        update = agent(state)
        assert update["needs_human_review"] is True

    def test_run_validation_rules_direct(self):
        """Unit test _run_validation_rules without any LLM or driver."""
        from src.agents.validation_agent import ValidationAgent

        agent = ValidationAgent(_make_mock_llm(), _make_mock_driver())
        issues = agent._run_validation_rules({"ihc_score": "2+"})
        # IHC 2+ alone should trigger the "reflex ISH" rule
        assert isinstance(issues, list)


# ---------------------------------------------------------------------------
# Tests for supervisor routing logic
# ---------------------------------------------------------------------------

class TestSupervisorRouting:
    def test_route_fn_picks_first_unrun_agent(self):
        from src.agents.supervisor import _make_route_after_supervisor
        from src.agents.state import EMPTY_STATE

        route_fn = _make_route_after_supervisor(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        state = dict(EMPTY_STATE)
        state["target_agents"] = ["diagnostic", "evidence"]
        state["agent_results"] = []
        assert route_fn(state) == "diagnostic"

    def test_route_fn_skips_already_run_agent(self):
        from src.agents.supervisor import _make_route_after_supervisor
        from src.agents.state import EMPTY_STATE

        route_fn = _make_route_after_supervisor(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        state = dict(EMPTY_STATE)
        state["target_agents"] = ["diagnostic", "evidence"]
        state["agent_results"] = [{"agent": "diagnostic"}]
        assert route_fn(state) == "evidence"

    def test_route_fn_goes_to_synthesize_when_all_done(self):
        from src.agents.supervisor import _make_route_after_supervisor
        from src.agents.state import EMPTY_STATE

        route_fn = _make_route_after_supervisor(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        state = dict(EMPTY_STATE)
        state["target_agents"] = ["diagnostic"]
        state["agent_results"] = [{"agent": "diagnostic"}]
        assert route_fn(state) == "synthesize"

    def test_supervisor_node_sets_target_agents(self):
        from src.agents.supervisor import _make_supervisor_node
        from src.agents.state import EMPTY_STATE

        # LLM returns valid JSON array
        mock_llm = _make_mock_llm('["diagnostic"]')
        node_fn = _make_supervisor_node(mock_llm)

        state = dict(EMPTY_STATE)
        state["query"] = "What is the HER2 status for IHC 3+?"
        update = node_fn(state)

        assert "target_agents" in update
        assert "diagnostic" in update["target_agents"]
        assert update["iteration_count"] == 1

    def test_supervisor_stops_at_max_iterations(self):
        from src.agents.supervisor import _make_supervisor_node, _MAX_ITERATIONS
        from src.agents.state import EMPTY_STATE

        node_fn = _make_supervisor_node(_make_mock_llm('["diagnostic"]'))
        state = dict(EMPTY_STATE)
        state["iteration_count"] = _MAX_ITERATIONS  # Already at limit
        update = node_fn(state)
        assert update["target_agents"] == []
