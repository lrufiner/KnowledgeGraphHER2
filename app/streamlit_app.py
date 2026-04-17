"""
HER2 Knowledge Graph — Streamlit Dashboard

Implements §8.2 of the implementation plan.

Panels:
    1. Case Simulator   — IHC/ISH input → deterministic classification + LLM narration
    2. Pathway Viewer   — Visual step-by-step algorithm path
    3. Evidence Lookup  — Therapeutic eligibility + guideline evidence
    4. Validation Check — Consistency validation for a case
    5. Query Interface  — Natural language → multi-agent response
    6. Graph Stats      — KG health metrics (requires Neo4j)

Run:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — allows running from project root or app/ directory
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HER2 KG Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Lazy imports from project
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_llm(_mode: str = "", _model: str = ""):
    """Load LLM via PipelineConfig (respects HER2_KG_LLM_MODE env var)."""
    try:
        from src.pipeline.config import PipelineConfig
        cfg = PipelineConfig.from_env()
        if _mode:
            cfg = cfg.model_copy(update={"llm_mode": _mode})
        if _model:
            key = {"ollama": "ollama_model", "openai": "openai_model", "claude": "claude_model"}.get(cfg.llm_mode, "ollama_model")
            cfg = cfg.model_copy(update={key: _model})
        return cfg.get_llm()
    except Exception as exc:
        st.warning(f"LLM not available: {exc}. Deterministic mode only.")
        return None


def _llm_provider_label() -> tuple[str, str]:
    """Return (provider, model) strings for display in the sidebar."""
    try:
        from src.pipeline.config import PipelineConfig
        cfg = PipelineConfig.from_env()
        mode = cfg.llm_mode
        if mode == "openai":
            return "OpenAI", cfg.openai_model
        elif mode == "claude":
            return "Anthropic", cfg.claude_model
        else:
            return "Ollama", cfg.ollama_model
    except Exception:
        return "Ollama", "qwen3:8b"


@st.cache_resource(show_spinner=False, ttl=60)
def _load_driver():
    """Load Neo4j driver (cached 60 s; retries if previously unavailable)."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        return driver
    except Exception:
        return None


def _get_diagnostic_agent(llm, driver):
    from src.agents.diagnostic_agent import DiagnosticAgent
    return DiagnosticAgent(llm=llm, driver=driver)


def _get_validation_agent(llm, driver):
    from src.agents.validation_agent import ValidationAgent
    return ValidationAgent(llm=llm, driver=driver)


def _get_evidence_agent(llm, driver):
    from src.agents.evidence_agent import EvidenceAgent
    return EvidenceAgent(llm=llm, driver=driver)


def _get_explanation_agent(llm, driver):
    from src.agents.explanation_agent import ExplanationAgent
    return ExplanationAgent(llm=llm, driver=driver)


def _build_supervisor(llm, driver):
    from src.agents.supervisor import build_agent_graph
    return build_agent_graph(llm=llm, driver=driver)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _classify_only(ihc_score: str, ish_group: str, ish_ratio: float | None,
                   signals: float | None) -> dict[str, Any]:
    """Run deterministic classification without LLM."""
    from src.agents.diagnostic_agent import _classify_from_data
    data = {"ihc_score": ihc_score}
    if ish_group:
        data["ish_group"] = ish_group
    if ish_ratio is not None:
        data["ish_ratio"] = ish_ratio
    if signals is not None:
        data["signals_per_cell"] = signals
    return _classify_from_data(data)


def _category_color(category: str) -> str:
    colors = {
        "HER2_Positive": "#16a34a",
        "HER2_Equivocal": "#d97706",
        "HER2_Low": "#2563eb",
        "HER2_Ultralow": "#7c3aed",
        "HER2_Null": "#6b7280",
    }
    return colors.get(category, "#374151")


def _confidence_badge(conf: str) -> str:
    badges = {
        "HIGH": "🟢 HIGH",
        "MEDIUM": "🟡 MEDIUM",
        "LOW": "🔴 LOW",
    }
    return badges.get(conf, conf)


def _render_pathway(pathway_str: str) -> None:
    """Render the decision pathway as a visual step sequence."""
    if not pathway_str:
        st.info("No pathway data available.")
        return
    steps = [s.strip() for s in pathway_str.split("→")]
    cols_per_row = 4
    # Split into rows
    for row_start in range(0, len(steps), cols_per_row):
        row_steps = steps[row_start : row_start + cols_per_row]
        cols = st.columns(len(row_steps))
        for col, step in zip(cols, row_steps):
            is_final = row_start + len(row_steps) == len(steps)
            bg = "#16a34a" if (is_final and step.startswith("Score")) else "#1e40af"
            col.markdown(
                f"""<div style="background:{bg};color:white;padding:8px 4px;
                border-radius:6px;text-align:center;font-size:0.75rem;
                font-weight:600;">{step}</div>""",
                unsafe_allow_html=True,
            )
        if row_start + cols_per_row < len(steps):
            st.markdown(
                "<div style='text-align:center;font-size:1.5rem;margin:-4px 0;'>↓</div>",
                unsafe_allow_html=True,
            )


# ===========================================================================
# SIDEBAR
# ===========================================================================

with st.sidebar:
    st.markdown("### 🧬 HER2 KG Platform")
    st.markdown("**HER2 Knowledge Graph**")
    st.caption("DigPatho · April 2026")
    st.divider()

    panel = st.radio(
        "Navigation",
        options=[
            "🔬 Case Simulator",
            "🗺️ Pathway Viewer",
            "💊 Evidence Lookup",
            "✅ Validation Check",
            "💬 Query Interface",
            "📊 Graph Stats",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    _provider, _model_name = _llm_provider_label()
    st.markdown(f"**LLM Provider:** `{_provider}`")
    st.markdown(f"**Model:** `{_model_name}`")

    neo4j_connected = _load_driver() is not None
    if neo4j_connected:
        st.success("Neo4j ✓ Connected")
    else:
        st.warning("Neo4j ✗ Not connected\n(seed data mode)")

# Load shared resources
llm = _load_llm()
driver = _load_driver()

# ===========================================================================
# PANEL 1 — CASE SIMULATOR
# ===========================================================================

if panel == "🔬 Case Simulator":
    st.title("🔬 HER2 Case Simulator")
    st.markdown(
        "Enter IHC and ISH values to receive a deterministic HER2 classification "
        "and optional LLM-narrated interpretation."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Case Input")
        ihc_score = st.selectbox(
            "IHC Score",
            options=["3+", "2+", "1+", "0+", "0"],
            index=1,
        )
        use_ish = st.checkbox("Include ISH result", value=(ihc_score == "2+"))
        ish_group = ""
        ish_ratio = None
        signals = None
        if use_ish:
            ish_group = st.selectbox(
                "ISH Group (ASCO/CAP 2023)",
                options=["Group1", "Group2", "Group3", "Group4", "Group5"],
                index=2,
            )
            ish_ratio = st.number_input("HER2/CEP17 ratio", min_value=0.1, max_value=10.0, value=1.7, step=0.1)
            signals = st.number_input("HER2 signals/cell", min_value=0.1, max_value=30.0, value=5.4, step=0.1)

        use_llm = st.checkbox("Narrate with LLM", value=(llm is not None), disabled=(llm is None))
        run_btn = st.button("▶ Classify", type="primary", use_container_width=True)

    with col2:
        st.subheader("Classification Result")
        if run_btn:
            with st.spinner("Classifying..."):
                result = _classify_only(ihc_score, ish_group, ish_ratio, signals)

            category = result.get("classification", "Unknown")
            confidence = result.get("confidence", "UNKNOWN")
            guideline = result.get("applicable_guideline", result.get("guideline", ""))
            action = result.get("action_required", result.get("action", ""))
            pathway = " → ".join(result.get("pathway_steps", [])) or result.get("pathway", "")

            color = _category_color(category)
            st.markdown(
                f"""<div style="background:{color};color:white;padding:16px;
                border-radius:10px;margin-bottom:12px;">
                <h2 style="margin:0;color:white;">{category.replace('_', '-')}</h2>
                <p style="margin:4px 0 0;opacity:0.9;">{_confidence_badge(confidence)} · {guideline}</p>
                </div>""",
                unsafe_allow_html=True,
            )

            if action:
                st.info(f"**Recommended Action:** {action}")

            if pathway:
                st.subheader("Decision Pathway")
                _render_pathway(pathway)

            # LLM narration
            if use_llm and llm is not None:
                st.subheader("Clinical Narration (LLM)")
                clinical_data = {"ihc_score": ihc_score}
                if ish_group:
                    clinical_data["ish_group"] = ish_group
                if ish_ratio is not None:
                    clinical_data["ish_ratio"] = ish_ratio
                if signals is not None:
                    clinical_data["signals_per_cell"] = signals

                from src.agents.state import EMPTY_STATE
                state = {**EMPTY_STATE, "clinical_data": clinical_data, "query": f"Classify IHC {ihc_score}"}
                agent = _get_diagnostic_agent(llm, driver)
                with st.spinner("LLM narrating..."):
                    new_state = agent(state)
                results = new_state.get("agent_results", [])
                narrative = ""
                for r in results:
                    if r.get("agent") == "diagnostic":
                        narrative = r.get("narrative", "")
                        break
                if narrative:
                    st.markdown(narrative)
        else:
            st.info("Fill in the case data and click **▶ Classify**.")

# ===========================================================================
# PANEL 2 — PATHWAY VIEWER
# ===========================================================================

elif panel == "🗺️ Pathway Viewer":
    st.title("🗺️ Diagnostic Pathway Viewer")
    st.markdown(
        "Visualize the complete IHC → ISH → Clinical Category decision pathway "
        "as defined in the ASCO/CAP 2023 algorithm."
    )

    from src.domain.algorithm_definitions import IHC_ALGORITHM

    col1, col2 = st.columns([1, 3])
    with col1:
        viewer_ihc = st.selectbox("IHC Score", options=["3+", "2+", "1+", "0+", "0"], index=1)
        viewer_ish = st.selectbox("ISH Group (optional)", options=["—", "Group1", "Group2", "Group3", "Group4", "Group5"])
        viewer_ish = "" if viewer_ish == "—" else viewer_ish

    with col2:
        result = _classify_only(viewer_ihc, viewer_ish, None, None)
        pathway = " → ".join(result.get("pathway_steps", [])) or result.get("pathway", "")
        category = result.get("classification", "Unknown")
        color = _category_color(category)

        st.markdown(f"**Classification:** <span style='color:{color};font-weight:700;'>{category.replace('_', '-')}</span>", unsafe_allow_html=True)

        if pathway:
            st.subheader("Step-by-Step Path")
            _render_pathway(pathway)

    st.divider()
    st.subheader("Full Algorithm Structure")

    # Display algorithm nodes as expandable table
    nodes = IHC_ALGORITHM.get("nodes", [])
    edges = IHC_ALGORITHM.get("edges", [])

    tab1, tab2 = st.tabs(["Decision Nodes", "Edges"])
    with tab1:
        import pandas as pd
        df_nodes = pd.DataFrame([
            {
                "ID": n.get("id", ""),
                "Type": n.get("type", ""),
                "Label": n.get("label", ""),
                "Question / Value": n.get("question", n.get("result", "")),
            }
            for n in nodes
        ])
        st.dataframe(df_nodes, use_container_width=True, height=350)
    with tab2:
        df_edges = pd.DataFrame([
            {
                "From": e.get("from", ""),
                "To": e.get("to", ""),
                "Label": e.get("label", ""),
                "Condition": e.get("condition", ""),
            }
            for e in edges
        ])
        st.dataframe(df_edges, use_container_width=True, height=350)

# ===========================================================================
# PANEL 3 — EVIDENCE LOOKUP
# ===========================================================================

elif panel == "💊 Evidence Lookup":
    st.title("💊 Evidence & Therapeutic Eligibility")
    st.markdown(
        "Retrieve guideline-sourced evidence and therapeutic eligibility "
        "for a given HER2 category or clinical question."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        ev_category = st.selectbox(
            "HER2 Category",
            options=["HER2_Positive", "HER2_Equivocal", "HER2_Low", "HER2_Ultralow", "HER2_Null"],
            index=2,
        )
        ev_query = st.text_input(
            "Additional question (optional)",
            placeholder="e.g. Is T-DXd approved for this category?",
        )
        run_ev = st.button("🔍 Search Evidence", type="primary")

    with col2:
        if run_ev:
            if llm is None:
                st.error("LLM not available. Start Ollama and reload.")
            else:
                query_text = ev_query or f"What are the treatment options for {ev_category.replace('_', '-')}?"
                from src.agents.state import EMPTY_STATE
                state = {
                    **EMPTY_STATE,
                    "clinical_data": {"ihc_score": "2+"},  # neutral case
                    "query": query_text,
                    "current_agent": "evidence",
                }
                agent = _get_evidence_agent(llm, driver)
                with st.spinner("Retrieving evidence..."):
                    new_state = agent(state)
                results = new_state.get("agent_results", [])
                for r in results:
                    if r.get("agent") == "evidence":
                        st.markdown("### Evidence Summary")
                        st.markdown(r.get("summary", r.get("evidence", r.get("narrative", "No evidence found."))))
                        eligibility = r.get("eligibility", [])
                        if eligibility:
                            st.markdown("### Therapeutic Eligibility")
                            for item in eligibility[:10]:
                                badge = "✅" if item.get("eligible") else "❌"
                                st.markdown(f"{badge} **{item.get('agent', '')}** — {item.get('context', '')}")
                        break
        else:
            st.info("Select a category and click **🔍 Search Evidence**.")

            # Static eligibility reference table
            st.subheader("Quick Reference — Eligibility by Category")
            import pandas as pd
            ref_data = {
                "Category": ["HER2-Positive", "HER2-Equivocal", "HER2-Low", "HER2-Ultralow", "HER2-Null"],
                "T-DXd": ["✅ (1L+)", "when ISH+", "✅ DESTINY-Breast04", "⚠️ DESTINY-Breast06", "❌"],
                "Trastuzumab": ["✅", "when ISH+", "❌", "❌", "❌"],
                "Pertuzumab": ["✅ (early + mBC)", "when ISH+", "❌", "❌", "❌"],
                "Tucatinib": ["✅", "when ISH+", "⚠️ investigational", "❌", "❌"],
                "Guideline": ["ASCO/CAP 2023", "ASCO/CAP 2023", "DESTINY-B04", "Rakha 2026", "—"],
            }
            st.dataframe(pd.DataFrame(ref_data), use_container_width=True)

# ===========================================================================
# PANEL 4 — VALIDATION CHECK
# ===========================================================================

elif panel == "✅ Validation Check":
    st.title("✅ Clinical Consistency Validation")
    st.markdown(
        "Check a case for IHC/ISH conflicts, missing required tests, "
        "and clinical rule violations."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        val_ihc = st.selectbox("IHC Score", options=["3+", "2+", "1+", "0+", "0"], index=0)
        val_ish = st.selectbox("ISH Group", options=["—", "Group1", "Group2", "Group3", "Group4", "Group5"])
        val_ish = "" if val_ish == "—" else val_ish
        val_fractal = st.number_input("Fractal D0 (optional)", min_value=0.0, max_value=3.0, value=0.0, step=0.01)
        run_val = st.button("✅ Validate Case", type="primary")

    with col2:
        if run_val:
            clinical_data: dict[str, Any] = {"ihc_score": val_ihc}
            if val_ish:
                clinical_data["ish_group"] = val_ish
            if val_fractal > 0:
                clinical_data["fractal_d0"] = val_fractal

            if llm is None:
                # Run deterministic validation only
                from src.agents.validation_agent import ValidationAgent
                dummy_agent = ValidationAgent(llm=None, driver=None)
                issues = dummy_agent._run_validation_rules(clinical_data)
                conflict = dummy_agent._check_ish_conflict(
                    val_ihc, val_ish, clinical_data
                )
                issues.extend(conflict)
                status = "FAIL" if any(i.get("severity") in ("CRITICAL", "HIGH") for i in issues) else "PASS"
            else:
                from src.agents.state import EMPTY_STATE
                state = {**EMPTY_STATE, "clinical_data": clinical_data, "query": "Validate this HER2 case"}
                agent = _get_validation_agent(llm, driver)
                with st.spinner("Validating..."):
                    new_state = agent(state)
                issues = []
                status = "PASS"
                for r in new_state.get("agent_results", []):
                    if r.get("agent") == "validation":
                        issues = r.get("issues", [])
                        status = r.get("status", "PASS")
                        break

            if status == "PASS":
                st.success("✅ **PASS** — No critical or high-severity issues found.")
            else:
                st.error("❌ **FAIL** — Consistency issues detected.")

            if issues:
                st.subheader("Issues")
                for issue in issues:
                    sev = issue.get("severity", "INFO")
                    icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(sev, "ℹ️")
                    with st.expander(f"{icon} [{sev}] {issue.get('rule_id', '')}"):
                        st.write(issue.get("message", ""))
            else:
                st.info("No issues recorded.")
        else:
            st.info("Fill in case values and click **✅ Validate Case**.")

# ===========================================================================
# PANEL 5 — QUERY INTERFACE
# ===========================================================================

elif panel == "💬 Query Interface":
    st.title("💬 Multi-Agent Query Interface")
    st.markdown(
        "Ask any clinical question about HER2. The supervisor agent will route "
        "your question to the appropriate specialist agents and synthesize a response."
    )

    if llm is None:
        st.error("LLM not available. Check your provider configuration and reload the page.")
    else:
        with st.form("query_form"):
            query_text = st.text_area(
                "Clinical Question",
                placeholder="e.g. IHC 2+, ISH ratio 1.7, Group 3 — classify, validate and give evidence for T-DXd",
                height=100,
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                q_ihc = st.selectbox("IHC Score (optional context)", ["—", "3+", "2+", "1+", "0+", "0"])
            with col2:
                q_ish = st.selectbox("ISH Group (optional context)", ["—", "Group1", "Group2", "Group3", "Group4", "Group5"])
            with col3:
                q_ratio = st.number_input("ISH ratio (optional)", min_value=0.0, value=0.0, step=0.1)
            submitted = st.form_submit_button("🚀 Ask", type="primary")

        if submitted and query_text.strip():
            clinical_data: dict[str, Any] = {}
            if q_ihc != "—":
                clinical_data["ihc_score"] = q_ihc
            if q_ish != "—":
                clinical_data["ish_group"] = q_ish
            if q_ratio > 0:
                clinical_data["ish_ratio"] = q_ratio

            graph = _build_supervisor(llm, driver)
            from src.agents.state import EMPTY_STATE
            init_state = {
                **EMPTY_STATE,
                "query": query_text,
                "clinical_data": clinical_data,
            }

            # ── Streaming execution with live agent-progress feedback ─────────
            _AGENT_ICONS = {
                "diagnostic": "🔬", "evidence": "📚",
                "explanation": "💡", "validation": "✅",
            }
            _progress = st.empty()
            _answer = st.empty()
            final: dict = {}
            _prev_agents: list[str] = []

            for _chunk in graph.stream(init_state, stream_mode="values"):
                final = _chunk
                # agents_run tracked via agent_results list
                _running = [r.get("agent", "") for r in _chunk.get("agent_results", [])]
                if _running != _prev_agents:
                    _prev_agents = _running
                    _steps = " → ".join(
                        f"{_AGENT_ICONS.get(a, '🤖')} **{a}**" for a in _running
                    ) or "⏳ Starting…"
                    _progress.info(f"Pipeline: {_steps}")
                if _partial := _chunk.get("final_response", ""):
                    _answer.markdown(_partial)

            _progress.empty()  # clear progress bar when done

            agents_used = [r.get("agent", "?") for r in final.get("agent_results", [])]
            confidence = final.get("confidence", 0.0)
            needs_review = final.get("needs_human_review", False)

            # Metadata bar
            m1, m2, m3 = st.columns(3)
            m1.metric("Agents invoked", len(agents_used))
            m2.metric("Confidence", f"{confidence:.0%}")
            m3.metric("Human review", "⚠️ YES" if needs_review else "✅ NO")

            # Final answer
            final_response = final.get("final_response", "")
            if final_response:
                st.divider()
                _answer.markdown(final_response)
            else:
                _answer.warning("No synthesis generated.")

            # Expand agent details
            with st.expander("Agent Results Detail"):
                for r in final.get("agent_results", []):
                    st.markdown(f"**{r.get('agent', '?').upper()} Agent**")
                    st.json({k: v for k, v in r.items() if k != "agent"}, expanded=False)

# ===========================================================================
# PANEL 6 — GRAPH STATS
# ===========================================================================

elif panel == "📊 Graph Stats":
    st.title("📊 Knowledge Graph Statistics")

    if driver is None:
        st.warning(
            "Neo4j is not connected. Set `NEO4J_URI`, `NEO4J_USER`, and "
            "`NEO4J_PASSWORD` environment variables and restart Streamlit."
        )
        st.info(
            "**Quick start:**\n"
            "```bash\n"
            "docker run -p7474:7474 -p7687:7687 \\\n"
            "  -e NEO4J_AUTH=neo4j/password neo4j:5\n"
            "```"
        )
        st.subheader("Ontology Summary (static)")
        from src.domain.ontology import CLASS_HIERARCHY
        for cls, children in list(CLASS_HIERARCHY.items())[:10]:
            st.markdown(f"- **{cls}** → {', '.join(children[:5])}")
    else:
        from src.graph.neo4j_builder import get_graph_stats
        with st.spinner("Fetching KG statistics..."):
            stats = get_graph_stats(driver)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Nodes", stats.get("total_nodes", 0))
        col2.metric("Total Relations", stats.get("total_relations", 0))
        col3.metric("Node Types", len(stats.get("node_counts", {})))

        st.subheader("Node Type Distribution")
        node_counts = stats.get("node_counts", {})
        if node_counts:
            import pandas as pd
            df = pd.DataFrame(
                [(k, v) for k, v in sorted(node_counts.items(), key=lambda x: -x[1]) if v > 0],
                columns=["Node Type", "Count"],
            )
            st.bar_chart(df.set_index("Node Type"))
        else:
            st.info("No nodes found. Run the pipeline first.")

        st.divider()
        st.subheader("Raw Stats JSON")
        st.json(stats, expanded=False)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "HER2 KG Platform · DigPatho · April 2026 · "
    "Built with LangChain, LangGraph, Neo4j, and Streamlit"
)
