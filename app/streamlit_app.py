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
        # json_mode=False: narración libre, no JSON estricto
        return cfg.get_llm(json_mode=False)
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
    user = os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j"))
    password = os.getenv("NEO4J_PASSWORD", "password")
    database = os.getenv("NEO4J_DATABASE") or None
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
            "🌐 Graph Explorer",
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
                clinical_data_llm = {"ihc_score": ihc_score}
                if ish_group:
                    clinical_data_llm["ish_group"] = ish_group
                if ish_ratio is not None:
                    clinical_data_llm["ish_ratio"] = ish_ratio
                if signals is not None:
                    clinical_data_llm["signals_per_cell"] = signals

                with st.spinner("LLM narrating..."):
                    try:
                        from langchain_core.messages import HumanMessage, SystemMessage
                        from src.agents.diagnostic_agent import _DIAGNOSTIC_SYSTEM_PROMPT
                        context_text = (
                            f"Clinical data: {clinical_data_llm}\n"
                            f"Rule-based classification: {result}\n"
                            f"User query: Classify IHC {ihc_score}"
                        )
                        messages = [
                            SystemMessage(content=_DIAGNOSTIC_SYSTEM_PROMPT),
                            HumanMessage(content=context_text),
                        ]
                        llm_response = llm.invoke(messages)
                        narrative = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
                        st.markdown(narrative)
                    except Exception as exc:
                        st.error(f"LLM error: {exc}")
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
            "Neo4j is not connected. Set `NEO4J_URI`, `NEO4J_USERNAME`, and "
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

# ===========================================================================
# PANEL 7 — GRAPH EXPLORER
# ===========================================================================

elif panel == "🌐 Graph Explorer":
    import tempfile
    import streamlit.components.v1 as _components

    st.title("🌐 Knowledge Graph Explorer")
    st.markdown(
        "Visualiza el grafo localmente con **pyvis** o ábrelo directamente "
        "en el navegador de Neo4j."
    )

    # ── Neo4j Browser deep link ─────────────────────────────────────────────
    _uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    _is_aura = "aura" in _uri or "neo4j+s://" in _uri
    if _is_aura:
        _host = _uri.replace("neo4j+s://", "").replace("neo4j://", "").rstrip("/")
        _browser_url = f"https://browser.neo4j.io/?dbms=neo4j%2Bs%3A%2F%2F{_host}"
        _browser_label = "Abrir en Neo4j Browser (AuraDB) ↗"
    else:
        _browser_url = "http://localhost:7474/browser/"
        _browser_label = "Abrir en Neo4j Browser (local) ↗"

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        st.link_button(_browser_label, _browser_url, type="secondary")
    with col_info:
        st.caption(
            "Se abrirá el navegador de Neo4j. Usa las queries de abajo "
            "para explorar el grafo completo."
        )

    with st.expander("📋 Queries Cypher para Neo4j Browser", expanded=False):
        st.code(
            "-- Grafo completo (sin chunks)\n"
            "MATCH (n) WHERE NOT n:Chunk RETURN n LIMIT 300",
            language="cypher",
        )
        st.code(
            "-- Solo categorías clínicas y sus relaciones\n"
            "MATCH p=(c:ClinicalCategory)-[r]->(m) RETURN p",
            language="cypher",
        )
        st.code(
            "-- Árbol IHC completo\n"
            "MATCH p=(d:DiagnosticDecision)-[:leadsTo*]->(r)\n"
            "WHERE d.id = 'IHC_ENTRY' RETURN p",
            language="cypher",
        )
        st.code(
            "-- Elegibilidad T-DXd\n"
            "MATCH p=(c:ClinicalCategory)-[:eligibleFor]->(t:TherapeuticAgent) RETURN p",
            language="cypher",
        )

    st.divider()
    st.subheader("Visualización interactiva")

    if driver is None:
        st.warning(
            "Neo4j no conectado. Conecta Neo4j para renderizar el grafo interactivo."
        )
    else:
        # Controls
        _NODE_TYPES = [
            "Ontología clínica",
            "ClinicalCategory",
            "IHCScore",
            "ISHGroup",
            "TherapeuticAgent",
            "ClinicalTrial",
            "Guideline",
            "DiagnosticDecision",
            "Assay",
            "FractalMetric",
        ]
        _NODE_COLORS = {
            "ClinicalCategory": "#e74c3c",
            "IHCScore": "#3498db",
            "ISHGroup": "#9b59b6",
            "TherapeuticAgent": "#2ecc71",
            "ClinicalTrial": "#f39c12",
            "Guideline": "#1abc9c",
            "DiagnosticDecision": "#e67e22",
            "Assay": "#95a5a6",
            "FractalMetric": "#fd79a8",
            "ToySpecimen": "#a29bfe",
        }

        fc1, fc2, fc3 = st.columns([2, 1, 1])
        with fc1:
            _sel_type = st.selectbox("Tipo de nodo", _NODE_TYPES, index=0)
        with fc2:
            _max_nodes = st.slider("Máx. nodos", 20, 400, 200)
        with fc3:
            _physics = st.checkbox("Animación física", value=True)

        if st.button("🔄 Renderizar grafo", type="primary"):
            if _sel_type == "Ontología clínica":
                _cypher = (
                    "MATCH (n) WHERE NOT n:Chunk "
                    "WITH n LIMIT $lim "
                    "OPTIONAL MATCH (n)-[r]->(m) WHERE NOT m:Chunk "
                    "RETURN n, r, m"
                )
            else:
                _cypher = (
                    f"MATCH (n:{_sel_type}) WITH n LIMIT $lim "
                    "OPTIONAL MATCH (n)-[r]->(m) WHERE NOT m:Chunk "
                    "RETURN n, r, m"
                )

            with st.spinner("Consultando Neo4j y renderizando…"):
                try:
                    with driver.session() as _sess:
                        _records = list(_sess.run(_cypher, lim=_max_nodes))

                    from pyvis.network import Network

                    _net = Network(
                        height="650px", width="100%",
                        bgcolor="#0e1117", font_color="white",
                        notebook=False, directed=True,
                    )
                    _net.set_options("""
                    {
                      "nodes": {"borderWidth": 2, "shadow": true},
                      "edges": {
                        "arrows": {"to": {"enabled": true, "scaleFactor": 0.8}},
                        "shadow": false, "smooth": {"type": "curvedCW", "roundness": 0.2}
                      },
                      "physics": {"enabled": %s,
                        "barnesHut": {"gravitationalConstant": -8000, "springLength": 120}},
                      "interaction": {"hover": true, "navigationButtons": true,
                        "keyboard": {"enabled": true}}
                    }
                    """ % str(_physics).lower())

                    _added: set = set()

                    def _node_label(node) -> str:
                        for k in ("label", "name", "id", "chunk_id"):
                            if node.get(k):
                                v = str(node.get(k))
                                return v[:40] + "…" if len(v) > 40 else v
                        return str(node.element_id)[:8]

                    def _add_node(node) -> None:
                        eid = node.element_id
                        if eid in _added:
                            return
                        lbls = list(node.labels)
                        color = _NODE_COLORS.get(lbls[0] if lbls else "", "#74b9ff")
                        tip = "\n".join(
                            f"{k}: {v}" for k, v in dict(node).items()
                            if k not in ("embedding", "text") and v is not None
                        )
                        _net.add_node(
                            eid, label=_node_label(node),
                            title=tip or eid, color=color,
                            size=20 if "Category" in (lbls[0] if lbls else "") else 14,
                        )
                        _added.add(eid)

                    for _rec in _records:
                        _n = _rec["n"]
                        _m = _rec["m"]
                        _r = _rec["r"]
                        if _n is not None:
                            _add_node(_n)
                        if _m is not None:
                            _add_node(_m)
                        if _r is not None and _n is not None and _m is not None:
                            _net.add_edge(
                                _n.element_id, _m.element_id,
                                label=_r.type, title=_r.type,
                                color="#636e72",
                            )

                    with tempfile.NamedTemporaryFile(
                        suffix=".html", mode="w",
                        delete=False, encoding="utf-8"
                    ) as _tf:
                        _net.save_graph(_tf.name)
                        _tmp = _tf.name

                    with open(_tmp, "r", encoding="utf-8") as _f:
                        _html = _f.read()

                    _node_count = len(_added)
                    _edge_count = len(_net.edges)
                    st.caption(
                        f"Renderizados **{_node_count}** nodos y "
                        f"**{_edge_count}** relaciones. "
                        "Arrastra, zoon con scroll, doble clic para fijar nodos."
                    )
                    _components.html(_html, height=660, scrolling=False)

                except ImportError:
                    st.error(
                        "pyvis no instalado. Ejecuta: `pip install pyvis` y reinicia."
                    )
                except Exception as _exc:
                    st.error(f"Error al renderizar el grafo: {_exc}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "HER2 KG Platform · DigPatho · April 2026 · "
    "Built with LangChain, LangGraph, Neo4j, and Streamlit"
)
