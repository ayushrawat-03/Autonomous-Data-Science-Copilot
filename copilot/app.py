"""
app.py

Streamlit frontend for the Autonomous Data Science Co-Pilot.

Flow: upload file -> profile it -> ask a question -> generate code ->
run in sandbox -> self-heal on failure -> display chart(s) + insight.

Control flow only. Visual design lives in theme.py; the agent
pipeline lives in data_layer.py / orchestrator.py / sandbox.py /
rag_repair.py / renderer.py.


Run with:
    export ANTHROPIC_API_KEY=sk-ant-...
    streamlit run app.py --server.fileWatcherType none
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.io as pio
import streamlit as st

import theme
from data_layer import load_and_profile
from orchestrator import suggest_followups
from rag_repair import RunResult, run_with_self_healing
from renderer import Chart, render_result
from suggestions import build_suggested_questions

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

CHART_OUTPUT_DIR = os.path.abspath("./output")
MAX_FOLLOWUP_ROUNDS = 3


@dataclass
class Turn:
    """The current question/answer exchange being displayed."""

    question: str
    run_result: RunResult
    charts: List[Chart]
    insight: str
    followups: List[str]


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

def _init_session_state() -> None:
    st.session_state.setdefault("current_turn", None)
    st.session_state.setdefault("question_input", "")
    # Staging slot for follow-up chip clicks: Streamlit disallows
    # writing to a widget's key after it's been instantiated, so
    # chips write here and _apply_pending_question() moves it into
    # question_input on the next rerun, before the widget exists.
    st.session_state.setdefault("pending_question", None)
    st.session_state.setdefault("pending_is_followup", False)
    st.session_state.setdefault("df", None)
    st.session_state.setdefault("profile", None)
    st.session_state.setdefault("filename", None)
    # Follow-up chips clicked so far for the current answer, capped
    # at MAX_FOLLOWUP_ROUNDS.
    st.session_state.setdefault("followup_rounds_used", 0)


def _apply_pending_question() -> None:
    """Must be called before st.text_input(key='question_input', ...) is instantiated."""
    pending = st.session_state.get("pending_question")
    if pending is not None:
        st.session_state["question_input"] = pending
        st.session_state["pending_question"] = None
        if st.session_state.get("pending_is_followup"):
            st.session_state["followup_rounds_used"] += 1
        st.session_state["pending_is_followup"] = False


def _provider_status() -> tuple[bool, str]:
    """Returns (is_configured, human_readable_label)."""
    provider = os.environ.get("COPILOT_PROVIDER", "anthropic")
    if provider == "anthropic":
        configured = bool(os.environ.get("ANTHROPIC_API_KEY"))
        model = os.environ.get("COPILOT_MODEL", "claude-sonnet-4-6")
        return configured, f"Anthropic · {model}"
    model = os.environ.get("COPILOT_MODEL", "unset")
    configured = all(
        os.environ.get(v) for v in ("COPILOT_API_BASE", "COPILOT_API_KEY", "COPILOT_MODEL")
    )
    return configured, f"OpenAI-compatible · {model}"


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

def _render_sidebar(provider_ok: bool, provider_label: str) -> None:
    with st.sidebar:
        st.markdown(
            "### 🔎 Data Science Co-Pilot\n"
            "<span style='color:#9ca0b4; font-size:.85rem;'>Autonomous analysis agent</span>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        st.markdown("**Model provider**")
        dot = "🟢" if provider_ok else "🔴"
        st.markdown(f"{dot} {provider_label}")
        if not provider_ok:
            st.caption("Missing credentials — see the README for setup.")

        st.markdown("---")
        st.markdown("**How it works**")
        st.markdown(
            "1. Upload a file\n"
            "2. Ask a question in plain English\n"
            "3. The agent writes & runs Pandas code\n"
            "4. Errors trigger self-correction\n"
            "5. You get a chart + a plain-English insight"
        )


# --------------------------------------------------------------------------
# Upload + profile section
# --------------------------------------------------------------------------

def _render_upload_section() -> Optional[pd.DataFrame]:
    with theme.card("card_upload"):
        theme.section_label("Step 1 · Upload your data")
        uploaded_file = st.file_uploader(
            "Upload a CSV, Excel, or JSON file",
            type=["csv", "xlsx", "xls", "json"],
            label_visibility="collapsed",
        )

    if uploaded_file is None:
        return None

    is_new_file = st.session_state.get("filename") != uploaded_file.name
    if st.session_state.get("df") is None or is_new_file:
        try:
            with st.spinner("Reading and profiling your file..."):
                df, profile = load_and_profile(uploaded_file, uploaded_file.name)
        except Exception as e:  # noqa: BLE001 - a bad file should show an error, not crash the app
            st.error(f"Couldn't read this file: {e}")
            return None
        st.session_state.df = df
        st.session_state.profile = profile
        st.session_state.filename = uploaded_file.name
        if is_new_file:
            st.session_state.current_turn = None
            st.session_state.followup_rounds_used = 0

    return st.session_state.df


def _render_profile_card(df: pd.DataFrame, profile: Dict[str, Any]) -> None:
    with theme.card("card_profile"):
        theme.section_label(f"📄 {st.session_state.filename}")

        avg_missing = sum(c["null_pct"] for c in profile["columns"].values()) / max(
            len(profile["columns"]), 1
        )
        theme.metric_row(
            [
                (f"{profile['n_rows']:,}", "Rows"),
                (str(profile["n_columns"]), "Columns"),
                (f"{avg_missing:.1f}%", "Avg. missing"),
            ]
        )

        with st.expander("Preview data & full schema profile"):
            tab1, tab2 = st.tabs(["Data preview", "Schema profile (JSON)"])
            with tab1:
                st.dataframe(df.head(20), use_container_width=True)
            with tab2:
                st.json(profile)


# --------------------------------------------------------------------------
# Question input section
# --------------------------------------------------------------------------

def _render_question_section(profile: Dict[str, Any]) -> Optional[str]:
    with theme.card("card_question"):
        theme.section_label("Step 2 · Ask a question")

        suggested = build_suggested_questions(profile)
        placeholder = next(iter(suggested.values()), "What's in this data?")

        _apply_pending_question()  # must run before text_input() below is instantiated
        st.text_input(
            "Ask a question about your data",
            key="question_input",
            placeholder=placeholder,
            label_visibility="collapsed",
        )

        if suggested:
            st.caption("Ideas for this file — type one in yourself, or ask your own:")
            theme.suggestion_list(suggested)

        col_a, _ = st.columns([1, 3])
        with col_a:
            analyze_clicked = st.button(
                "✨ Analyze",
                type="primary",
                use_container_width=True,
                disabled=not st.session_state.question_input,
                key="btn_analyze",
            )
    return st.session_state.question_input if analyze_clicked else None


# --------------------------------------------------------------------------
# Analysis execution
# --------------------------------------------------------------------------

def _run_analysis(question: str, df: pd.DataFrame, profile: Dict[str, Any]) -> Optional[Turn]:
    os.makedirs(CHART_OUTPUT_DIR, exist_ok=True)
    for stale_path in glob.glob(os.path.join(CHART_OUTPUT_DIR, "chart_*.png")) + glob.glob(
        os.path.join(CHART_OUTPUT_DIR, "chart_*.json")
    ):
        os.remove(stale_path)

    try:
        with st.spinner("Writing and running the analysis (self-correcting on errors)..."):
            run_result = run_with_self_healing(
                question=question,
                df=df,
                profile=profile,
                chart_output_dir=CHART_OUTPUT_DIR,
                max_retries=3,
            )
    except Exception as e:  # noqa: BLE001 - never let a provider/network error crash the whole app
        st.error(f"Something went wrong talking to the model provider. Details: {e}")
        return None

    charts, insight = render_result(run_result)

    followups: List[str] = []
    if run_result.success:
        try:
            with st.spinner("Thinking of good follow-up questions..."):
                followups = suggest_followups(question, insight, profile)
        except Exception:  # noqa: BLE001 - follow-ups are a nice-to-have, never fatal
            followups = []

    return Turn(
        question=question,
        run_result=run_result,
        charts=charts,
        insight=insight,
        followups=followups,
    )


# --------------------------------------------------------------------------
# Turn rendering
# --------------------------------------------------------------------------

def _render_charts(charts: List[Chart], turn_index: int) -> None:
    """A single chart gets the full width; two or more share a 2-column grid."""
    if not charts:
        return

    if len(charts) == 1:
        path, chart_type, n = charts[0]
        _render_single_chart(path, chart_type, f"{turn_index}_{n}")
        return

    for row_start in range(0, len(charts), 2):
        row = charts[row_start : row_start + 2]
        cols = st.columns(len(row))
        for col, (path, chart_type, n) in zip(cols, row):
            with col:
                st.caption(f"Chart {n}")
                _render_single_chart(path, chart_type, f"{turn_index}_{n}")


def _render_single_chart(path: str, chart_type: str, widget_key: str) -> None:
    if chart_type == "image":
        st.image(path, use_container_width=True)
    else:
        with open(path) as f:
            fig = pio.from_json(f.read())
        st.plotly_chart(fig, use_container_width=True, key=f"chart_{widget_key}")


def _render_followups(followups: List[str], turn_index: int) -> None:
    if not followups:
        return

    rounds_used = st.session_state.get("followup_rounds_used", 0)
    rounds_left = MAX_FOLLOWUP_ROUNDS - rounds_used

    if rounds_left <= 0:
        st.markdown(
            f'<div class="cp-followup-note">You\'ve used all {MAX_FOLLOWUP_ROUNDS} quick '
            "follow-ups for this thread — type your next question above to keep going.</div>",
            unsafe_allow_html=True,
        )
        return

    theme.section_label("Dig deeper", spaced=True)
    st.caption(f"{rounds_left} of {MAX_FOLLOWUP_ROUNDS} quick follow-ups left")
    cols = st.columns(len(followups))
    for i, (col, fq) in enumerate(zip(cols, followups)):
        with col:
            if st.button(fq, key=f"followup_{turn_index}_{i}", use_container_width=True):
                st.session_state.pending_question = fq
                st.session_state.pending_is_followup = True
                st.rerun()


def _render_turn(turn: Turn, index: int) -> None:
    result = turn.run_result

    with st.container(key=f"turn_{index}"):
        st.markdown(f'<div class="cp-q">🗨️ {turn.question}</div>', unsafe_allow_html=True)

        badge_html = ""
        for attempt in result.attempts:
            badge = theme.repair_path_badge(attempt.repair_path, result.success)
            badge_html += f'<span class="cp-pill cp-pill--{badge.kind}">{badge.label}</span>'
        st.markdown(f'<div class="cp-a-wrap">{badge_html}</div>', unsafe_allow_html=True)

        if result.success:
            _render_charts(turn.charts, index)
            st.markdown(f'<div class="cp-insight">{turn.insight}</div>', unsafe_allow_html=True)
            _render_followups(turn.followups, index)
        else:
            st.error("Couldn't complete the analysis after self-correction attempts.")
            st.markdown(f'<div class="cp-insight">{turn.insight}</div>', unsafe_allow_html=True)

        with st.expander("🔍 How the agent got here"):
            for attempt in result.attempts:
                label = {
                    "none": "✅ Ran successfully" if result.success else "❌ Gave up",
                    "schema": "🗂️ Schema-grounded repair (re-showed the real column list)",
                    "doc": "📖 Doc-grounded repair (retrieved official docs via RAG)",
                    "import": "🚫 Removed an import that isn't available in this sandbox",
                }.get(attempt.repair_path, attempt.repair_path)
                st.markdown(f"**Attempt {attempt.attempt_number + 1}:** {label}")
                if attempt.doc_used:
                    st.caption(
                        f"Retrieved doc: `{attempt.doc_used['function_name']}` "
                        f"— {attempt.doc_used['source_url']}"
                    )
                if attempt.traceback:
                    st.code(attempt.traceback, language="text")

            st.markdown("**Generated code (final version)**")
            st.code(result.final_code, language="python")
            st.download_button(
                "⬇️ Download code (.py)",
                data=result.final_code,
                file_name=f"copilot_analysis_{index + 1}.py",
                mime="text/x-python",
                key=f"dl_code_{index}",
            )
            for path, chart_type, n in turn.charts:
                if chart_type == "image" and os.path.exists(path):
                    with open(path, "rb") as f:
                        st.download_button(
                            f"⬇️ Download chart {n} (.png)",
                            data=f.read(),
                            file_name=f"copilot_chart_{index + 1}_{n}.png",
                            mime="image/png",
                            key=f"dl_chart_{index}_{n}",
                        )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Autonomous Data Science Co-Pilot",
        page_icon="🔎",
        layout="wide",
    )
    theme.inject()
    _init_session_state()

    provider_ok, provider_label = _provider_status()
    _render_sidebar(provider_ok, provider_label)

    theme.hero(
        "🔎 Autonomous Data Science Co-Pilot",
        "Ask a question and get the right answer instantly.",
        badges=["Simple", "Smart", "Fast"],
    )

    if not provider_ok:
        st.warning(
            f"**{provider_label}** is not fully configured. Set the required "
            "environment variables before analyzing — see the README.",
            icon="⚠️",
        )

    df = _render_upload_section()

    if df is None:
        theme.empty_state(
            "📁",
            "Upload a file to get started",
            "Try a monthly sales CSV, a customer export, or any spreadsheet you "
            "have questions about.",
        )
        return

    profile = st.session_state.profile
    _render_profile_card(df, profile)
    question = _render_question_section(profile)

    if question:
        turn = _run_analysis(question, df, profile)
        if turn is not None:
            st.session_state.current_turn = turn
            st.session_state.followup_rounds_used = 0  # fresh answer, fresh follow-up budget

    turn = st.session_state.current_turn
    if turn is None:
        theme.empty_state(
            "💬",
            "Ask your first question",
            "Pick a use case above or type your own question about the uploaded data.",
        )
        return

    theme.section_label("Answer", spaced=True)
    _render_turn(turn, 0)


if __name__ == "__main__":
    main()
