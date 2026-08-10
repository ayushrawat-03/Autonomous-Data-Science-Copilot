"""
theme.py

Design system (color tokens, typography, component CSS) and reusable
rendering helpers for the Streamlit UI. Keeps app.py focused on
control flow rather than markup.

textwrap.dedent() is used on the CSS string because
st.markdown(..., unsafe_allow_html=True) still parses content as
Markdown first, and Markdown renders 4+ space indented lines as a
preformatted code block instead of HTML.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass

import streamlit as st

from tokens import ACCENT, ACCENT_DARK, ACCENT_SOFT, INK, MUTED, PAPER, FONT_IMPORT


def inject() -> None:
    """
    Inject global CSS once per session. Call at the top of app.py.

    Must contain no blank lines: Markdown's HTML-block rule for a
    leading <link> tag ends raw-HTML passthrough at the first blank
    line, so a blank line here would cause the rest to render as
    plain text instead of CSS.
    """
    raw = textwrap.dedent(
        f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="{FONT_IMPORT}" rel="stylesheet">
        <style>
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        code, pre, .stCodeBlock, .stCode {{
            font-family: 'JetBrains Mono', ui-monospace, monospace !important;
        }}
        /* Page chrome */
        #MainMenu, footer, header {{ visibility: hidden; }}
        .block-container {{
            padding-top: 1.6rem;
            padding-bottom: 3rem;
            max-width: 980px;
        }}
        [data-testid="stSidebar"] {{
            background: {INK};
            border-right: 1px solid rgba(255,255,255,0.06);
        }}
        [data-testid="stSidebar"] * {{ color: #e7e8ee; }}
        [data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.1); }}
        /* Hero */
        .cp-hero {{
            background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_DARK} 55%, #2c1f8f 100%);
            border-radius: 20px;
            padding: 2.1rem 2.4rem;
            margin-bottom: 1.4rem;
            box-shadow: 0 14px 32px -12px rgba(79,63,240,0.45);
            position: relative;
            overflow: hidden;
        }}
        .cp-hero::after {{
            content: "";
            position: absolute; inset: 0;
            background: radial-gradient(circle at 82% 15%, rgba(255,255,255,0.16), transparent 55%);
        }}
        .cp-hero h1 {{
            color: white; font-size: 1.9rem; font-weight: 800;
            margin: 0 0 .35rem 0; letter-spacing: -0.02em;
        }}
        .cp-hero p {{
            color: rgba(255,255,255,0.88); font-size: .98rem; margin: 0;
            max-width: 640px; line-height: 1.5;
        }}
        .cp-badge-row {{ margin-top: 1rem; display: flex; gap: .5rem; flex-wrap: wrap; position: relative; z-index: 1;}}
        .cp-badge {{
            background: rgba(255,255,255,0.16); color: white; font-size: .76rem;
            font-weight: 600; padding: .3rem .7rem; border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.25);
        }}
        /* Cards: styles Streamlit's own st.container(border=True),
           not a manually opened/closed div, so the border always
           wraps its actual children. */
        [data-testid="stAppViewContainer"] [data-testid="stVerticalBlockBorderWrapper"] {{
            background: white !important;
            border: 1px solid #eceef4 !important;
            border-radius: 16px !important;
            box-shadow: 0 1px 2px rgba(16,24,40,0.04);
        }}
        [data-testid="stAppViewContainer"] [data-testid="stVerticalBlockBorderWrapper"] > div {{
            padding: .15rem .15rem;
        }}
        .cp-card h4 {{ margin: 0 0 .5rem 0; font-size: 1.02rem; font-weight: 700; color: {INK}; }}
        .cp-section-label {{
            font-size: .74rem; font-weight: 700; letter-spacing: .08em;
            text-transform: uppercase; color: {MUTED}; margin-bottom: .4rem;
        }}
        .cp-section-label--spaced {{ margin-top: 1.6rem; }}
        /* Metric tiles */
        .cp-metric-row {{ display: flex; gap: .7rem; flex-wrap: wrap; }}
        .cp-metric {{
            flex: 1 1 130px; background: {PAPER}; border-radius: 12px;
            padding: .75rem .9rem; border: 1px solid #eceef4;
        }}
        .cp-metric .v {{ font-size: 1.35rem; font-weight: 800; color: {INK}; line-height: 1.2; }}
        .cp-metric .l {{ font-size: .74rem; color: {MUTED}; font-weight: 600; }}
        /* Buttons */
        div[data-testid="stButton"] > button {{
            border-radius: 10px !important;
            font-weight: 600 !important;
        }}
        div[data-testid="stButton"] > button[kind="secondary"] {{
            background: {ACCENT_SOFT} !important;
            border: 1px solid #ddd6ff !important;
            color: {ACCENT_DARK} !important;
        }}
        div[data-testid="stButton"] > button[kind="primary"] {{
            background: linear-gradient(135deg, {ACCENT}, {ACCENT_DARK}) !important;
            border: none !important;
            box-shadow: 0 8px 18px -8px rgba(79,63,240,0.55);
        }}
        /* Chat-style turn (each turn is a real st.container(key=f"turn_{{i}}"),
           targeted here via an attribute selector since the index varies) */
        [class*="st-key-turn_"] {{ margin: 1.1rem 0 1.4rem 0; }}
        .cp-q {{
            display: inline-flex; align-items: center; gap: .5rem;
            background: {INK}; color: white; font-weight: 600; font-size: .92rem;
            padding: .55rem .95rem; border-radius: 12px 12px 12px 2px;
            max-width: 85%;
        }}
        .cp-a-wrap {{ margin-top: .6rem; }}
        .cp-pill {{
            display: inline-block; font-size: .72rem; font-weight: 700;
            padding: .18rem .6rem; border-radius: 999px; margin-right: .35rem;
        }}
        .cp-pill--ok {{ background: #d7f7e8; color: #067a45; }}
        .cp-pill--fail {{ background: #fde2e1; color: #b3231f; }}
        .cp-pill--muted {{ background: #eef0f5; color: {MUTED}; }}
        .cp-insight {{
            background: {PAPER}; border-left: 3px solid {ACCENT};
            border-radius: 0 10px 10px 0; padding: .8rem 1rem;
            font-size: .95rem; line-height: 1.55; color: #1c2233;
        }}
        .cp-empty {{
            text-align: center; padding: 2.6rem 1rem; color: {MUTED};
        }}
        .cp-empty .big {{ font-size: 2.2rem; margin-bottom: .4rem; }}
        /* Suggestions: plain text you copy/type yourself, not a button */
        .cp-suggestions {{ display: flex; flex-direction: column; gap: .4rem; margin: .3rem 0 .9rem 0; }}
        .cp-suggestion {{
            display: flex; align-items: baseline; gap: .5rem;
            background: {PAPER}; border: 1px dashed #dcdfe9; border-radius: 10px;
            padding: .5rem .75rem; font-size: .88rem; color: #1c2233;
        }}
        .cp-suggestion .tag {{
            flex: 0 0 auto; font-size: .72rem; font-weight: 700; color: {ACCENT_DARK};
        }}
        .cp-suggestion .txt {{ color: {INK}; }}
        .cp-followup-note {{ font-size: .82rem; color: {MUTED}; margin-top: .5rem; }}
        </style>
        """
    )
    # Strip any blank line (see docstring above for why).
    css = "\n".join(line for line in raw.splitlines() if line.strip() != "")
    st.markdown(css, unsafe_allow_html=True)


def hero(title: str, subtitle: str, badges: list[str]) -> None:
    badge_html = "".join(f'<span class="cp-badge">{b}</span>' for b in badges)
    st.markdown(
        f'<div class="cp-hero"><h1>{title}</h1><p>{subtitle}</p>'
        f'<div class="cp-badge-row">{badge_html}</div></div>',
        unsafe_allow_html=True,
    )


def section_label(text: str, spaced: bool = False) -> None:
    cls = "cp-section-label cp-section-label--spaced" if spaced else "cp-section-label"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def card(key: str):
    """A bordered Streamlit container styled to look like a white card.

    Use as `with theme.card("some_key"): ...`.
    """
    return st.container(border=True, key=key)


def suggestion_list(items: dict[str, str]) -> None:
    """Plain, non-clickable suggested questions the user copies/types themselves."""
    rows = "".join(
        f'<div class="cp-suggestion"><span class="tag">{label}</span>'
        f'<span class="txt">{question}</span></div>'
        for label, question in items.items()
    )
    st.markdown(f'<div class="cp-suggestions">{rows}</div>', unsafe_allow_html=True)


def metric_row(items: list[tuple[str, str]]) -> None:
    """items: list of (value, label)"""
    tiles = "".join(
        f'<div class="cp-metric"><div class="v">{value}</div><div class="l">{label}</div></div>'
        for value, label in items
    )
    st.markdown(f'<div class="cp-metric-row">{tiles}</div>', unsafe_allow_html=True)


def empty_state(icon: str, title: str, body: str) -> None:
    st.markdown(
        f'<div class="cp-empty"><div class="big">{icon}</div>'
        f'<div style="font-weight:700; color:{INK}; font-size:1.05rem;">{title}</div>'
        f'<div style="max-width:420px; margin:.35rem auto 0 auto;">{body}</div></div>',
        unsafe_allow_html=True,
    )


@dataclass
class RepairBadge:
    label: str
    kind: str  # "ok" | "fail" | "muted"


_REPAIR_LABELS = {
    "schema": "Schema-grounded fix",
    "doc": "Doc-grounded fix (RAG)",
    "import": "Removed unavailable import",
}


def repair_path_badge(repair_path: str, success: bool) -> RepairBadge:
    if repair_path == "none":
        return RepairBadge("Succeeded" if success else "Gave up", "ok" if success else "fail")
    return RepairBadge(_REPAIR_LABELS.get(repair_path, repair_path), "muted")
