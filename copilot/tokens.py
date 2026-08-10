"""
tokens.py

Color/typography tokens used by the Streamlit UI (theme.py).

Deliberately has zero dependencies (no streamlit import) so it could in
principle be unit-tested or reused outside the app.
"""

INK = "#0f1420"
PAPER = "#f6f7fb"
ACCENT = "#6d5efc"
ACCENT_DARK = "#4f3ff0"
ACCENT_SOFT = "#efeaff"
MUTED = "#6b7280"

FONT_IMPORT = (
    "https://fonts.googleapis.com/css2?"
    "family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap"
)
