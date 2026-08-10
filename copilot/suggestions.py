"""
suggestions.py

Builds the "quick question" chips shown under the question box from
the uploaded file's own schema (profile['columns']) rather than
generic hardcoded examples, so every suggestion is answerable against
the actual uploaded data.
"""

from __future__ import annotations

from typing import Any, Dict


def _is_numeric(col_info: Dict[str, Any]) -> bool:
    return col_info["dtype"].startswith(("int", "float", "uint"))


def _is_likely_identifier(col_name: str) -> bool:
    """
    Numeric ID/key columns (employee_id, order_id, index, ...) are
    numeric but meaningless to sum/average, so they're excluded from
    aggregation suggestions. Matches "_id"/"id" as a whole word, not a
    bare endswith("id"), so columns like "amount_paid" aren't caught.
    """
    lowered = col_name.strip().lower()
    return lowered in ("id", "index", "key", "row", "rowid", "uuid", "guid") or lowered.endswith(
        "_id"
    )


def _is_datetime(col_info: Dict[str, Any]) -> bool:
    return "datetime" in col_info["dtype"]


def _is_low_cardinality_categorical(col_info: Dict[str, Any]) -> bool:
    # value_counts is only attached for low-cardinality columns
    # (see profile_dataframe), which is what's worth grouping by.
    return "value_counts" in col_info


def build_suggested_questions(profile: Dict[str, Any], max_suggestions: int = 5) -> Dict[str, str]:
    """
    Returns {chip_label: question_text}, built from column names that
    exist in this specific file. A file with only numeric columns
    still gets useful suggestions, just fewer of them.
    """
    columns = profile.get("columns", {})
    numeric_cols = [
        c for c, info in columns.items() if _is_numeric(info) and not _is_likely_identifier(c)
    ]
    datetime_cols = [c for c, info in columns.items() if _is_datetime(info)]
    categorical_cols = [c for c, info in columns.items() if _is_low_cardinality_categorical(info)]
    any_missing = any(info.get("null_pct", 0) > 0 for info in columns.values())

    suggestions: Dict[str, str] = {}

    if numeric_cols and categorical_cols:
        n, c = numeric_cols[0], categorical_cols[0]
        suggestions[f"📊 {n} by {c}"] = f"What is the total {n} by {c}?"

    if datetime_cols and numeric_cols:
        d, n = datetime_cols[0], numeric_cols[0]
        suggestions[f"📈 {n} over time"] = f"Is there a trend in {n} over time, using {d}?"

    second_categorical = next((c for c in categorical_cols[1:]), None)
    breakdown_col = second_categorical or (categorical_cols[0] if categorical_cols else None)
    if breakdown_col and f"📊 {numeric_cols[0] if numeric_cols else ''} by {breakdown_col}" not in suggestions:
        suggestions[f"🧭 Breakdown by {breakdown_col}"] = f"What's the breakdown of {breakdown_col}?"

    if any_missing:
        suggestions["🧹 Data quality"] = (
            "Which columns have missing values, duplicates, or outliers?"
        )

    if len(suggestions) < max_suggestions:
        suggestions["🔎 Full overview"] = (
            "How is this dataset doing overall? Show me the key patterns."
        )

    return dict(list(suggestions.items())[:max_suggestions])
