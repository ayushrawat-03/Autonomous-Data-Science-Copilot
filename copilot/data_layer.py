"""
data_layer.py

Parses uploaded files (CSV / XLSX / JSON) into a pandas DataFrame and
produces a *profile* of that DataFrame -- dtypes, null counts, ranges,
and a small sample of representative values per column.

The profile, not the raw rows, is what gets sent to the LLM in
orchestrator.py: smaller prompts, and no row-level data leaves the
app.
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


MAX_SAMPLE_VALUES = 5
MAX_UNIQUE_FOR_CATEGORICAL_SUMMARY = 20


def load_dataframe(file_obj, filename: str) -> pd.DataFrame:
    """
    Load a CSV, XLSX, or JSON file-like object into a DataFrame.

    `file_obj` can be a Streamlit UploadedFile, an open file handle, or
    a path string. `filename` is used to infer the format.
    """
    name = filename.lower()

    if name.endswith(".csv"):
        df = pd.read_csv(file_obj)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(file_obj, engine="openpyxl")
    elif name.endswith(".json"):
        # pd.json_normalize flattens nested objects into dot-separated
        # columns (e.g. "details.age"); pd.read_json would leave them
        # as unhashable dict/list values and break profiling.
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        raw = json.load(file_obj)
        if isinstance(raw, dict):
            # Either a single record, or a wrapper dict with a
            # records-holding key (e.g. {"records": [...]}).
            list_valued_keys = [k for k, v in raw.items() if isinstance(v, list)]
            if list_valued_keys:
                raw = raw[list_valued_keys[0]]
            else:
                raw = [raw]
        df = pd.json_normalize(raw)
    else:
        raise ValueError(
            f"Unsupported file type for '{filename}'. "
            "Supported: .csv, .xlsx, .xls, .json"
        )

    df = _clean_headers(df)
    return df


def _clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names and drop fully-unnamed index columns."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Common artifact: an unnamed index column saved from Excel/CSV exports
    unnamed_index_cols = [
        c for c in df.columns if c.startswith("Unnamed:") and df[c].isna().all()
    ]
    if unnamed_index_cols:
        df = df.drop(columns=unnamed_index_cols)

    return df


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build a JSON-serializable profile of the DataFrame:
    - column names and dtypes
    - null counts per column
    - min/max for numeric columns
    - value counts for low-cardinality categorical columns
    - a small sample of representative values per column

    This dict is what gets serialized into the LLM prompt -- never the
    raw DataFrame itself.
    """
    profile: Dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": {},
    }

    for col in df.columns:
        series = df[col]
        dtype_str = str(series.dtype)
        col_info: Dict[str, Any] = {
            "dtype": dtype_str,
            "null_count": int(series.isna().sum()),
            "null_pct": round(float(series.isna().mean()) * 100, 2),
        }

        non_null = series.dropna()

        if pd.api.types.is_numeric_dtype(series):
            if len(non_null) > 0:
                col_info["min"] = _safe_scalar(non_null.min())
                col_info["max"] = _safe_scalar(non_null.max())
                col_info["mean"] = _safe_scalar(non_null.mean())
        elif pd.api.types.is_datetime64_any_dtype(series):
            if len(non_null) > 0:
                col_info["min"] = str(non_null.min())
                col_info["max"] = str(non_null.max())
        else:
            try:
                n_unique = int(non_null.nunique())
                col_info["n_unique"] = n_unique
                if n_unique <= MAX_UNIQUE_FOR_CATEGORICAL_SUMMARY:
                    col_info["value_counts"] = {
                        str(k): int(v)
                        for k, v in non_null.value_counts().head(
                            MAX_UNIQUE_FOR_CATEGORICAL_SUMMARY
                        ).items()
                    }
            except TypeError:
                # Unhashable values (leftover dict/list, mixed types)
                # break nunique()/value_counts(); skip them instead of
                # crashing the whole profile.
                col_info["n_unique"] = None
                col_info["note"] = "Contains complex/unhashable values; unique count unavailable."

        sample = non_null.head(MAX_SAMPLE_VALUES).tolist()
        col_info["sample_values"] = [_safe_scalar(v) for v in sample]

        profile["columns"][col] = col_info

    return profile


def _safe_scalar(value: Any) -> Any:
    """Convert numpy/pandas scalar types into plain JSON-serializable Python types."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return str(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (dict, list)):
        # Unflattened nested structure; stringify so it stays JSON-safe.
        return str(value)[:200]
    return value


def load_and_profile(file_obj, filename: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Convenience entrypoint: parse the file and return (DataFrame, profile)."""
    df = load_dataframe(file_obj, filename)
    profile = profile_dataframe(df)
    return df, profile


if __name__ == "__main__":
    # Quick manual smoke test
    sample_csv = io.StringIO(
        "date,region,sales,rep\n"
        "2024-01-01,West,120.5,Alice\n"
        "2024-01-02,East,,Bob\n"
        "2024-01-03,West,300.0,Alice\n"
    )
    df, profile = load_and_profile(sample_csv, "sample.csv")
    print(json.dumps(profile, indent=2))
