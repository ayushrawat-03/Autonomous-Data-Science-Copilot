"""
docs_ingest/build_index.py

Embeds documentation into a persistent ChromaDB collection, chunked
per function/method, for the RAG self-repair loop.

Docstrings are extracted directly from the installed
pandas/matplotlib/seaborn packages via `inspect.getdoc()` rather than
scraped from the docs website: no network dependency at ingestion
time, and always version-matched to what's actually installed. Each
chunk is one function's full docstring (signature, parameters, notes),
so a KeyError on .groupby() retrieves the complete .groupby()
reference entry rather than an arbitrary window of text.

Run this once (or whenever you add functions to FUNCTION_REFERENCES):
    python docs_ingest/build_index.py
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List

import chromadb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHROMA_DB_PATH = str(Path(__file__).parent.parent / "chroma_db")
COLLECTION_NAME = "pandas_docs"

# Supplementary corpus: hand-written notes for topics that aren't a
# single function's docstring (e.g. general indexing gotchas). Fills
# gaps; not the primary source.
SUPPLEMENTARY_CORPUS_PATH = Path(__file__).parent / "corpus" / "supplementary.json"

# Maps a friendly function_name (used for retrieval/display) to a
# callable returning the live object whose docstring we want, and a
# doc URL for attribution in the UI.
FUNCTION_REFERENCES: Dict[str, Dict[str, Any]] = {
    "groupby": {
        "getter": lambda: pd.DataFrame.groupby,
        "url": "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html",
    },
    "merge": {
        "getter": lambda: pd.DataFrame.merge,
        "url": "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html",
    },
    "pivot_table": {
        "getter": lambda: pd.pivot_table,
        "url": "https://pandas.pydata.org/docs/reference/api/pandas.pivot_table.html",
    },
    "astype": {
        "getter": lambda: pd.Series.astype,
        "url": "https://pandas.pydata.org/docs/reference/api/pandas.Series.astype.html",
    },
    "to_datetime": {
        "getter": lambda: pd.to_datetime,
        "url": "https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html",
    },
    "loc": {
        "getter": lambda: pd.DataFrame.loc,
        "url": "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html",
    },
    "iloc": {
        "getter": lambda: pd.DataFrame.iloc,
        "url": "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html",
    },
    "fillna": {
        "getter": lambda: pd.DataFrame.fillna,
        "url": "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html",
    },
    "dropna": {
        "getter": lambda: pd.DataFrame.dropna,
        "url": "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html",
    },
    "value_counts": {
        "getter": lambda: pd.Series.value_counts,
        "url": "https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html",
    },
    "resample": {
        "getter": lambda: pd.DataFrame.resample,
        "url": "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html",
    },
    "rolling": {
        "getter": lambda: pd.DataFrame.rolling,
        "url": "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html",
    },
    "isna": {
        "getter": lambda: pd.DataFrame.isna,
        "url": "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isna.html",
    },
    "duplicated": {
        "getter": lambda: pd.DataFrame.duplicated,
        "url": "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.duplicated.html",
    },
    "describe": {
        "getter": lambda: pd.DataFrame.describe,
        "url": "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html",
    },
    "corr": {
        "getter": lambda: pd.DataFrame.corr,
        "url": "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html",
    },
    "sort_values": {
        "getter": lambda: pd.DataFrame.sort_values,
        "url": "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html",
    },
    "nlargest": {
        "getter": lambda: pd.DataFrame.nlargest,
        "url": "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.nlargest.html",
    },
    "pct_change": {
        "getter": lambda: pd.Series.pct_change,
        "url": "https://pandas.pydata.org/docs/reference/api/pandas.Series.pct_change.html",
    },
    "diff": {
        "getter": lambda: pd.Series.diff,
        "url": "https://pandas.pydata.org/docs/reference/api/pandas.Series.diff.html",
    },
    "melt": {
        "getter": lambda: pd.melt,
        "url": "https://pandas.pydata.org/docs/reference/api/pandas.melt.html",
    },
    "cut": {
        "getter": lambda: pd.cut,
        "url": "https://pandas.pydata.org/docs/reference/api/pandas.cut.html",
    },
    "qcut": {
        "getter": lambda: pd.qcut,
        "url": "https://pandas.pydata.org/docs/reference/api/pandas.qcut.html",
    },
    "plt.savefig": {
        "getter": lambda: plt.savefig,
        "url": "https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html",
    },
    "plt.subplots": {
        "getter": lambda: plt.subplots,
        "url": "https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html",
    },
    "sns.barplot": {
        "getter": lambda: sns.barplot,
        "url": "https://seaborn.pydata.org/generated/seaborn.barplot.html",
    },
    "sns.lineplot": {
        "getter": lambda: sns.lineplot,
        "url": "https://seaborn.pydata.org/generated/seaborn.lineplot.html",
    },
    "sns.heatmap": {
        "getter": lambda: sns.heatmap,
        "url": "https://seaborn.pydata.org/generated/seaborn.heatmap.html",
    },
}


def _truncate_at_examples_section(docstring: str, max_chars: int = 2500) -> str:
    """
    Cut the docstring at its "Examples" section if present (mostly
    noise for a repair prompt, which needs Parameters/Raises/Notes),
    and hard-cap length so a few chunks don't blow the prompt budget.
    """
    match = re.search(r"\n\s*Examples\s*\n\s*-+\s*\n", docstring)
    if match:
        docstring = docstring[: match.start()]
    return docstring[:max_chars].strip()


def _load_supplementary_corpus() -> List[Dict[str, Any]]:
    if SUPPLEMENTARY_CORPUS_PATH.exists():
        with open(SUPPLEMENTARY_CORPUS_PATH) as f:
            return json.load(f)
    return []


def build_corpus_from_installed_libraries() -> List[Dict[str, Any]]:
    """Extract real docstrings from the installed libraries for every entry in FUNCTION_REFERENCES."""
    entries: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for function_name, ref in FUNCTION_REFERENCES.items():
        try:
            obj = ref["getter"]()
            doc = inspect.getdoc(obj)
            if not doc:
                skipped.append(function_name)
                continue
            entries.append(
                {
                    "function_name": function_name,
                    "source_url": ref["url"],
                    "text": _truncate_at_examples_section(doc),
                }
            )
        except Exception as e:  # noqa: BLE001 -- log and continue, don't let one bad entry kill ingestion
            print(f"  Skipping '{function_name}': {e}")
            skipped.append(function_name)

    if skipped:
        print(f"Skipped {len(skipped)} entries (no docstring found or getter failed): {skipped}")

    return entries


def build_index() -> None:
    print("Extracting real docstrings from installed pandas/matplotlib/seaborn...")
    entries = build_corpus_from_installed_libraries()
    entries.extend(_load_supplementary_corpus())

    if not entries:
        raise RuntimeError("No corpus entries were built -- check FUNCTION_REFERENCES.")

    print(f"Built {len(entries)} doc entries ({len(entries)} from live docstrings + supplementary).")
    print(f"Embedding with '{EMBEDDING_MODEL_NAME}' ...")

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    texts = [e["text"] for e in entries]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        COLLECTION_NAME, metadata={"embedding_model": EMBEDDING_MODEL_NAME}
    )

    collection.add(
        ids=[f"doc_{i}" for i in range(len(entries))],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {"function_name": e["function_name"], "source_url": e["source_url"]}
            for e in entries
        ],
    )

    print(f"Indexed {collection.count()} chunks into ChromaDB at {CHROMA_DB_PATH}")


if __name__ == "__main__":
    build_index()
