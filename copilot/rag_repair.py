"""
rag_repair.py

The self-healing loop: when generated code fails in the sandbox, this
module decides how to fix it and drives the generate -> run -> repair
-> retry cycle.

Two repair paths:
  - SCHEMA-GROUNDED REPAIR: KeyError/AttributeError referencing a
    column -- re-inject the real column list from the profile, no
    RAG lookup needed.
  - DOC-GROUNDED REPAIR (RAG): genuine API misuse (wrong argument,
    dtype mismatch, etc.) -- extract the failing function name from
    the traceback, embed it, retrieve the matching doc chunk from
    ChromaDB, and feed it back to the LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from docs_ingest.build_index import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)
from orchestrator import generate_code, repair_code_with_llm
from sandbox import run_in_sandbox

# Functions with curated docs, used to pull a candidate name out of a
# traceback for the doc-grounded path. Kept in sync with
# docs_ingest/build_index.py's FUNCTION_REFERENCES.
KNOWN_FUNCTIONS = [
    "groupby", "merge", "pivot_table", "astype", "to_datetime",
    "loc", "iloc", "fillna", "dropna", "value_counts", "savefig",
    "resample", "rolling", "isna", "duplicated", "describe", "corr",
    "sort_values", "nlargest", "pct_change", "diff", "melt", "cut",
    "qcut", "subplots", "barplot", "lineplot", "heatmap",
]

_embedder: Optional[SentenceTransformer] = None
_collection = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


@dataclass
class RepairAttempt:
    attempt_number: int
    repair_path: str  # "schema" | "doc" | "none"
    traceback: Optional[str]
    doc_used: Optional[Dict[str, Any]] = None


@dataclass
class RunResult:
    success: bool
    final_code: str
    stdout: str
    chart_output_dir: Optional[str]
    attempts: List[RepairAttempt] = field(default_factory=list)
    gave_up_reason: Optional[str] = None


def _is_column_reference_error(traceback_text: str) -> bool:
    """KeyError/AttributeError are the classic 'guessed a column that doesn't exist' errors."""
    return bool(re.search(r"\b(KeyError|AttributeError)\b", traceback_text))


_MISSING_MODULE_RE = re.compile(r"No module named ['\"]([\w.]+)['\"]")


def _is_import_error(traceback_text: str) -> bool:
    """ModuleNotFoundError/ImportError -- the LLM tried a library that isn't installed."""
    return bool(re.search(r"\b(ModuleNotFoundError|ImportError)\b", traceback_text))


def _import_error_note(traceback_text: str) -> str:
    match = _MISSING_MODULE_RE.search(traceback_text)
    module = match.group(1).split(".")[0] if match else "that library"
    return (
        f"'{module}' is not installed in this sandbox. Only pandas, numpy, "
        "matplotlib.pyplot, seaborn, and plotly are available. Rewrite the "
        f"code without importing '{module}' -- e.g. use numpy.polyfit or a "
        "pandas rolling/ewm mean for trend lines instead of statsmodels, and "
        "compute simple stats/correlations manually instead of scikit-learn "
        "or scipy."
    )


def _extract_failing_function(traceback_text: str) -> Optional[str]:
    """Best-effort extraction of a known Pandas/Matplotlib function name from a traceback."""
    for fn in KNOWN_FUNCTIONS:
        if re.search(rf"\.{fn}\(", traceback_text) or f"'{fn}'" in traceback_text:
            return fn
    return None


def _retrieve_doc_for_error(traceback_text: str) -> Optional[Dict[str, Any]]:
    """
    Embed the traceback (or the extracted function name, if found) and
    retrieve the closest matching doc chunk from ChromaDB.
    """
    function_name = _extract_failing_function(traceback_text)
    query_text = function_name if function_name else traceback_text

    embedder = _get_embedder()
    collection = _get_collection()

    query_embedding = embedder.encode([query_text]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=1)

    if not results["documents"] or not results["documents"][0]:
        return None

    return {
        "text": results["documents"][0][0],
        "function_name": results["metadatas"][0][0]["function_name"],
        "source_url": results["metadatas"][0][0]["source_url"],
    }


def run_with_self_healing(
    question: str,
    df,
    profile: Dict[str, Any],
    chart_output_dir: str = "./output",
    max_retries: int = 3,
) -> RunResult:
    """
    Drive the full generate -> run -> repair -> retry loop.

    The generated code may produce multiple numbered charts
    (chart_1.png/.json, chart_2.png/.json, ...) into `chart_output_dir`;
    `renderer.py` scans the directory afterwards rather than this
    function enumerating them.

    Returns a RunResult with the final code, success flag, and a log
    of every repair attempt (path used, doc retrieved if any).
    """
    code = generate_code(question, profile, chart_output_dir)
    attempts: List[RepairAttempt] = []

    for attempt_num in range(max_retries + 1):
        result = run_in_sandbox(code, df, output_dir=chart_output_dir)

        if result["success"]:
            attempts.append(
                RepairAttempt(
                    attempt_number=attempt_num,
                    repair_path="none",
                    traceback=None,
                )
            )
            return RunResult(
                success=True,
                final_code=code,
                stdout=result["stdout"],
                chart_output_dir=chart_output_dir,
                attempts=attempts,
            )

        traceback_text = result["traceback"] or result["stderr"]

        if attempt_num == max_retries:
            attempts.append(
                RepairAttempt(
                    attempt_number=attempt_num,
                    repair_path="none",
                    traceback=traceback_text,
                )
            )
            return RunResult(
                success=False,
                final_code=code,
                stdout=result["stdout"],
                chart_output_dir=None,
                attempts=attempts,
                gave_up_reason=(
                    f"Gave up after {max_retries} repair attempts. "
                    f"Last error:\n{traceback_text}"
                ),
            )

        if _is_import_error(traceback_text):
            # IMPORT-ERROR REPAIR: tell the LLM which module is
            # unavailable and to rewrite without it.
            code = repair_code_with_llm(
                original_code=code,
                traceback_text=traceback_text,
                profile=profile,
                doc_context=_import_error_note(traceback_text),
            )
            attempts.append(
                RepairAttempt(
                    attempt_number=attempt_num,
                    repair_path="import",
                    traceback=traceback_text,
                )
            )
        elif _is_column_reference_error(traceback_text):
            # SCHEMA-GROUNDED REPAIR: re-show the LLM the real column list.
            code = repair_code_with_llm(
                original_code=code,
                traceback_text=traceback_text,
                profile=profile,
                doc_context=None,
            )
            attempts.append(
                RepairAttempt(
                    attempt_number=attempt_num,
                    repair_path="schema",
                    traceback=traceback_text,
                )
            )
        else:
            # DOC-GROUNDED REPAIR: retrieve relevant Pandas/Matplotlib
            # documentation for the failing function and hand it to
            # the LLM alongside the traceback.
            doc = _retrieve_doc_for_error(traceback_text)
            doc_context = (
                f"[{doc['function_name']}] {doc['text']} (source: {doc['source_url']})"
                if doc
                else None
            )
            code = repair_code_with_llm(
                original_code=code,
                traceback_text=traceback_text,
                profile=profile,
                doc_context=doc_context,
            )
            attempts.append(
                RepairAttempt(
                    attempt_number=attempt_num,
                    repair_path="doc",
                    traceback=traceback_text,
                    doc_used=doc,
                )
            )

    # Unreachable, but keeps type checkers happy.
    raise RuntimeError("Repair loop exited unexpectedly")
