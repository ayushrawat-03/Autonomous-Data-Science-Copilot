"""
orchestrator.py

Centralizes every LLM call in the project via LangChain's chat model
interface, so rag_repair.py's generate/run/repair loop doesn't need
to know which provider or API shape is in use.

SUPPORTED PROVIDERS (set via the COPILOT_PROVIDER env var):
- "anthropic" (default) -- Claude via langchain-anthropic, requires
  ANTHROPIC_API_KEY.
- "openai_compatible"   -- any OpenAI-chat-completions-compatible
  endpoint via langchain-openai's ChatOpenAI with a custom base_url:
  Groq, OpenRouter, Google AI Studio, a local Ollama server, etc.
  Requires COPILOT_API_BASE, COPILOT_API_KEY, and COPILOT_MODEL.

Example for Groq's free tier:
    export COPILOT_PROVIDER=openai_compatible
    export COPILOT_API_BASE=https://api.groq.com/openai/v1
    export COPILOT_API_KEY=gsk_...
    export COPILOT_MODEL=llama-3.3-70b-versatile

Example for a local Ollama server:
    export COPILOT_PROVIDER=openai_compatible
    export COPILOT_API_BASE=http://localhost:11434/v1
    export COPILOT_API_KEY=ollama          # value is ignored, just needs to be non-empty
    export COPILOT_MODEL=qwen2.5-coder:7b
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

MODEL_NAME = os.environ.get(
    "COPILOT_MODEL",
    "llama-3.3-70b-versatile",
)

_chat_model: Optional[BaseChatModel] = None


def _get_chat_model() -> BaseChatModel:
    global _chat_model

    if _chat_model is not None:
        return _chat_model

    from langchain_openai import ChatOpenAI

    api_base = os.environ.get(
        "COPILOT_API_BASE",
        "https://api.groq.com/openai/v1",
    )
    api_key = os.environ.get("COPILOT_API_KEY")
    model_name = os.environ.get(
        "COPILOT_MODEL",
        "llama-3.3-70b-versatile",
    )

    if not api_key:
        raise RuntimeError(
            "COPILOT_API_KEY is not set."
        )

    _chat_model = ChatOpenAI(
        model=model_name,
        base_url=api_base,
        api_key=api_key,
        max_tokens=1500,
    )

    return _chat_model

def _invoke(system_prompt: str, user_message: str) -> str:
    """Provider-agnostic chat call via LangChain's unified message interface."""
    chat_model = _get_chat_model()
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
    response = chat_model.invoke(messages)
    return response.content if isinstance(response.content, str) else str(response.content)


MAX_CHARTS = 5

CODE_GEN_SYSTEM_PROMPT = """\
You are a senior data analyst who writes Python/Pandas code to answer \
business questions about a DataFrame -- including broad, open-ended, or \
multi-part questions that a junior analyst would break into several \
angles rather than answer with one generic chart.

DECOMPOSE BROAD QUESTIONS:
If the question is broad or open-ended (e.g. "how is the business doing?", \
"tell me about this data", "any interesting patterns?", "how are we trending?"), \
do NOT settle for one shallow chart. Decompose it into 2-5 concrete \
sub-analyses that together give a genuinely useful answer -- for example: \
an overall trend, a breakdown by the most relevant category, a \
distribution/outlier check, and a comparison across a second dimension. \
If the question is already narrow and specific, one focused chart is fine \
-- do not pad a simple question with filler charts.

Rules you must follow exactly:
1. A pandas DataFrame named `df` is already loaded. Do not re-load or \
recreate it.
2. Use ONLY the column names and dtypes given in the schema profile below. \
Never invent or guess a column name that isn't listed.
3. Use ONLY pandas, numpy, matplotlib.pyplot (as plt), seaborn (as sns), \
and plotly (as px / go) -- these are the ONLY libraries installed in this \
sandbox. Do NOT import statsmodels, scikit-learn, scipy, or anything else, \
even for things like trend lines or regressions -- those packages are not \
available and importing them will crash the code. For a trend line, use \
numpy.polyfit(...) or a pandas rolling/ewm mean instead of statsmodels.
4. For matplotlib/seaborn charts, call plotting functions directly \
(e.g. plt.bar(...), sns.barplot(...)) rather than DataFrame.plot() or \
Series.plot() -- the latter route through pandas' plotting-backend \
resolution, which is unreliable in this sandboxed environment.
5. You may produce MULTIPLE charts (up to MAX_CHARTS_PLACEHOLDER), one per \
sub-analysis. Number them starting at 1, with no gaps (chart 1, chart 2, ...). \
For EACH chart, independently choose ONE charting library and save it to \
the matching path in OUTPUT_DIR_PLACEHOLDER, using that chart's number `n`:
   - matplotlib/seaborn: create a NEW figure per chart (plt.figure() or \
plt.subplots()), then call plt.savefig("OUTPUT_DIR_PLACEHOLDER/chart_<n>.png") \
and plt.close() before starting the next chart -- do not call plt.show().
   - plotly (preferred when a chart benefits from interactivity, e.g. \
hovering over data points, or a time series with many points): build a \
`fig` (a plotly Figure), then save it with \
fig.write_json("OUTPUT_DIR_PLACEHOLDER/chart_<n>.json").
   For any single chart number `n`, create ONLY ONE of chart_<n>.png or \
chart_<n>.json (whichever library you used for that chart) -- never both.
6. After all charts are saved, print your findings as separate print() \
statements -- one per distinct insight/sub-analysis (1 line if the question \
was narrow, up to 5 short lines for a broad/decomposed question). Each line \
should be a complete, plain-English sentence or two. These prints must be \
the LAST lines of output, in the same order as the charts they relate to.
7. Handle nulls sensibly (e.g. dropna() or fillna() where it affects the \
calculation) given the null counts in the profile.
8. Return ONLY a Python code block. No explanation before or after it.
"""


def _extract_code_block(text: str) -> str:
    """Pull the first ```python ... ``` (or bare ``` ... ```) block out of an LLM response."""
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # No code fence found; fall back to the raw text.
    return text.strip()


def generate_code(question: str, profile: Dict[str, Any], chart_output_dir: str) -> str:
    """
    Ask the LLM to write Pandas/Matplotlib/Plotly code answering
    `question` against a DataFrame matching `profile`. The LLM may emit
    multiple numbered charts (chart_1, chart_2, ...) into
    `chart_output_dir`, one per sub-analysis, and picks the charting
    library independently for each one.
    """
    system_prompt = (
        CODE_GEN_SYSTEM_PROMPT
        .replace("OUTPUT_DIR_PLACEHOLDER", chart_output_dir)
        .replace("MAX_CHARTS_PLACEHOLDER", str(MAX_CHARTS))
    )

    user_message = (
        f"Schema profile (JSON):\n{json.dumps(profile, indent=2)}\n\n"
        f"Business question: {question}"
    )

    text = _invoke(system_prompt, user_message)
    return _extract_code_block(text)


REPAIR_SYSTEM_PROMPT = """\
You are debugging Python/Pandas code that just failed. You will be given \
the original code, the error it produced, the DataFrame's schema profile, \
and (if relevant) documentation for the function involved.

Rules:
1. Fix the code so it runs successfully against the schema given.
2. Keep using only pandas, numpy, matplotlib.pyplot (as plt), seaborn \
(as sns), and plotly (as px / go) -- the same constraints as before. If the \
error is an ImportError/ModuleNotFoundError for anything else (statsmodels, \
sklearn, scipy, etc.), that package is NOT installed -- remove the import \
and rewrite that part of the analysis using only the libraries above (e.g. \
numpy.polyfit or a pandas rolling/ewm mean instead of statsmodels).
3. Do not invent column names -- use only what's in the schema profile.
4. Preserve the original intent and structure of the code -- the same set \
of numbered charts (chart_1, chart_2, ...), each still saved as EITHER \
chart_<n>.png (matplotlib/seaborn) or chart_<n>.json (plotly), and the same \
per-chart print() insight lines as the last lines of output -- unless the \
bug requires changing the underlying approach. Do not drop charts or \
insights that were working just to fix an unrelated error elsewhere.
5. Return ONLY the corrected Python code block. No explanation.
"""


def repair_code_with_llm(
    original_code: str,
    traceback_text: str,
    profile: Dict[str, Any],
    doc_context: Optional[str] = None,
) -> str:
    """
    Ask the LLM to fix `original_code` given the traceback it produced.
    `doc_context` is an optional retrieved documentation snippet (from
    the RAG loop) relevant to the failing function.
    """
    parts = [
        f"Schema profile (JSON):\n{json.dumps(profile, indent=2)}",
        f"Original code:\n```python\n{original_code}\n```",
        f"Error / traceback:\n```\n{traceback_text}\n```",
    ]
    if doc_context:
        parts.append(f"Additional context:\n{doc_context}")

    user_message = "\n\n".join(parts)

    text = _invoke(REPAIR_SYSTEM_PROMPT, user_message)
    return _extract_code_block(text)


FOLLOWUP_SYSTEM_PROMPT = """\
You are a sharp data analyst suggesting what a curious stakeholder would \
naturally ask NEXT, after seeing the answer to their last question about \
this dataset.

Rules:
1. Suggest exactly 3 follow-up questions.
2. Each must be answerable using ONLY the columns in the schema profile \
given -- never reference a column that isn't listed.
3. Each must go DEEPER or SIDEWAYS from the previous question/insight -- \
not repeat it. Prefer things like: a different slice of the same finding, \
a "why" behind the pattern, a comparison the first answer didn't cover, or \
a natural next drill-down.
4. Keep each question short (under 12 words) and phrased the way a \
non-technical business person would ask it, not a technical query.
5. Return ONLY a JSON array of 3 strings. No explanation, no markdown \
fences, nothing else.
"""


def suggest_followups(question: str, insight: str, profile: Dict[str, Any]) -> List[str]:
    """
    Ask the LLM for 3 short, schema-grounded follow-up questions given
    the question just asked and the insight it produced. Best-effort:
    on any parsing/provider failure, returns an empty list so the UI
    can simply skip showing suggestions rather than breaking the turn.
    """
    columns = list(profile.get("columns", {}).keys())
    user_message = (
        f"Available columns: {columns}\n\n"
        f"Previous question: {question}\n"
        f"Insight it produced: {insight}"
    )

    try:
        text = _invoke(FOLLOWUP_SYSTEM_PROMPT, user_message)
        cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        suggestions = json.loads(cleaned)
        if isinstance(suggestions, list):
            return [str(s).strip() for s in suggestions if str(s).strip()][:3]
    except Exception:  # noqa: BLE001 - suggestions are a nice-to-have, never fatal
        pass
    return []


if __name__ == "__main__":
    # Manual smoke test (requires the configured provider's API key to be set)
    fake_profile = {
        "n_rows": 100,
        "columns": {
            "region": {"dtype": "object", "n_unique": 4, "sample_values": ["West", "East"]},
            "sales": {"dtype": "float64", "min": 10.0, "max": 500.0},
        },
    }
    code = generate_code(
        "How is the business doing overall?", fake_profile, "./output"
    )
    print(code)
