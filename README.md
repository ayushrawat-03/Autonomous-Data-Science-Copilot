# Autonomous Data Science Co-Pilot

An AI agent that lets a non-technical user upload a CSV/XLSX/JSON file, ask a
question in plain English, and get back a chart + written insight — with
**self-healing**: if the generated Pandas code errors out, the agent retrieves
relevant official documentation via RAG, rewrites the code, and retries
automatically.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

export ANTHROPIC_API_KEY=sk-ant-...   # required (or see "Using a free provider" below)

# One-time: build the RAG index. This extracts REAL docstrings from your
# installed pandas/matplotlib/seaborn (see "RAG documentation source" below),
# then downloads the sentence-transformers embedding model (~90MB) on first run.
python docs_ingest/build_index.py

streamlit run app.py --server.fileWatcherType none
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

(`--server.fileWatcherType none` avoids a noisy-but-harmless traceback from
Streamlit's file watcher probing `sentence-transformers`' optional
dependencies — see the FAQ at the bottom if you want auto-reload instead.)

## Using a free provider instead of Anthropic billing

LLM calls go through LangChain's chat model interface (see `orchestrator.py`),
so switching providers is a config change, not a rewrite. Instead of Claude,
point the app at any OpenAI-chat-completions-compatible endpoint — including
free options as of mid-2026 (verify current terms before relying on one):

**Groq (recommended — fast, free, no local download):**
```bash
export COPILOT_PROVIDER=openai_compatible
export COPILOT_API_BASE=https://api.groq.com/openai/v1
export COPILOT_API_KEY=gsk_...                    # from console.groq.com
export COPILOT_MODEL=llama-3.3-70b-versatile
```

**A local Ollama server (fully offline, needs disk space for model weights):**
```bash
# after installing Ollama and running `ollama pull qwen2.5-coder:7b`
export COPILOT_PROVIDER=openai_compatible
export COPILOT_API_BASE=http://localhost:11434/v1
export COPILOT_API_KEY=ollama    # Ollama ignores the value, but the client needs a non-empty string
export COPILOT_MODEL=qwen2.5-coder:7b
```

Any other OpenAI-compatible provider (OpenRouter, Google AI Studio, etc.)
works the same way.

## The UI

`app.py` shows only the **current** question/answer on the main page: the
chart(s), the plain-English insight, badges for whichever repair path the
self-healing loop took, and an expandable panel with the traceback and
generated code (downloadable as `.py`, alongside each chart as `.png`).
There's no chat history — asking a new question replaces the one on screen,
and nothing is kept once you move on. All CSS/layout lives in `theme.py`,
kept separate from the control flow in `app.py`.

## Multi-chart, broader answers

The agent isn't limited to one chart per question. For narrow
questions ("total sales by region?") it still produces a single
focused chart. For broad or open-ended questions ("how is the
business doing overall?"), the code-gen prompt instructs it to
decompose the question into 2-5 concrete sub-analyses -- e.g. an
overall trend, a category breakdown, an outlier/distribution check,
and a second-dimension comparison -- and emit one numbered chart per
sub-analysis (`chart_1.png`/`.json`, `chart_2.png`/`.json`, ...),
choosing matplotlib/seaborn or Plotly independently for each. It then
prints one insight line per sub-analysis, in the same order as the
charts. `renderer.py` scans the output directory for however many
numbered charts were produced (rather than assuming exactly one), and
the UI lays them out in a responsive grid. The self-healing repair
loop preserves this structure: a repair attempt is told to keep the
same set of numbered charts and insight lines unless the bug requires
changing the underlying approach.

## Follow-up questions

After each successful answer, the agent suggests 3 short, schema-grounded
follow-up questions -- a different slice of the same finding, a "why" behind
the pattern, or a natural next drill-down -- as clickable chips
(`orchestrator.suggest_followups`). Clicking one loads it straight into the
question box. This is capped at 3 follow-up rounds per answer -- after
that, the chips are replaced with a note to type your next question
directly. It's a best-effort call: if it fails for any reason, the chips
are simply omitted rather than breaking the turn.

The "Ideas for this file" suggestions shown under the question box (from
`suggestions.py`) are plain text, not buttons -- they're schema-grounded
examples for you to type or adapt, not one-click actions.

## How it works

```
Upload file (CSV/XLSX/JSON)
        │
        ▼
data_layer.py    → parses file, builds a schema PROFILE (dtypes, nulls,
                     ranges, samples) — raw rows never leave this stage.
                     JSON is routed through pd.json_normalize so nested
                     objects (e.g. {"details": {"age": 30}}) flatten into
                     columns instead of crashing profiling.
        │
        ▼
orchestrator.py  → LangChain ChatPromptTemplate + chat model call
                     (Claude via langchain-anthropic, or any OpenAI-
                     compatible provider via langchain-openai) generates
                     Pandas/Matplotlib/Seaborn/Plotly code
        │
        ▼
sandbox.py       → runs the code in a restricted subprocess:
                     - blocklist of dangerous modules (subprocess, socket,
                       ctypes, multiprocessing, ...) blocked at import time
                     - "os" itself stays importable (matplotlib's font
                       manager needs it internally) but its dangerous
                       functions (system/popen/exec*/spawn*) are neutered
                     - writes confined to ./output
                     - hard timeout, whole process-group killed on expiry
        │
        ├── success ──────────────────────────────► renderer.py → chart
        │                                             (PNG or interactive
        │                                             Plotly) + insight
        │
        └── failure (traceback)
                │
                ▼
        rag_repair.py picks ONE of two repair paths:
          • KeyError/AttributeError on a column  → SCHEMA-grounded repair
            (re-inject the real column list, no RAG lookup needed)
          • anything else (wrong API usage, etc.) → DOC-grounded repair
            (embed the traceback, retrieve the matching REAL pandas
            docstring chunk from ChromaDB, feed it back to the LLM)
                │
                ▼
        retries (capped, default 3) → back to sandbox.py
```

## RAG documentation source (real docs, not a hand-written corpus)

`docs_ingest/build_index.py` extracts docstrings **directly from your
installed pandas/matplotlib/seaborn packages** via `inspect.getdoc()`,
covering 28 functions across all 5 use cases (groupby, merge, pivot_table,
resample, rolling, isna, duplicated, describe, cut, plt.savefig, sns.barplot,
and more — see `FUNCTION_REFERENCES` in that file for the full list).

This is a deliberate choice over scraping pandas.pydata.org:
- These docstrings are the literal source Sphinx renders into the official
  online docs — extracting them isn't a workaround, it's going to the source.
- Zero network dependency at ingestion time, and automatically version-matched
  to whatever pandas version is actually installed (no scraper-vs-version
  drift).
- The original hand-curated notes are kept as a **supplementary** fallback at
  `docs_ingest/corpus/supplementary.json` for topics that don't map to a
  single function's docstring (e.g. general indexing gotchas).

## Agent orchestration (LangChain)

`orchestrator.py` uses LangChain's `ChatAnthropic`/`ChatOpenAI` chat model
classes and message-based prompting, so the provider is a config switch (see
`COPILOT_PROVIDER` above), not a code change. The generate → run → repair →
retry control loop itself (`rag_repair.py`) is a plain, auditable Python loop
with a hard retry cap — not a LangGraph agent deciding its own next action.
That's intentional for a pipeline that executes LLM-written code: a fixed,
inspectable sequence is easier to reason about for security than a fully
autonomous agent. LangGraph is already installed as a LangChain dependency if
a more dynamic agent is wanted later.

## The 5 use cases (from the project brief)

The sandbox environment and libraries support each use case's typical
operations (tested with hand-written code standing in for LLM output).
The LLM's actual code-generation quality for each is worth
spot-checking once you have a provider configured.

| # | Use case | Example question | Confirmed working |
|---|---|---|---|
| 1 | Sales Dashboard | "What are total sales by region?" | ✅ groupby + bar chart |
| 2 | Data Quality Audit | "Which columns have missing values, duplicates, or outliers?" | ✅ isna/duplicated/outlier detection, multi-line report |
| 3 | Trend Analysis | "Is there a trend in sales over time?" | ✅ interactive Plotly line chart |
| 4 | Cohort Analysis | "Group customers into spend-based cohorts and compare them." | ✅ pd.cut segmentation + grouped chart |
| 5 | Ad-hoc Queries | "What's the average order value by category?" | ✅ groupby + bar chart |

## Project structure

```
app.py                    Streamlit UI control flow (upload -> ask -> analyze)
theme.py                   Design system: CSS + reusable UI components for app.py
tokens.py                   Plain color/font constants used by theme.py
data_layer.py              File parsing + profiling (CSV/XLSX/JSON, nested JSON handling)
orchestrator.py             LangChain-based LLM calls (provider swappable via env vars)
sandbox.py                  Restricted subprocess executor
rag_repair.py                Self-healing loop (schema-grounded + doc-grounded repair)
renderer.py                  Detects chart type (PNG/Plotly) + formats insight text
docs_ingest/
  build_index.py             Extracts real docstrings + embeds into ChromaDB
  corpus/supplementary.json   Hand-written fallback notes (not the primary source)
output/                     Generated charts land here
chroma_db/                  Persistent vector store (created by build_index.py)
```

## Known limitations

- **Sandbox is subprocess-based, not container-based.** The import
  blocklist and neutered `os` functions stop accidental misuse, but an
  in-interpreter guard is not a hard security boundary against a
  deliberately adversarial payload (see `sandbox.py`). For real
  deployment, run each execution in its own Docker container with
  `--network none`, memory/CPU caps, and a read-only filesystem.
- **RAG corpus covers 28 functions, not the entire pandas API surface.**
  Extend coverage by adding entries to `FUNCTION_REFERENCES` in
  `docs_ingest/build_index.py` and re-running it.
- **Silent correctness failures aren't caught.** The self-healing loop
  only triggers on exceptions/tracebacks — code that runs but produces
  a subtly wrong answer won't trigger a retry.
- **Retry loop is a fixed Python loop, not a LangGraph agent.** See
  "Agent orchestration" above.

## FAQ

**Q: Streamlit prints a `ModuleNotFoundError: No module named 'torchvision'`
traceback repeatedly, but the app still works.**
A: Cosmetic. Streamlit's file watcher probes every loaded module (including
`transformers`, a `sentence-transformers` dependency) to support
auto-reloading on file edits; one of `transformers`' optional submodules
lazily imports `torchvision`, which isn't installed, and that failed probe
prints a traceback. Run with `--server.fileWatcherType none` (as in Setup
above) to silence it — trade-off: you'll need to manually refresh the browser
after editing code, instead of it auto-reloading.

**Q: `anthropic.AuthenticationError: invalid x-api-key` / `401`.**
A: Usually one of: (1) the key was exported in a different terminal session
than the one running Streamlit — env vars don't carry across terminals; (2)
hidden whitespace/newline in the copied key; (3) an OpenAI key was used where
an Anthropic key was expected (they're not interchangeable — see the
"Using a free provider" section if you want a non-Anthropic option instead).

**Q: `pip install` fails with "No space left on device."**
A: That's disk space on your machine, not a venv-specific limit — run
`df -h /` to check, and `pip cache purge` plus `rm -rf ~/.cache/huggingface`
to reclaim space. Budget roughly 2–3GB free for the full dependency set
(sentence-transformers' `torch` dependency is the largest single piece).
