"""
renderer.py

Takes the result of a sandbox run and packages every chart the
generated code produced (chart_1, chart_2, ...) plus the
plain-English insight text, for display in Streamlit.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Dict, List, Tuple

from rag_repair import RunResult

# A rendered chart: (file_path, "image" | "plotly", chart_number)
Chart = Tuple[str, str, int]

_CHART_FILENAME_RE = re.compile(r"chart_(\d+)\.(png|json)$")


def _discover_charts(chart_output_dir: str) -> List[Chart]:
    """
    Scan `chart_output_dir` for chart_<n>.png / chart_<n>.json files
    and return them sorted by chart number. If both extensions exist
    for the same number, the PNG wins.
    """
    if not chart_output_dir or not os.path.isdir(chart_output_dir):
        return []

    found: Dict[int, Chart] = {}
    for path in glob.glob(os.path.join(chart_output_dir, "chart_*.png")):
        match = _CHART_FILENAME_RE.search(os.path.basename(path))
        if match:
            found[int(match.group(1))] = (path, "image", int(match.group(1)))
    for path in glob.glob(os.path.join(chart_output_dir, "chart_*.json")):
        match = _CHART_FILENAME_RE.search(os.path.basename(path))
        if match:
            n = int(match.group(1))
            found.setdefault(n, (path, "plotly", n))

    return [found[n] for n in sorted(found)]


def render_result(run_result: RunResult) -> Tuple[List[Chart], str]:
    """
    Returns (charts, insight_text).

    `charts` is a list of (path, chart_type, chart_number) tuples in
    chart-number order, empty if no charts were produced or the run
    failed. A single printed stdout line is shown as-is; multiple
    lines are shown as a bulleted list.
    """
    if not run_result.success:
        return [], run_result.gave_up_reason or "The analysis could not be completed."

    lines = [line for line in run_result.stdout.strip().splitlines() if line.strip()]
    if not lines:
        insight = "(No insight text was printed.)"
    elif len(lines) == 1:
        insight = lines[0]
    else:
        insight = "\n".join(f"- {line}" for line in lines)

    charts = _discover_charts(run_result.chart_output_dir)
    return charts, insight
