"""
sandbox.py

Runs LLM-generated Pandas/Matplotlib code in an isolated subprocess,
with an import guard and a neutered `os` module as the security
boundary within that process.

- Subprocess (not exec()): gives the generated code its own OS
  process, so a crash or runaway loop can't take down the parent app,
  and it can be killed on timeout.
- Import guard: BLOCKED_IMPORTS below is a blocklist, not an
  allowlist -- matplotlib/pandas lazily import many harmless stdlib
  modules internally, so an allowlist breaks on ordinary chart
  rendering. The blocklist targets modules that reach the network or
  spawn processes (subprocess, socket, ctypes, multiprocessing, etc.).
- `os` stays importable (matplotlib's font manager needs it at
  render time) but its dangerous functions (system, popen, exec*,
  spawn*, fork) are monkeypatched to raise.

Limitation: an import guard enforced inside the same interpreter is
not a hard security boundary -- it can potentially be bypassed via
introspection (__builtins__, getattr chains, etc.). This is
appropriate for trusted internal use, not untrusted public
deployment. A real boundary would run each execution in its own
container (`docker run --network none --read-only ...`).
"""

from __future__ import annotations

import os
import pickle
import signal
import subprocess
import sys
import tempfile
import textwrap
from typing import Any, Dict

# Libraries the generated code is expected to use (documentation only;
# not enforced directly -- see BLOCKED_IMPORTS below).
ALLOWED_IMPORTS = {"pandas", "numpy", "matplotlib", "seaborn", "plotly"}

# Modules that would let generated code spawn processes or reach the
# network. "os" and "sys" are deliberately excluded: matplotlib's font
# manager lazily imports "os" at render time, so it's left importable
# with only its dangerous functions neutralized (see the guard below).
BLOCKED_IMPORTS = {
    "subprocess", "socket", "shutil", "ctypes",
    "multiprocessing", "pty", "pwd", "grp", "resource", "fcntl",
    "urllib", "http", "ftplib", "telnetlib", "smtplib", "poplib",
    "imaplib", "nntplib", "webbrowser", "requests", "urllib3",
    "socketserver", "asyncio",
}

# Injected into the subprocess before the generated code runs.
# Wraps __import__ to block BLOCKED_IMPORTS, and neutralizes the
# process-spawning functions on the already-imported "os" module.
# Uses plain string substitution (not .format/f-string) to avoid the
# literal { } braces in the dict literals below.
_IMPORT_GUARD_TEMPLATE = """
import builtins as _builtins
import os as _os_for_guard

_BLOCKED = BLOCKED_IMPORTS_PLACEHOLDER
_real_import = _builtins.__import__

def _guarded_import(name, *args, **kwargs):
    top_level = name.split(".")[0]
    if top_level in _BLOCKED:
        raise ImportError(
            "Import of '" + name + "' is blocked in this sandbox for security reasons."
        )
    return _real_import(name, *args, **kwargs)

_builtins.__import__ = _guarded_import

def _blocked_os_call(*args, **kwargs):
    raise PermissionError(
        "This operation is blocked in the sandbox for security reasons."
    )

for _dangerous_fn in (
    "system", "popen", "popen2", "popen3", "popen4",
    "execl", "execle", "execlp", "execlpe",
    "execv", "execve", "execvp", "execvpe",
    "spawnl", "spawnle", "spawnlp", "spawnlpe",
    "spawnv", "spawnve", "spawnvp", "spawnvpe",
    "posix_spawn", "fork", "forkpty",
):
    if hasattr(_os_for_guard, _dangerous_fn):
        setattr(_os_for_guard, _dangerous_fn, _blocked_os_call)

# Block direct filesystem writes outside the designated output path.
_real_open = _builtins.open
_ALLOWED_WRITE_PREFIX = OUTPUT_DIR_PLACEHOLDER

def _guarded_open(file, mode="r", *args, **kwargs):
    if any(m in mode for m in ("w", "a", "x", "+")):
        file_str = str(file)
        if not file_str.startswith(_ALLOWED_WRITE_PREFIX):
            raise PermissionError(
                "Writing to '" + str(file) + "' is blocked. "
                "Generated code may only write inside " + _ALLOWED_WRITE_PREFIX
            )
    return _real_open(file, mode, *args, **kwargs)

_builtins.open = _guarded_open
"""


def _build_run_script(user_code: str, df_pickle_path: str, output_dir: str) -> str:
    """
    Loads the DataFrame and imports the allowed libraries first, using
    the real unguarded import machinery (their internal stdlib
    imports must not be blocked). The import guard is installed only
    after those finish, so it applies to whatever the generated code
    imports from that point on.
    """
    guard = _IMPORT_GUARD_TEMPLATE.replace(
        "BLOCKED_IMPORTS_PLACEHOLDER", repr(BLOCKED_IMPORTS)
    ).replace("OUTPUT_DIR_PLACEHOLDER", repr(output_dir))

    header = textwrap.dedent(f"""
        import matplotlib
        matplotlib.use("Agg")

        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        try:
            import seaborn as sns
        except ImportError:
            sns = None
        try:
            import plotly.express as px
            import plotly.graph_objects as go
        except ImportError:
            px = None
            go = None

        # Force matplotlib's Agg backend to initialize now, while
        # unguarded imports are still allowed, so it isn't triggered
        # (and blocked) by the first plot call in generated code.
        _warmup_fig = plt.figure()
        plt.close(_warmup_fig)

        with open({df_pickle_path!r}, "rb") as _f:
            df = pd.read_pickle(_f)
    """)

    return header + "\n\n" + guard + "\n\n" + user_code


def run_in_sandbox(
    code: str,
    df,
    output_dir: str = "./output",
    timeout: int = 15,
) -> Dict[str, Any]:
    """
    Execute `code` against `df` in an isolated subprocess.

    Returns:
        {
            "success": bool,
            "stdout": str,
            "stderr": str,
            "traceback": str | None,
        }
    """
    os.makedirs(output_dir, exist_ok=True)
    output_dir_abs = os.path.abspath(output_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        df_pickle_path = os.path.join(tmpdir, "df.pkl")
        script_path = os.path.join(tmpdir, "generated_code.py")

        df.to_pickle(df_pickle_path)

        full_script = _build_run_script(code, df_pickle_path, output_dir_abs)
        with open(script_path, "w") as f:
            f.write(full_script)

        try:
            # start_new_session=True puts the child in its own process
            # group so the whole tree can be killed on timeout.
            proc = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                cwd=tmpdir,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
                success = proc.returncode == 0
            except subprocess.TimeoutExpired:
                _kill_process_group(proc)
                stdout, stderr = proc.communicate()
                return {
                    "success": False,
                    "stdout": stdout,
                    "stderr": stderr,
                    "traceback": f"TimeoutError: code did not finish within {timeout}s",
                }
        except Exception as e:  # noqa: BLE001 - want to surface any launch failure
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "traceback": str(e),
            }

    return {
        "success": success,
        "stdout": stdout,
        "stderr": stderr,
        "traceback": stderr if not success else None,
    }


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Kill the entire process group started by the subprocess, not just the parent."""
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


if __name__ == "__main__":
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    good_code = "print('sum of a:', df['a'].sum())"
    print("GOOD CODE RESULT:", run_in_sandbox(good_code, df))

    bad_import_code = "import os\nos.system('echo pwned')"
    print("BLOCKED IMPORT RESULT:", run_in_sandbox(bad_import_code, df))

    error_code = "print(df['does_not_exist'].sum())"
    print("ERROR RESULT:", run_in_sandbox(error_code, df))

    timeout_code = "while True: pass"
    print("TIMEOUT RESULT:", run_in_sandbox(timeout_code, df, timeout=3))
