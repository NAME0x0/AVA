"""Subprocess-isolated Python execution for eval scoring and verification.

Not the hardened MCP sandbox (that is `mcp/sandbox/`, ships with the product).
This is the *training-side* executor: benchmark test harnesses and self-play
verification, where the code being run is benchmark/test code. Isolation
level: separate process, wall-clock timeout, no stdin, cwd in a temp dir,
process tree killed on timeout. Network is NOT blocked — do not run
adversarial code through this; use the MCP sandbox for that.

Cross-platform: Windows (laptop) + Linux (Colab/Kaggle).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass


@dataclass
class ExecResult:
    ok: bool          # process exited 0 within the timeout
    timeout: bool
    returncode: int
    stdout: str
    stderr: str


def run_python(code: str, timeout_s: float = 10.0) -> ExecResult:
    """Run `code` in a fresh Python subprocess. Never raises on user-code failure."""
    with tempfile.TemporaryDirectory(prefix="ava_exec_") as tmp:
        # Write to a file rather than `python -c <code>`: EvalPlus tests inline
        # thousands of cases and blow the OS command-line length limit
        # (Windows CreateProcess ~32K -> "filename too long"; also a laptop-lane
        # portability wart). No -I: tests import numpy, which -I hides where
        # packages live in the user site. Isolation = process + tmp cwd +
        # timeout; hostile code belongs in the MCP sandbox, not this runner.
        script = os.path.join(tmp, "_run.py")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(code)
        try:
            proc = subprocess.run(
                [sys.executable, script],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                stdin=subprocess.DEVNULL,
            )
            return ExecResult(
                ok=proc.returncode == 0,
                timeout=False,
                returncode=proc.returncode,
                stdout=proc.stdout[-8000:],
                stderr=proc.stderr[-8000:],
            )
        except subprocess.TimeoutExpired as err:
            return ExecResult(
                ok=False,
                timeout=True,
                returncode=-1,
                stdout=(err.stdout or b"" if isinstance(err.stdout, bytes) else (err.stdout or ""))[-8000:]
                if err.stdout else "",
                stderr="TIMEOUT",
            )
        except OSError as err:  # spawn failure — environment problem, surface it
            return ExecResult(False, False, -2, "", f"spawn failed: {err!r}")


def check_solution(solution_code: str, test_code: str, timeout_s: float = 10.0) -> ExecResult:
    """Concatenate candidate solution + tests, run isolated. Pass = exit 0."""
    return run_python(solution_code + "\n\n" + test_code, timeout_s=timeout_s)
