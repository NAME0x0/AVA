"""Sandboxed Python code execution for HumanEval+/MBPP+ evaluation."""
from __future__ import annotations

import json
import multiprocessing as mp
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


def _run_in_subprocess(code: str, timeout: float) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        if proc.returncode == 0:
            return True, proc.stdout
        return False, proc.stderr or proc.stdout
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout}s"
    except Exception as e:
        return False, f"EXCEPTION: {e!r}"
    finally:
        Path(path).unlink(missing_ok=True)


def humaneval_check(generated: str, prompt: str, test: str,
                     entry_point: str, timeout: float = 5.0) -> tuple[bool, str]:
    full_prompt = prompt
    if generated.lstrip().startswith(("def ", "class ", "import ", "from ")):
        full_code = generated
    else:
        full_code = full_prompt + "\n" + generated
    runner = textwrap.dedent(f"""
        import sys, signal
        def _alarm(*_): raise TimeoutError('exec timeout')
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (4*1024*1024*1024,)*2)
        except Exception:
            pass
    """)
    code = runner + "\n" + full_code + "\n" + test + f"\ncheck({entry_point})\n"
    return _run_in_subprocess(code, timeout=timeout)


def mbpp_check(generated: str, tests: str,
                timeout: float = 5.0) -> tuple[bool, str]:
    if not generated.lstrip().startswith(("def ", "class ", "import ", "from ")):
        generated = "def _wrap():\n" + textwrap.indent(generated, "    ")
    code = generated + "\n" + tests + "\n"
    return _run_in_subprocess(code, timeout=timeout)
