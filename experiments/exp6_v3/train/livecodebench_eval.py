"""LiveCodeBench probe — a HARDER, contamination-resistant execution eval.

Why (PLAN_2026-07-15_v31 s5): HumanEval+/MBPP+ are saturated vs the donor, so
they can only show "didn't break it". LiveCodeBench is competitive-programming
(Codeforces/LeetCode), time-windowed (pick a release AFTER the donor's cutoff to
kill contamination), and has real headroom to show specialization. Self-contained
execution — no Docker, unlike SWE-bench — so it fits our sandbox.

Dataset shape (verified 2026-07-18, livecodebench/code_generation_lite/test.jsonl):
  question_content, starter_code, difficulty, contest_date, platform,
  public_test_cases  = JSON  list[{input, output, testtype}]
  private_test_cases = base64 -> zlib -> PICKLE list[{input, output, testtype}]
  testtype in {"stdin", "functional"}.

This first version scores **stdin-type** problems only (the majority; robust:
feed input, compare stdout). Functional/LeetCode problems are reported as skipped,
not scored wrong — honest n. A solution passes iff it passes EVERY public+private
test (pass@1). Time-window filter via `min_date` keeps it contamination-clean.
"""
from __future__ import annotations

import base64
import json
import pickle
import zlib
from typing import Any

from .c1_eval import _extract_code, _progress, _set_postfix, generate
from .sandbox_exec import run_python


def _parse_field(field: Any) -> list:
    """A test-case field -> list. Handles JSON, double-JSON, and the private
    base64->zlib->pickle encoding. Field forms found in the wild (verified
    2026-07-18): plain JSON list, a JSON *string* wrapping the list
    (double-encoded), and binary pickle. Returns [] on anything unusable."""
    if not field:
        return []
    obj: Any = field
    for _ in range(2):  # unwrap up to one double-encoding layer
        if isinstance(obj, str):
            try:
                obj = json.loads(obj)
            except (json.JSONDecodeError, UnicodeDecodeError):
                try:
                    obj = pickle.loads(zlib.decompress(base64.b64decode(obj)))  # noqa: S301
                except Exception:  # noqa: BLE001 - unusable field -> skip
                    return []
        else:
            break
    return obj if isinstance(obj, list) else []


def _decode_tests(row: dict) -> list[dict]:
    """Public + private test cases as a clean list of {input, output, testtype}
    dicts. Non-dict / malformed entries are dropped (some rows mix formats)."""
    tests: list[dict] = []
    for field in (row.get("public_test_cases"), row.get("private_test_cases")):
        for t in _parse_field(field):
            if isinstance(t, dict) and "input" in t and "output" in t:
                tests.append(t)
    return tests


def _norm(s: str) -> str:
    """Competitive-judge output comparison: trailing whitespace per line ignored."""
    return "\n".join(line.rstrip() for line in s.strip().splitlines())


def _score_stdin(solution: str, tests: list[dict], timeout_s: float) -> bool:
    for t in tests:
        res = run_python(solution, timeout_s=timeout_s, stdin_data=t["input"])
        if not res.ok or _norm(res.stdout) != _norm(t["output"]):
            return False
    return True


def _build_prompt(row: dict) -> str:
    starter = row.get("starter_code") or ""
    body = (
        f"Solve this competitive programming problem. Read input from standard "
        f"input and print the answer to standard output. Reply with a single "
        f"```python code block containing the complete program.\n\n"
        f"{row['question_content']}"
    )
    if starter.strip():
        body += f"\n\nStarter code:\n```python\n{starter}\n```"
    return body


def run_livecodebench(
    model: Any,
    tokenizer: Any,
    dataset_file: str = "test.jsonl",
    limit: int | None = 100,
    min_date: str | None = None,   # e.g. "2026-03-01" -> after donor cutoff
    thinking: bool = False,
    timeout_s: float = 12.0,
    max_new_tokens: int = 1400,
) -> dict:
    """Greedy pass@1 on stdin-type LiveCodeBench problems. Returns report dict."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "livecodebench/code_generation_lite", dataset_file, repo_type="dataset"
    )
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if min_date and r.get("contest_date", "") < min_date:
                continue
            tests = _decode_tests(r)
            if tests and all(t.get("testtype") == "stdin" for t in tests):
                r["_tests"] = tests
                rows.append(r)
            if limit and len(rows) >= limit:
                break

    per_task: dict[str, bool] = {}
    by_diff: dict[str, list[bool]] = {}
    bar = _progress(rows, desc="livecodebench(stdin)", total=len(rows))
    for r in bar:
        gen = generate(model, tokenizer, _build_prompt(r), thinking=thinking,
                       max_new_tokens=max_new_tokens)
        solution = _extract_code(gen)
        ok = _score_stdin(solution, r["_tests"], timeout_s)
        per_task[r["question_id"]] = ok
        by_diff.setdefault(r.get("difficulty", "?"), []).append(ok)
        _set_postfix(bar, pass_rate=f"{100.0 * sum(per_task.values()) / len(per_task):.1f}%")

    score = 100.0 * sum(per_task.values()) / max(len(per_task), 1)
    return {
        "score": round(score, 2),
        "n": len(per_task),
        "by_difficulty": {d: round(100.0 * sum(v) / len(v), 1) for d, v in by_diff.items()},
        "per_task": per_task,
    }
