"""Phase-gate regression check (CODE_PIVOT section 8).

Rule: a phase checkpoint may not regress the donor coding baseline by more
than `max_regression_pp` on any benchmark, and sanity floors must hold
(ARC-E >= 75, MMLU >= 45 — catastrophic-forgetting alarms).

Consumes the JSON reports written by c1_eval.run_c1: baseline (donor at C1)
vs candidate (current checkpoint, same eval code, same limits, same mode).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

FLOORS = {"arc_easy": 75.0, "mmlu": 45.0}
CODE_BENCHES = ("humaneval_plus", "mbpp_plus")


@dataclass
class GateResult:
    passed: bool
    failures: list[str]
    deltas: dict[str, float]


def check_gate(
    baseline_report: dict | str | Path,
    candidate_report: dict | str | Path,
    mode: str = "non_thinking",
    max_regression_pp: float = 2.0,
) -> GateResult:
    base = _load(baseline_report).get(mode, {})
    cand = _load(candidate_report).get(mode, {})
    failures: list[str] = []
    deltas: dict[str, float] = {}

    for bench in CODE_BENCHES:
        b, c = base.get(bench, {}).get("score"), cand.get(bench, {}).get("score")
        if b is None or c is None:
            failures.append(f"{bench}: missing score (baseline={b}, candidate={c})")
            continue
        deltas[bench] = round(c - b, 2)
        if c < b - max_regression_pp:
            failures.append(f"{bench}: {c} < baseline {b} - {max_regression_pp}pp")

    for bench, floor in FLOORS.items():
        c = cand.get(bench, {}).get("score")
        if c is None:
            failures.append(f"{bench}: missing floor score")
        elif c < floor:
            failures.append(f"{bench}: {c} below floor {floor}")
        if c is not None:
            deltas[bench] = round(c - base.get(bench, {}).get("score", c), 2)

    return GateResult(passed=not failures, failures=failures, deltas=deltas)


def _load(report: dict | str | Path) -> dict:
    if isinstance(report, dict):
        return report
    return json.loads(Path(report).read_text(encoding="utf-8"))
