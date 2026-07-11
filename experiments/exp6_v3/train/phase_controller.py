"""Automatic phase selection from Hub state — the notebook's autopilot.

Reads the checkpoint repo and decides what this session should do, so a user
can open the notebook on any platform, press Run-all, and always advance the
pipeline (REVIEW_2026-07.md section 9 flow):

    C1       donor baseline eval — until reports/c1_donor_baseline.json has
             BOTH modes (the report saves per mode, so a killed session leaves
             a partial file; we resume with only the missing modes)
    C5       SFT shards — until checkpoints/C5/LATEST.json reaches total_steps
    C5_EVAL  eval the trained candidate + gate vs the C1 baseline
    DONE     nothing left this lane can do automatically

Hub access is injected (file_exists_fn / load_json_fn) so every branch is
CPU-testable offline; defaults wrap huggingface_hub.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

C1_REPORT = "reports/c1_donor_baseline.json"
C5_LATEST = "checkpoints/C5/LATEST.json"
C5_EVAL_REPORT = "reports/c5_candidate_eval.json"
_MODES = ("non_thinking", "thinking")
_REQUIRED_BENCHES = ("humaneval_plus", "mbpp_plus", "arc_easy", "mmlu")


@dataclass
class PhaseDecision:
    phase: str                 # "C1" | "C5" | "C5_EVAL" | "DONE"
    reason: str
    # C1 only: which eval modes still need running (thinking-flag values)
    modes_needed: tuple[bool, ...] = field(default=())


def _default_file_exists(repo_id: str, path: str) -> bool:
    from huggingface_hub import file_exists

    return file_exists(repo_id, path)


def _default_load_json(repo_id: str, path: str) -> dict:
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(repo_id, path), encoding="utf-8") as fh:
        return json.load(fh)


def decide_phase(
    ckpt_repo: str,
    total_steps: int,
    file_exists_fn: Callable[[str, str], bool] | None = None,
    load_json_fn: Callable[[str, str], dict] | None = None,
) -> PhaseDecision:
    exists = file_exists_fn or _default_file_exists
    load = load_json_fn or _default_load_json

    # -- C1: baseline report present and complete? --------------------------
    try:
        report_exists = exists(ckpt_repo, C1_REPORT)
    except Exception as err:  # repo not created yet -> first ever run
        return PhaseDecision("C1", f"checkpoint repo unreachable ({err!r}) — first run")
    if not report_exists:
        return PhaseDecision(
            "C1", "no donor baseline on Hub", modes_needed=(False, True)
        )
    report = load(ckpt_repo, C1_REPORT)
    # Gate-relevant mode is non_thinking ONLY (gate.py default) — C1 blocks
    # the pipeline solely on it. The thinking probe is informational: worth
    # having, never worth stalling training or burning paid compute for
    # (observed: full thinking eval = 20+ GPU-h with truncation artifacts).
    def _mode_done(m: str) -> bool:
        return all(b in report.get(m, {}) for b in _REQUIRED_BENCHES)

    if not _mode_done("non_thinking"):
        return PhaseDecision(
            "C1", "baseline partial — gate mode (non_thinking) incomplete",
            modes_needed=(False,),
        )
    if not _mode_done("thinking"):
        print("[phase] note: thinking probe incomplete — optional; backfill "
              "on free quota anytime with AVA_PHASE=C1")

    # -- C5: training progressed to total_steps? ----------------------------
    if not exists(ckpt_repo, C5_LATEST):
        return PhaseDecision("C5", "baseline complete — start SFT shards")
    step = int(load(ckpt_repo, C5_LATEST).get("step", 0))
    if step + 1 < total_steps:
        return PhaseDecision(
            "C5", f"resume SFT at step {step + 1}/{total_steps}"
        )

    # -- C5_EVAL: candidate evaluated + gated? -------------------------------
    if not exists(ckpt_repo, C5_EVAL_REPORT):
        return PhaseDecision(
            "C5_EVAL", f"SFT reached {step + 1} steps — evaluate candidate vs baseline"
        )

    return PhaseDecision(
        "DONE",
        "C1 + C5 + candidate eval all on Hub — next: review gate verdict, "
        "export merged weights (manual for now)",
    )


def describe(decision: PhaseDecision) -> str:
    return f"PHASE={decision.phase} — {decision.reason}"


# --------------------------------------------------------------------------- gate helper

def gate_candidate(
    ckpt_repo: str,
    load_json_fn: Callable[[str, str], dict] | None = None,
    mode: str = "non_thinking",
) -> Any:
    """Run the <=2pp gate on the two Hub reports. Returns gate.GateResult."""
    from .gate import check_gate

    load = load_json_fn or _default_load_json
    baseline = load(ckpt_repo, C1_REPORT)
    candidate = load(ckpt_repo, C5_EVAL_REPORT)
    return check_gate(baseline, candidate, mode=mode)
