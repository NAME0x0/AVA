"""Mid-training probe eval — is more training still buying anything?

Runs on a SECOND session (any GPU) while the trainer keeps going elsewhere:
loads donor + the latest C5 adapters from the Hub (read-only on checkpoints),
evaluates fast subsets, pushes the result to reports/probes/ (NOT the
C5_EVAL path — the autopilot must still run the full gate at the end), and
prints deltas vs the C1 donor baseline.

Decision guide printed at the end:
  code deltas still climbing between probes  -> keep training
  flat over ~2 probes (>=1K steps apart)     -> saturated; consider stopping
  negative vs donor                          -> something is wrong; stop, talk
"""
from __future__ import annotations

import json
from pathlib import Path


def run_probe(
    ckpt_repo: str,
    donor: str,
    config_path: str,
    code_limit: int = 30,
    floor_limits: tuple[int, int] = (100, 150),
) -> dict:
    import torch
    from huggingface_hub import hf_hub_download
    from scripts.checkpoint_sync import CheckpointSync

    from .c1_eval import run_c1
    from .hw_profile import detect_profile
    from .sft import SFTConfig, _build_qlora_model

    cfg = SFTConfig.from_yaml(config_path)
    cfg.ckpt_repo, cfg.donor = ckpt_repo, donor

    prof = detect_profile()
    dtype = torch.bfloat16 if prof.compute_dtype == "bfloat16" else torch.float16
    model, tokenizer = _build_qlora_model(cfg, dtype)
    step = CheckpointSync(ckpt_repo, phase="C5", trainable_only=True).resume(model) - 1
    model.eval()

    out = f"probe_step{step:08d}.json"
    report = run_c1(
        model, tokenizer, out_path=out,
        modes=(False,),                    # gate mode only — probes are cheap
        code_limit=code_limit,
        floor_limits=floor_limits,
        hub_repo=None,                     # manual push to probes/ below
    )
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(ckpt_repo, private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=out, path_in_repo=f"reports/probes/{out}", repo_id=ckpt_repo)

    baseline = json.loads(
        Path(hf_hub_download(ckpt_repo, "reports/c1_donor_baseline.json")).read_text()
    )["non_thinking"]
    probe = report["non_thinking"]
    print(f"\n[probe] checkpoint step {step} vs donor baseline "
          f"(subsets n={code_limit}/{floor_limits}):")
    for bench in ("humaneval_plus", "mbpp_plus", "arc_easy", "mmlu"):
        b, p = baseline[bench]["score"], probe[bench]["score"]
        # MATCHED comparison when per-task results exist (code benches):
        # probe subsets are the FIRST n tasks — comparing them against the
        # full-set score misleads (field lesson 2026-07-12: +9.0 "gain" on
        # HumanEval+ was a -10.0 regression on the matched 30 tasks).
        base_tasks = baseline[bench].get("per_task") or {}
        probe_tasks = probe[bench].get("per_task") or {}
        common = [t for t in probe_tasks if t in base_tasks]
        if common:
            bm = 100.0 * sum(bool(base_tasks[t]) for t in common) / len(common)
            pm = 100.0 * sum(bool(probe_tasks[t]) for t in common) / len(common)
            print(f"  {bench:16s} MATCHED n={len(common)}: donor {bm:6.2f} -> "
                  f"probe {pm:6.2f}  ({pm - bm:+.2f})")
        else:
            print(f"  {bench:16s} unmatched subset (indicative): donor(full) "
                  f"{b:6.2f} -> probe {p:6.2f}  ({p - b:+.2f})")
    print("[probe] guide (MATCHED code deltas only): climbing -> keep training | "
          "flat 2 probes -> saturated | negative -> stop and investigate")
    return report
