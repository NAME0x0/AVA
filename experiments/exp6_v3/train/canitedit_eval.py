"""CanItEdit probe — the ON-TARGET eval: instruction-guided code editing.

Why (LiveCodeBench told us v3 is behind the donor on competitive stdin, but that
is OFF our training distribution — the mix is CommitPackFT edits + OpenCodeReasoning
reasoning, not Codeforces). CanItEdit measures the skill we actually train for:
given an existing program + a natural-language change request, produce the edited
program. Execution-scored, no Docker → fits the training-side sandbox, same shape as
`livecodebench_eval.py`. This is the scoreboard that decides whether the SFT helps.

Dataset (nuprl/CanItEdit, verified 2026-07-18, data/test-00000-of-00001.parquet,
105 rows):
  before                  = original program (str)
  after                   = gold edited program (str)  [canonical, passes `tests`]
  tests                   = assert block, references top-level names in the program
  instruction_descriptive = detailed change request
  instruction_lazy        = terse change request (harder)
  taxonomy                = {change_kind, libraries, topic}

Protocol: model gets `before` + instruction, must return the COMPLETE edited program
(the standard CanItEdit contract — tests reference module-level names, so a partial
answer legitimately fails instruction-following). Pass@1 = `edited + tests` exits 0.
Gold `after` validated to pass its own tests through our sandbox (20/20) before this
harness was trusted.
"""
from __future__ import annotations

from typing import Any

from .c1_eval import _extract_code, _progress, _set_postfix, generate
from .sandbox_exec import check_solution

_DATASET = "nuprl/CanItEdit"
_PARQUET = "data/test-00000-of-00001.parquet"


def _build_prompt(before: str, instruction: str) -> str:
    return (
        "You are editing an existing Python program. Apply the requested change "
        "and reply with the COMPLETE edited program in a single ```python code "
        "block — not a diff, not only the changed part.\n\n"
        "Current program:\n```python\n"
        f"{before}\n```\n\n"
        f"Change to make:\n{instruction}"
    )


def _kind(row: dict) -> str:
    tax = row.get("taxonomy")
    if isinstance(tax, dict):
        return str(tax.get("change_kind", "?"))
    return "?"


def _load_rows(limit: int | None) -> list[dict]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(_DATASET, _PARQUET, repo_type="dataset")
    df = pd.read_parquet(path)
    rows = [r for _, r in df.iterrows()]
    if limit:
        rows = rows[:limit]
    return rows


def run_canitedit(
    model: Any,
    tokenizer: Any,
    limit: int | None = None,          # None = all 105
    instruction: str = "descriptive",  # "descriptive" (default) | "lazy" (harder)
    thinking: bool = False,
    timeout_s: float = 15.0,
    max_new_tokens: int = 1400,
) -> dict:
    """Greedy pass@1 on CanItEdit. Returns report dict. Raises on empty selection."""
    if instruction not in ("descriptive", "lazy"):
        raise ValueError(f"instruction must be 'descriptive' or 'lazy', got {instruction!r}")
    rows = _load_rows(limit)
    if not rows:
        raise RuntimeError(f"canitedit: 0 rows loaded (limit={limit}) — check the dataset.")

    instr_col = f"instruction_{instruction}"
    per_task: dict[str, bool] = {}
    by_kind: dict[str, list[bool]] = {}
    bar = _progress(rows, desc=f"canitedit({instruction})", total=len(rows))
    for r in bar:
        prompt = _build_prompt(r["before"], r[instr_col])
        gen = generate(model, tokenizer, prompt, thinking=thinking, max_new_tokens=max_new_tokens)
        edited = _extract_code(gen)
        # tests reference module-level names -> run the full edited program + asserts
        ok = check_solution(edited, r["tests"], timeout_s=timeout_s).ok
        per_task[r["full_name"]] = ok
        by_kind.setdefault(_kind(r), []).append(ok)
        _set_postfix(bar, pass_rate=f"{100.0 * sum(per_task.values()) / len(per_task):.1f}%")

    score = 100.0 * sum(per_task.values()) / max(len(per_task), 1)
    return {
        "score": round(score, 2),
        "n": len(per_task),
        "instruction": instruction,
        "by_change_kind": {k: round(100.0 * sum(v) / len(v), 1) for k, v in by_kind.items()},
        "per_task": per_task,
    }


def edit_compare(
    ckpt_repo: str,
    donor: str,
    config_path: str,
    limit: int | None = None,
    instruction: str = "descriptive",
    thinking: bool = False,
) -> dict:
    """Donor vs trained checkpoint on the SAME CanItEdit problems, one model load.

    Same trick as `livecodebench_eval.lcb_compare`: a zero-init LoRA (lora_B=0) is
    numerically the donor, so eval adapters-on-but-zero (= donor), then resume the
    trained adapters (= AVA v3) and eval the identical problems. This is the
    ON-TARGET decision: v3 > donor here => the SFT is working (LiveCodeBench -10pp
    was off-target tax); v3 <= donor here => the mix/method needs revisiting before
    more compute.
    """
    import json

    import torch
    from scripts.checkpoint_sync import CheckpointSync

    from .hw_profile import detect_profile
    from .sft import SFTConfig, _build_qlora_model

    cfg = SFTConfig.from_yaml(config_path)
    cfg.ckpt_repo, cfg.donor = ckpt_repo, donor
    cfg.use_dora = False   # step-2167 preview predates the DoRA switch (plain LoRA)
    prof = detect_profile()
    dtype = torch.bfloat16 if prof.compute_dtype == "bfloat16" else torch.float16

    model, tok = _build_qlora_model(cfg, dtype)
    model.eval()
    print(f"[edit] evaluating DONOR (adapters zeroed) on CanItEdit ({instruction})...")
    donor_rep = run_canitedit(model, tok, limit=limit, instruction=instruction, thinking=thinking)
    print(f"[edit] DONOR: {donor_rep['score']}%  (n={donor_rep['n']})  {donor_rep['by_change_kind']}")

    step = CheckpointSync(ckpt_repo, phase="C5", trainable_only=True).resume(model) - 1
    print(f"[edit] evaluating AVA v3 @ step {step} on the SAME problems...")
    v3_rep = run_canitedit(model, tok, limit=limit, instruction=instruction, thinking=thinking)
    print(f"[edit] AVA v3 @{step}: {v3_rep['score']}%  (n={v3_rep['n']})  {v3_rep['by_change_kind']}")

    delta = round(v3_rep["score"] - donor_rep["score"], 2)
    print(f"\n[edit] === DELTA (v3 - donor) = {delta:+.2f} pp on the same {donor_rep['n']} "
          f"CanItEdit problems (ON-TARGET: instruction-guided editing) ===")
    out = {"donor": donor_rep, "ava_v3": v3_rep, "delta": delta, "step": step,
           "limit": limit, "instruction": instruction, "thinking": thinking}
    with open("edit_compare.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    try:
        from huggingface_hub import HfApi

        HfApi().upload_file(path_or_fileobj="edit_compare.json",
                            path_in_repo=f"reports/edit_compare_step{step}.json",
                            repo_id=ckpt_repo)
        print(f"[edit] report pushed -> {ckpt_repo}/reports/edit_compare_step{step}.json")
    except Exception as err:  # noqa: BLE001
        print(f"[edit] report push failed (non-fatal): {err!r}")
    return out
