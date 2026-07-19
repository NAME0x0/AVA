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

RESUMABLE (added 2026-07-18): each problem's pass/fail is checkpointed to the Hub
after every `save_every` items (same fabric as training — Hub is truth). A Colab
disconnect loses at most one problem; a restart skips everything already done. The
donor half caches under its own key so it is NEVER re-run once complete — the pain
was every restart redoing the 71-min donor pass before even reaching v3.
"""
from __future__ import annotations

import json
from pathlib import Path
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


def _classify(edited: str, tests: str, timeout_s: float) -> tuple[bool, str]:
    """Split failures so the score is interpretable (iron law): a truncated/
    incomplete program (SyntaxError) is a MEASUREMENT artifact (token budget /
    verbosity), whereas a program that runs but fails the asserts is a REAL wrong
    edit. Mixing them hides which one a low score actually means."""
    if not edited.strip():
        return False, "empty"          # no code block extracted at all
    try:
        compile(edited, "<edit>", "exec")
    except SyntaxError:
        return False, "syntax"         # incomplete / truncated -> artifact-ish
    res = check_solution(edited, tests, timeout_s=timeout_s)
    if res.ok:
        return True, "pass"
    return False, "timeout" if res.timeout else "tests"   # "tests" = ran, wrong edit


def _report(per_task: dict, by_kind: dict, reasons: dict, instruction: str) -> dict:
    """Report from per-task bools + per-kind bool lists + per-task failure reasons.

    `by_reason` decomposes the failures: pass / syntax (truncated-incomplete) /
    tests (complete but wrong) / timeout / empty.
    """
    by_reason: dict[str, int] = {}
    for rz in reasons.values():
        by_reason[rz] = by_reason.get(rz, 0) + 1
    return {
        "score": round(100.0 * sum(per_task.values()) / max(len(per_task), 1), 2),
        "n": len(per_task),
        "instruction": instruction,
        "by_change_kind": {k: round(100.0 * sum(v) / len(v), 1) for k, v in by_kind.items()},
        "by_reason": by_reason,
        "per_task": per_task,
        "reasons": reasons,
    }


def _seed(resume_key: str | None, hub_repo: str | None) -> dict:
    """Resume-safe (mirrors c1_eval._seed_report): prior per-task results from the
    local file first (freshest in-session), else the Hub copy (cross-session)."""
    if not resume_key:
        return {}
    local = Path(f"{resume_key}.json")
    if local.exists():
        return json.loads(local.read_text(encoding="utf-8"))
    if hub_repo:
        try:
            from huggingface_hub import hf_hub_download

            got = hf_hub_download(hub_repo, f"reports/{resume_key}.json")
            return json.loads(Path(got).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - no prior report == fresh start
            pass
    return {}


def _persist(resume_key: str | None, hub_repo: str | None, report: dict) -> None:
    """Write local always; push to Hub best-effort (a transient Hub error must not
    kill a 70-min run — local mirror still lets the session resume)."""
    if not resume_key:
        return
    Path(f"{resume_key}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if hub_repo:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(hub_repo, private=True, exist_ok=True)
            api.upload_file(path_or_fileobj=f"{resume_key}.json",
                            path_in_repo=f"reports/{resume_key}.json", repo_id=hub_repo)
        except Exception as err:  # noqa: BLE001
            print(f"[edit] checkpoint push failed (non-fatal, local saved): {err!r}")


def run_canitedit(
    model: Any,
    tokenizer: Any,
    limit: int | None = None,          # None = all 105
    instruction: str = "descriptive",  # "descriptive" (default) | "lazy" (harder)
    thinking: bool = False,
    timeout_s: float = 15.0,
    max_new_tokens: int = 2560,        # full-program rewrites of 50-80-line files
    hub_repo: str | None = None,       # push per-problem checkpoints here
    resume_key: str | None = None,     # report basename; enables resume + skip-done
    save_every: int = 5,
) -> dict:
    """Greedy pass@1 on CanItEdit, resumable per problem. Raises on empty selection."""
    if instruction not in ("descriptive", "lazy"):
        raise ValueError(f"instruction must be 'descriptive' or 'lazy', got {instruction!r}")
    rows = _load_rows(limit)
    if not rows:
        raise RuntimeError(f"canitedit: 0 rows loaded (limit={limit}) — check the dataset.")

    instr_col = f"instruction_{instruction}"
    prior = _seed(resume_key, hub_repo)
    per_task: dict[str, bool] = {k: bool(v) for k, v in prior.get("per_task", {}).items()}
    reasons: dict[str, str] = dict(prior.get("reasons", {}))
    by_kind: dict[str, list[bool]] = {}
    # rebuild per-kind tallies (+ back-fill reasons for legacy reports) for done rows
    for r in rows:
        name = r["full_name"]
        if name in per_task:
            by_kind.setdefault(_kind(r), []).append(per_task[name])
            reasons.setdefault(name, "pass" if per_task[name] else "legacy_fail")
    todo = [r for r in rows if r["full_name"] not in per_task]
    if per_task:
        print(f"[edit] resuming {resume_key}: {len(per_task)} done, {len(todo)} to go")
    if not todo:                       # fully cached — skip the model entirely
        return _report(per_task, by_kind, reasons, instruction)

    bar = _progress(todo, desc=f"canitedit({instruction})", total=len(todo))
    for i, r in enumerate(bar):
        prompt = _build_prompt(r["before"], r[instr_col])
        gen = generate(model, tokenizer, prompt, thinking=thinking, max_new_tokens=max_new_tokens)
        ok, reason = _classify(_extract_code(gen), r["tests"], timeout_s)
        per_task[r["full_name"]] = ok
        reasons[r["full_name"]] = reason
        by_kind.setdefault(_kind(r), []).append(ok)
        _set_postfix(bar, pass_rate=f"{100.0 * sum(per_task.values()) / len(per_task):.1f}%")
        if (i + 1) % save_every == 0:
            _persist(resume_key, hub_repo, _report(per_task, by_kind, reasons, instruction))

    report = _report(per_task, by_kind, reasons, instruction)
    _persist(resume_key, hub_repo, report)   # final flush
    return report


def edit_compare(
    ckpt_repo: str,
    donor: str,
    config_path: str,
    limit: int | None = None,
    instruction: str = "descriptive",
    thinking: bool = False,
    max_new_tokens: int = 2560,
) -> dict:
    """Donor vs trained checkpoint on the SAME CanItEdit problems, one model load.

    Same trick as `livecodebench_eval.lcb_compare`: a zero-init LoRA (lora_B=0) is
    numerically the donor, so eval adapters-on-but-zero (= donor), then resume the
    trained adapters (= AVA v3) and eval the identical problems. This is the
    ON-TARGET decision: v3 > donor here => the SFT is working (LiveCodeBench -10pp
    was off-target tax); v3 <= donor here => the mix/method needs revisiting before
    more compute.

    Resumable: donor and v3 each checkpoint per problem under their own key, so a
    disconnect resumes both and the donor half is never recomputed once complete.
    """
    import torch
    from scripts.checkpoint_sync import CheckpointSync

    from .hw_profile import detect_profile
    from .sft import SFTConfig, _build_qlora_model

    cfg = SFTConfig.from_yaml(config_path)
    cfg.ckpt_repo, cfg.donor = ckpt_repo, donor
    cfg.use_dora = False   # step-2167 preview predates the DoRA switch (plain LoRA)
    prof = detect_profile()
    dtype = torch.bfloat16 if prof.compute_dtype == "bfloat16" else torch.float16
    n_tag = limit or 105

    model, tok = _build_qlora_model(cfg, dtype)
    model.eval()
    donor_key = f"canitedit_donor_{instruction}_n{n_tag}_t{max_new_tokens}"
    print(f"[edit] evaluating DONOR (adapters zeroed) on CanItEdit ({instruction})...")
    donor_rep = run_canitedit(model, tok, limit=limit, instruction=instruction,
                              thinking=thinking, max_new_tokens=max_new_tokens,
                              hub_repo=ckpt_repo, resume_key=donor_key)
    print(f"[edit] DONOR: {donor_rep['score']}%  (n={donor_rep['n']})  "
          f"{donor_rep['by_change_kind']}  reasons={donor_rep['by_reason']}")

    step = CheckpointSync(ckpt_repo, phase="C5", trainable_only=True).resume(model) - 1
    v3_key = f"canitedit_v3_step{step}_{instruction}_n{n_tag}_t{max_new_tokens}"
    print(f"[edit] evaluating AVA v3 @ step {step} on the SAME problems...")
    v3_rep = run_canitedit(model, tok, limit=limit, instruction=instruction,
                           thinking=thinking, max_new_tokens=max_new_tokens,
                           hub_repo=ckpt_repo, resume_key=v3_key)
    print(f"[edit] AVA v3 @{step}: {v3_rep['score']}%  (n={v3_rep['n']})  "
          f"{v3_rep['by_change_kind']}  reasons={v3_rep['by_reason']}")

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
