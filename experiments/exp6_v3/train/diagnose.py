"""Diagnostic: dump actual code generations from a trained checkpoint.

A probe score says code regressed; it does not say WHY. This loads the latest
C5 checkpoint and prints, for failed HumanEval+ tasks, the raw generation +
extracted code + pass/fail — so you can SEE whether the model is:

  (a) emitting verbose reasoning prose (OpenCodeReasoning style) that either
      truncates before the code or makes _extract_code grab an intermediate
      snippet  -> format/extraction fix, and a data-style finding;
  (b) wrong output format entirely  -> prompt/template mismatch;
  (c) genuinely producing wrong code -> capability regression (LR too high /
      mixture wrong / over-training).

Run on a spare session; read-only on checkpoints.
"""
from __future__ import annotations


def dump_code_generations(
    ckpt_repo: str,
    donor: str,
    config_path: str,
    n: int = 6,
    only_failures: bool = True,
    max_new_tokens: int = 1024,
) -> None:
    import torch
    from datasets import load_dataset
    from scripts.checkpoint_sync import CheckpointSync

    from .c1_eval import _extract_code, generate
    from .hw_profile import detect_profile
    from .sandbox_exec import check_solution
    from .sft import SFTConfig, _build_qlora_model

    cfg = SFTConfig.from_yaml(config_path)
    cfg.ckpt_repo, cfg.donor = ckpt_repo, donor
    prof = detect_profile()
    dtype = torch.bfloat16 if prof.compute_dtype == "bfloat16" else torch.float16
    model, tok = _build_qlora_model(cfg, dtype)
    step = CheckpointSync(ckpt_repo, phase="C5", trainable_only=True).resume(model) - 1
    model.eval()

    ds = load_dataset("evalplus/humanevalplus", split="test").select(range(n * 4))
    shown = 0
    for ex in ds:
        if shown >= n:
            break
        prompt_code = ex["prompt"]
        instruction = (
            "Complete the following Python function. Reply with a single "
            f"```python code block containing the full function.\n\n```python\n{prompt_code}\n```"
        )
        gen = generate(model, tok, instruction, thinking=False, max_new_tokens=max_new_tokens)
        solution = _extract_code(gen)
        test_code = ex.get("test", "")
        entry = ex.get("entry_point", "")
        harness = solution + "\n\n" + test_code
        if "def check(" in test_code and entry:
            harness += f"\n\ncheck({entry})\n"
        ok = check_solution("", harness, timeout_s=15).ok
        if only_failures and ok:
            continue
        shown += 1
        n_blocks = gen.count("```python")
        print("=" * 72)
        print(f"TASK {ex['task_id']}  pass={ok}  gen={len(gen)} chars  "
              f"python-blocks={n_blocks}  truncated={not gen.rstrip().endswith('```') and len(gen) > max_new_tokens}")
        print("--- RAW GENERATION (first 1500 chars) ---")
        print(gen[:1500])
        print("--- EXTRACTED CODE (what got scored) ---")
        print(solution[:700])
    print(f"\n[diagnose] step {step}: showed {shown} tasks. "
          "Many python-blocks or long prose before code => extraction/style issue.")
