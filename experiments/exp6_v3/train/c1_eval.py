"""C1 donor baseline eval — sets the bar every later phase is gated against.

Coverage (CODE_PIVOT section 7 subset that runs on free hardware today):
  - HumanEval+ and MBPP+ (EvalPlus datasets from the HF Hub), pass@1 greedy,
    scored by real execution in the training-side sandbox.
  - Sanity floors: ARC-Easy + MMLU (stratified subsample), generative
    letter-matching protocol (log-prob MC is unreliable for chat models).
  - Dual mode: thinking and non-thinking (Qwen3.5 chat-template flag) —
    REVIEW section on think/non-think split demands both numbers.

Output: JSON report {benchmark -> {mode -> {score, n, per_task}}} written
locally and (optionally) pushed to the checkpoint Hub repo. gate.py consumes
these reports.

Full matrix (LiveCodeBench, MultiPL-E, SWE-bench via mini-swe-agent, CRUXEval,
BigCodeBench) intentionally lives OUTSIDE this file — standard third-party
harnesses, run per their own docs for comparability (REVIEW section 11.1).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from .sandbox_exec import check_solution

# --------------------------------------------------------------------------- generation


def generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    thinking: bool,
    max_new_tokens: int = 1024,
) -> str:
    import torch

    kwargs: dict[str, Any] = {}
    try:  # Qwen3-family templates accept enable_thinking; older ones raise
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=thinking,
            return_tensors="pt",
        )
    except TypeError:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
    ids = ids.to(model.device)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens if not thinking else max_new_tokens * 4,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            **kwargs,
        )
    text = tokenizer.decode(out[0][ids.shape[1] :], skip_special_tokens=True)
    # strip thinking block if present — score the answer, not the monologue
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# --------------------------------------------------------------------------- code benchmarks


def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


def eval_evalplus(
    model: Any,
    tokenizer: Any,
    dataset_id: str,          # "evalplus/humanevalplus" | "evalplus/mbppplus"
    thinking: bool,
    limit: int | None = None,
    timeout_s: float = 15.0,
) -> dict:
    """Greedy pass@1 with plus-tests execution. Returns report dict."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    per_task: dict[str, bool] = {}
    for ex in ds:
        task_id = str(ex.get("task_id"))
        prompt_code = ex["prompt"]
        instruction = (
            "Complete the following Python function. Reply with a single "
            f"```python code block containing the full function.\n\n```python\n{prompt_code}\n```"
        )
        gen = generate(model, tokenizer, instruction, thinking=thinking)
        solution = _extract_code(gen)
        # EvalPlus schema: `test` defines check(candidate); call it when present,
        # else the test block is a bare assertion suite.
        test_code = ex.get("test", "")
        entry = ex.get("entry_point", "")
        harness = solution + "\n\n" + test_code
        if "def check(" in test_code and entry:
            harness += f"\n\ncheck({entry})\n"
        res = check_solution("", harness, timeout_s=timeout_s)
        per_task[task_id] = res.ok
    score = 100.0 * sum(per_task.values()) / max(len(per_task), 1)
    return {"score": round(score, 2), "n": len(per_task), "per_task": per_task}


# --------------------------------------------------------------------------- MC floors (ARC-E, MMLU)

_LETTER_RE = re.compile(r"\b([ABCD])\b")


def _mc_prompt(question: str, choices: list[str]) -> str:
    letters = "ABCD"
    lines = [f"{letters[i]}. {c}" for i, c in enumerate(choices[:4])]
    return (
        f"{question}\n\n" + "\n".join(lines) + "\n\nAnswer with the single letter of the correct option."
    )


def _mc_score(gen: str, correct_letter: str) -> bool:
    m = _LETTER_RE.search(gen.strip()[:80])
    return bool(m and m.group(1) == correct_letter)


def eval_arc_easy(model, tokenizer, thinking: bool, limit: int = 500) -> dict:
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test").select(range(limit))
    correct = 0
    n = 0
    for ex in ds:
        choices, labels = ex["choices"]["text"], ex["choices"]["label"]
        if len(choices) != 4:
            continue
        # normalize labels (may be "1".."4" or "A".."D") to A..D
        key = dict(zip(labels, "ABCD", strict=False))
        gold = key.get(ex["answerKey"])
        if gold is None:
            continue
        gen = generate(model, tokenizer, _mc_prompt(ex["question"], choices), thinking, 64)
        correct += _mc_score(gen, gold)
        n += 1
    return {"score": round(100.0 * correct / max(n, 1), 2), "n": n}


def eval_mmlu(model, tokenizer, thinking: bool, limit: int = 1000, seed: int = 7) -> dict:
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=seed).select(range(limit))
    correct = 0
    for ex in ds:
        gold = "ABCD"[ex["answer"]]
        gen = generate(model, tokenizer, _mc_prompt(ex["question"], ex["choices"]), thinking, 64)
        correct += _mc_score(gen, gold)
    return {"score": round(100.0 * correct / max(len(ds), 1), 2), "n": len(ds)}


# --------------------------------------------------------------------------- orchestrator


def run_c1(
    model: Any,
    tokenizer: Any,
    out_path: str | Path,
    modes: tuple[bool, ...] = (False, True),   # non-think first (cheap)
    code_limit: int | None = None,             # None = full sets
    floor_limits: tuple[int, int] = (500, 1000),
    hub_repo: str | None = None,
) -> dict:
    report: dict[str, Any] = {"meta": {"unix": time.time()}}
    for thinking in modes:
        mode = "thinking" if thinking else "non_thinking"
        report[mode] = {
            "humaneval_plus": eval_evalplus(
                model, tokenizer, "evalplus/humanevalplus", thinking, code_limit
            ),
            "mbpp_plus": eval_evalplus(
                model, tokenizer, "evalplus/mbppplus", thinking, code_limit
            ),
            "arc_easy": eval_arc_easy(model, tokenizer, thinking, floor_limits[0]),
            "mmlu": eval_mmlu(model, tokenizer, thinking, floor_limits[1]),
        }
        Path(out_path).write_text(json.dumps(report, indent=2))  # save after each mode
    if hub_repo:
        from huggingface_hub import HfApi

        HfApi().upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo=f"reports/{Path(out_path).name}",
            repo_id=hub_repo,
        )
    return report
