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

# --------------------------------------------------------------------------- progress


def _progress(iterable: Any, desc: str, total: int | None = None) -> Any:
    """tqdm live bar (rate + ETA) when available; silent passthrough otherwise."""
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True)
    except Exception:  # noqa: BLE001 - progress is cosmetic, never fatal
        return iterable


def _set_postfix(bar: Any, **kv: Any) -> None:
    if hasattr(bar, "set_postfix"):
        bar.set_postfix(**kv)


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
            return_dict=False,
        )
    except TypeError:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False,
        )
    if hasattr(ids, "keys"):  # newer transformers return BatchEncoding anyway
        ids = ids["input_ids"]
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
    bar = _progress(ds, desc=dataset_id.split("/")[-1], total=len(ds))
    for ex in bar:
        task_id = str(ex.get("task_id"))
        prompt_code = ex["prompt"]
        entry = ex.get("entry_point", "")
        # MBPP+ prompts are natural-language descriptions whose tests call a
        # specific function name — without naming it, the model invents one
        # and every test dies with NameError (observed: 6% pass vs ~70% real).
        name_clause = f" The function MUST be named exactly `{entry}`." if entry else ""
        instruction = (
            "Complete the following Python task. Reply with a single "
            f"```python code block containing the full solution.{name_clause}"
            f"\n\n```python\n{prompt_code}\n```"
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
        _set_postfix(bar, pass_rate=f"{100.0 * sum(per_task.values()) / len(per_task):.1f}%")
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
    bar = _progress(ds, desc="arc_easy", total=len(ds))
    for ex in bar:
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
        _set_postfix(bar, acc=f"{100.0 * correct / n:.1f}%")
    return {"score": round(100.0 * correct / max(n, 1), 2), "n": n}


def eval_mmlu(model, tokenizer, thinking: bool, limit: int = 1000, seed: int = 7) -> dict:
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=seed).select(range(limit))
    correct = 0
    done = 0
    bar = _progress(ds, desc="mmlu", total=len(ds))
    for ex in bar:
        gold = "ABCD"[ex["answer"]]
        gen = generate(model, tokenizer, _mc_prompt(ex["question"], ex["choices"]), thinking, 64)
        correct += _mc_score(gen, gold)
        done += 1
        _set_postfix(bar, acc=f"{100.0 * correct / done:.1f}%")
    return {"score": round(100.0 * correct / max(len(ds), 1), 2), "n": len(ds)}


# --------------------------------------------------------------------------- orchestrator


def _seed_report(out_path: str | Path, hub_repo: str | None) -> dict:
    """Resume-safe: merge into any existing report (local file, else Hub copy)
    so a partial run's completed modes survive a second session that only runs
    the missing ones (train/phase_controller.py resume flow)."""
    if Path(out_path).exists():
        return json.loads(Path(out_path).read_text(encoding="utf-8"))
    if hub_repo:
        try:
            from huggingface_hub import hf_hub_download

            local = hf_hub_download(hub_repo, f"reports/{Path(out_path).name}")
            return json.loads(Path(local).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - no prior report == fresh start
            pass
    return {}


def run_c1(
    model: Any,
    tokenizer: Any,
    out_path: str | Path,
    modes: tuple[bool, ...] = (False, True),   # non-think first (cheap)
    code_limit: int | None = None,             # None = full sets
    floor_limits: tuple[int, int] = (500, 1000),
    hub_repo: str | None = None,
) -> dict:
    report: dict[str, Any] = _seed_report(out_path, hub_repo)
    report["meta"] = {"unix": time.time()}

    def _persist() -> None:
        Path(out_path).write_text(json.dumps(report, indent=2))
        if hub_repo:
            from huggingface_hub import HfApi

            api = HfApi()
            # C1 may run before anything else ever touched the repo — create it
            # here or the first upload dies with RepositoryNotFoundError after
            # hours of eval.
            api.create_repo(hub_repo, private=True, exist_ok=True)
            api.upload_file(
                path_or_fileobj=str(out_path),
                path_in_repo=f"reports/{Path(out_path).name}",
                repo_id=hub_repo,
            )

    for thinking in modes:
        mode = "thinking" if thinking else "non_thinking"
        mode_report = report.setdefault(mode, {})
        todo = [b for b in ("humaneval_plus", "mbpp_plus", "arc_easy", "mmlu")
                if b not in mode_report]
        print(f"[c1] === mode={mode} | remaining: {todo or 'nothing (resumed complete)'} ===")
        benches: list[tuple[str, Any]] = [
            ("humaneval_plus", lambda t=thinking: eval_evalplus(
                model, tokenizer, "evalplus/humanevalplus", t, code_limit)),
            ("mbpp_plus", lambda t=thinking: eval_evalplus(
                model, tokenizer, "evalplus/mbppplus", t, code_limit)),
            ("arc_easy", lambda t=thinking: eval_arc_easy(model, tokenizer, t, floor_limits[0])),
            ("mmlu", lambda t=thinking: eval_mmlu(model, tokenizer, t, floor_limits[1])),
        ]
        for name, fn in benches:
            if name in mode_report:  # benchmark-level resume
                continue
            mode_report[name] = fn()
            _persist()  # save + push after EVERY benchmark — max loss = one bench
            print(f"[c1] {mode}/{name}: {mode_report[name].get('score')}")
    return report
