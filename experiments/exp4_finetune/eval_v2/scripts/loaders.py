"""Benchmark loaders for AVA v2 full evaluation.

Each loader returns list[BenchTask] where BenchTask describes:
  - prompt(s) (or message list) to send to the model
  - eval kind (mcq_logprob, mcq_continuation, generate)
  - candidate labels / continuations / expected answer
  - extraction & match logic owner (runner picks correct path)

We deliberately avoid mutating the existing src/ava/external_benchmarks.py to
keep production code stable. This module is the v2-specific eval surface.
"""
from __future__ import annotations

import json
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import load_dataset


@dataclass
class BenchTask:
    benchmark: str
    task_id: str
    kind: str
    prompt: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    continuations: list[str] = field(default_factory=list)
    expected: str = ""
    expected_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    max_tokens: int = 512
    stop: list[str] | None = None


def _slice(split: str, limit: int | None, offset: int = 0) -> str:
    if "[" in split:
        return split
    start = max(offset, 0)
    if limit is None:
        return f"{split}[{start}:]" if start else split
    return f"{split}[{start}:{start + max(limit, 0)}]"


def _format_mcq(question: str, choices: Iterable[tuple[str, str]]) -> str:
    lines = [question.strip(), ""]
    for label, text in choices:
        lines.append(f"{label}. {text}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


def _gsm8k_extract(text: str) -> str:
    if "####" in text:
        text = text.split("####", 1)[1]
    matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if not matches:
        return text.strip()
    return matches[-1].replace(",", "")


def load_arc_challenge(*, limit: int | None = None, offset: int = 0) -> list[BenchTask]:
    ds = load_dataset(
        "allenai/ai2_arc", "ARC-Challenge",
        split=_slice("test", limit, offset),
    )
    out: list[BenchTask] = []
    for row in ds:
        labels = [str(x) for x in row["choices"]["label"]]
        texts = [str(x) for x in row["choices"]["text"]]
        choices = list(zip(labels, texts, strict=True))
        prompt = _format_mcq(str(row["question"]), choices)
        out.append(BenchTask(
            benchmark="arc-challenge",
            task_id=str(row["id"]),
            kind="mcq_logprob",
            prompt=prompt,
            labels=labels,
            expected=str(row["answerKey"]),
        ))
    return out


def load_arc_easy(*, limit: int | None = None, offset: int = 0) -> list[BenchTask]:
    ds = load_dataset(
        "allenai/ai2_arc", "ARC-Easy",
        split=_slice("test", limit, offset),
    )
    out: list[BenchTask] = []
    for row in ds:
        labels = [str(x) for x in row["choices"]["label"]]
        texts = [str(x) for x in row["choices"]["text"]]
        choices = list(zip(labels, texts, strict=True))
        out.append(BenchTask(
            benchmark="arc-easy",
            task_id=str(row["id"]),
            kind="mcq_logprob",
            prompt=_format_mcq(str(row["question"]), choices),
            labels=labels,
            expected=str(row["answerKey"]),
        ))
    return out


def load_gsm8k(*, limit: int | None = None, offset: int = 0) -> list[BenchTask]:
    ds = load_dataset("gsm8k", "main", split=_slice("test", limit, offset))
    out: list[BenchTask] = []
    for idx, row in enumerate(ds):
        question = str(row["question"]).strip()
        gold = _gsm8k_extract(str(row["answer"]))
        prompt_user = (
            f"{question}\n\n"
            "Solve step by step. End with: The answer is <number>."
        )
        out.append(BenchTask(
            benchmark="gsm8k",
            task_id=str(idx),
            kind="generate",
            messages=[{"role": "user", "content": prompt_user}],
            expected=gold,
            max_tokens=512,
        ))
    return out


def load_piqa(*, limit: int | None = None, offset: int = 0) -> list[BenchTask]:
    ds = load_dataset("baber/piqa", split=_slice("validation", limit, offset))
    out: list[BenchTask] = []
    for idx, row in enumerate(ds):
        choices = [("A", str(row["sol1"])), ("B", str(row["sol2"]))]
        expected = "A" if int(row["label"]) == 0 else "B"
        out.append(BenchTask(
            benchmark="piqa",
            task_id=str(idx),
            kind="mcq_logprob",
            prompt=_format_mcq(str(row["goal"]), choices),
            labels=["A", "B"],
            expected=expected,
        ))
    return out


def load_hellaswag(*, limit: int | None = None, offset: int = 0) -> list[BenchTask]:
    ds = load_dataset("Rowan/hellaswag",
                      split=_slice("validation", limit, offset))
    out: list[BenchTask] = []
    for row in ds:
        ctx = (str(row["activity_label"]) + ": " + str(row["ctx"])).strip()
        endings = [str(x) for x in row["endings"]]
        labels = ["A", "B", "C", "D"]
        choices = list(zip(labels, endings, strict=True))
        out.append(BenchTask(
            benchmark="hellaswag",
            task_id=str(row["ind"]),
            kind="mcq_logprob",
            prompt=_format_mcq(ctx, choices),
            labels=labels,
            expected=labels[int(row["label"])],
        ))
    return out


def load_winogrande(*, limit: int | None = None, offset: int = 0) -> list[BenchTask]:
    ds = load_dataset(
        "allenai/winogrande", "winogrande_xl",
        split=_slice("validation", limit, offset),
    )
    out: list[BenchTask] = []
    for idx, row in enumerate(ds):
        sentence = str(row["sentence"])
        opt1, opt2 = str(row["option1"]), str(row["option2"])
        prompt = (
            f"Fill in the blank with the most likely option.\n\n"
            f"Sentence: {sentence}\n\n"
            f"A. {opt1}\nB. {opt2}\n\nAnswer:"
        )
        expected = "A" if str(row["answer"]) == "1" else "B"
        out.append(BenchTask(
            benchmark="winogrande",
            task_id=str(idx),
            kind="mcq_logprob",
            prompt=prompt,
            labels=["A", "B"],
            expected=expected,
        ))
    return out


def load_boolq(*, limit: int | None = None, offset: int = 0) -> list[BenchTask]:
    ds = load_dataset("google/boolq", split=_slice("validation", limit, offset))
    out: list[BenchTask] = []
    for idx, row in enumerate(ds):
        passage = str(row["passage"])
        question = str(row["question"])
        prompt = (
            f"Passage: {passage}\n\n"
            f"Question: {question}?\n"
            f"A. Yes\nB. No\n\nAnswer:"
        )
        expected = "A" if bool(row["answer"]) else "B"
        out.append(BenchTask(
            benchmark="boolq",
            task_id=str(idx),
            kind="mcq_logprob",
            prompt=prompt,
            labels=["A", "B"],
            expected=expected,
        ))
    return out


def load_truthfulqa_mc(*, limit: int | None = None, offset: int = 0) -> list[BenchTask]:
    ds = load_dataset(
        "truthfulqa/truthful_qa", "multiple_choice",
        split=_slice("validation", limit, offset),
    )
    out: list[BenchTask] = []
    for idx, row in enumerate(ds):
        choices_data = row["mc1_targets"]
        texts = [str(c) for c in choices_data["choices"]]
        labels_int = [int(x) for x in choices_data["labels"]]
        if len(texts) > 26:
            texts = texts[:26]
            labels_int = labels_int[:26]
        labels = [chr(ord("A") + i) for i in range(len(texts))]
        choices = list(zip(labels, texts, strict=True))
        expected_idx = labels_int.index(1)
        out.append(BenchTask(
            benchmark="truthfulqa-mc1",
            task_id=str(idx),
            kind="mcq_logprob",
            prompt=_format_mcq(str(row["question"]), choices),
            labels=labels,
            expected=labels[expected_idx],
        ))
    return out


_MMLU_DEV_CACHE: dict[str, list[dict[str, Any]]] = {}


def _mmlu_few_shot(subject: str, k: int = 5) -> str:
    if subject not in _MMLU_DEV_CACHE:
        try:
            dev = load_dataset("cais/mmlu", subject, split="dev")
        except Exception:
            return ""
        _MMLU_DEV_CACHE[subject] = list(dev)
    rows = _MMLU_DEV_CACHE[subject][:k]
    parts: list[str] = []
    letters = ["A", "B", "C", "D"]
    for row in rows:
        q = str(row["question"]).strip()
        choices = [str(c) for c in row["choices"]]
        ans = letters[int(row["answer"])]
        block = q + "\n" + "\n".join(
            f"{letters[i]}. {choices[i]}" for i in range(len(choices))
        ) + f"\nAnswer: {ans}"
        parts.append(block)
    return "\n\n".join(parts) + ("\n\n" if parts else "")


def load_mmlu(*, limit: int | None = None, offset: int = 0,
              few_shot: int = 5) -> list[BenchTask]:
    ds = load_dataset("cais/mmlu", "all", split=_slice("test", limit, offset))
    letters = ["A", "B", "C", "D"]
    out: list[BenchTask] = []
    for idx, row in enumerate(ds):
        subject = str(row["subject"])
        prefix = _mmlu_few_shot(subject, k=few_shot) if few_shot else ""
        choices_text = [str(c) for c in row["choices"]]
        choices = list(zip(letters, choices_text, strict=True))
        question_block = (
            str(row["question"]).strip() + "\n" +
            "\n".join(f"{lab}. {txt}" for lab, txt in choices) +
            "\nAnswer:"
        )
        prompt = prefix + question_block
        out.append(BenchTask(
            benchmark="mmlu",
            task_id=f"{subject}__{idx}",
            kind="mcq_logprob",
            prompt=prompt,
            labels=letters,
            expected=letters[int(row["answer"])],
            metadata={"subject": subject, "few_shot": few_shot},
        ))
    return out


def load_mmlu_pro(*, limit: int | None = None, offset: int = 0) -> list[BenchTask]:
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split=_slice("test", limit, offset))
    out: list[BenchTask] = []
    for row in ds:
        opts = [str(o) for o in row["options"]]
        labels = [chr(ord("A") + i) for i in range(len(opts))]
        choices = list(zip(labels, opts, strict=True))
        out.append(BenchTask(
            benchmark="mmlu-pro",
            task_id=str(row["question_id"]),
            kind="mcq_logprob",
            prompt=_format_mcq(str(row["question"]), choices),
            labels=labels,
            expected=str(row["answer"]),
            metadata={"category": str(row["category"])},
        ))
    return out


def load_math500(*, limit: int | None = None, offset: int = 0) -> list[BenchTask]:
    ds = load_dataset("HuggingFaceH4/MATH-500", split=_slice("test", limit, offset))
    out: list[BenchTask] = []
    for idx, row in enumerate(ds):
        problem = str(row["problem"]).strip()
        gold = str(row["answer"]).strip()
        prompt_user = (
            f"{problem}\n\n"
            "Solve step by step. Put your final answer in \\boxed{}."
        )
        out.append(BenchTask(
            benchmark="math-500",
            task_id=str(idx),
            kind="generate",
            messages=[{"role": "user", "content": prompt_user}],
            expected=gold,
            max_tokens=1024,
            metadata={"subject": str(row["subject"]), "level": int(row["level"])},
        ))
    return out


def load_mgsm(*, langs: tuple[str, ...] = ("en", "es", "fr"),
              per_lang_limit: int | None = None) -> list[BenchTask]:
    out: list[BenchTask] = []
    for lang in langs:
        try:
            ds = load_dataset(
                "juletxara/mgsm", lang,
                split=_slice("test", per_lang_limit, 0),
            )
        except Exception:
            continue
        for idx, row in enumerate(ds):
            question = str(row["question"]).strip()
            ans_raw = str(row.get("answer_number") or row.get("answer") or "")
            gold = _gsm8k_extract(ans_raw)
            prompt_user = (
                f"{question}\n\nSolve step by step. End with: The answer is <number>."
            )
            out.append(BenchTask(
                benchmark=f"mgsm-{lang}",
                task_id=f"{lang}-{idx}",
                kind="generate",
                messages=[{"role": "user", "content": prompt_user}],
                expected=gold,
                max_tokens=512,
                metadata={"lang": lang},
            ))
    return out


def load_ifeval(*, limit: int | None = None, offset: int = 0) -> list[BenchTask]:
    ds = load_dataset("google/IFEval", split=_slice("train", limit, offset))
    out: list[BenchTask] = []
    for row in ds:
        instr = str(row["prompt"])
        out.append(BenchTask(
            benchmark="ifeval",
            task_id=str(row["key"]),
            kind="generate",
            messages=[{"role": "user", "content": instr}],
            expected="",
            max_tokens=1024,
            metadata={
                "instruction_id_list": list(row["instruction_id_list"]),
                "kwargs": list(row["kwargs"]),
                "prompt": instr,
            },
        ))
    return out


def load_humaneval_plus(*, limit: int | None = None,
                         offset: int = 0) -> list[BenchTask]:
    try:
        ds = load_dataset("evalplus/humanevalplus",
                          split=_slice("test", limit, offset))
    except Exception:
        ds = load_dataset("openai/openai_humaneval",
                          split=_slice("test", limit, offset))
    out: list[BenchTask] = []
    for row in ds:
        task_id = str(row["task_id"])
        prompt = str(row["prompt"])
        out.append(BenchTask(
            benchmark="humaneval-plus",
            task_id=task_id,
            kind="generate",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Complete the following Python function. Return only the "
                        "function body inside a Python code block.\n\n"
                        f"```python\n{prompt}\n```"
                    ),
                },
            ],
            expected="",
            max_tokens=768,
            metadata={
                "prompt": prompt,
                "test": str(row.get("test", "")),
                "entry_point": str(row.get("entry_point", "")),
                "canonical_solution": str(row.get("canonical_solution", "")),
            },
        ))
    return out


def load_mbpp_plus(*, limit: int | None = None, offset: int = 0) -> list[BenchTask]:
    try:
        ds = load_dataset("evalplus/mbppplus", split=_slice("test", limit, offset))
    except Exception:
        ds = load_dataset("google-research-datasets/mbpp", "sanitized",
                          split=_slice("test", limit, offset))
    out: list[BenchTask] = []
    for row in ds:
        task_id = str(row.get("task_id", row.get("id", "")))
        prompt = str(row.get("prompt") or row.get("text") or "")
        tests = row.get("test_list") or row.get("test") or []
        if isinstance(tests, list):
            tests_text = "\n".join(str(t) for t in tests)
        else:
            tests_text = str(tests)
        out.append(BenchTask(
            benchmark="mbpp-plus",
            task_id=task_id,
            kind="generate",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"{prompt}\n\nYour code should pass these tests:\n"
                        f"{tests_text}\n\n"
                        "Return only the Python implementation in a code block."
                    ),
                },
            ],
            expected="",
            max_tokens=768,
            metadata={"prompt": prompt, "tests": tests_text},
        ))
    return out


def load_bfcl_simple(*, limit: int | None = None,
                     offset: int = 0) -> list[BenchTask]:
    """BFCL v3 simple AST checks. Uses local mirror of gorilla-llm BFCL data
    if present, else returns []. Not fatal — we report skip."""
    bfcl_path = Path("D:/AVA/data/bfcl/BFCL_v3_simple.json")
    if not bfcl_path.exists():
        return []
    rows = json.loads(bfcl_path.read_text(encoding="utf-8"))
    if offset:
        rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]
    out: list[BenchTask] = []
    for row in rows:
        out.append(BenchTask(
            benchmark="bfcl-v3-simple",
            task_id=str(row["id"]),
            kind="generate",
            messages=[
                {"role": "system", "content":
                 "You are a function-calling assistant. Respond with a single "
                 "JSON object: {\"name\": \"...\", \"arguments\": {...}}."},
                {"role": "user", "content": json.dumps({
                    "question": row["question"],
                    "function": row["function"],
                })},
            ],
            expected=json.dumps(row["ground_truth"]),
            max_tokens=512,
            metadata={"function": row["function"], "ground_truth": row["ground_truth"]},
        ))
    return out


def load_agentic_gsm8k(*, limit: int | None = None,
                         offset: int = 0) -> list[BenchTask]:
    """GSM8K with tool-use prompt. Runner dispatches to agentic loop."""
    base = load_gsm8k(limit=limit, offset=offset)
    for t in base:
        t.benchmark = "agentic-gsm8k"
    return base


# Self-consistency: same as gsm8k but flagged for multi-sample voting.
def load_gsm8k_selfcons(*, limit: int = 200, k: int = 5,
                         offset: int = 0) -> list[BenchTask]:
    base = load_gsm8k(limit=limit, offset=offset)
    rng = random.Random(20260504)
    for t in base:
        t.benchmark = "gsm8k-selfcons"
        t.metadata["k"] = k
        t.metadata["seeds"] = [rng.randint(1, 2**31 - 1) for _ in range(k)]
    return base


LOADERS: dict[str, Any] = {
    "arc-challenge": load_arc_challenge,
    "arc-easy": load_arc_easy,
    "gsm8k": load_gsm8k,
    "piqa": load_piqa,
    "hellaswag": load_hellaswag,
    "winogrande": load_winogrande,
    "boolq": load_boolq,
    "truthfulqa-mc1": load_truthfulqa_mc,
    "mmlu": load_mmlu,
    "mmlu-pro": load_mmlu_pro,
    "math-500": load_math500,
    "mgsm": load_mgsm,
    "ifeval": load_ifeval,
    "humaneval-plus": load_humaneval_plus,
    "mbpp-plus": load_mbpp_plus,
    "bfcl-v3-simple": load_bfcl_simple,
    "gsm8k-selfcons": load_gsm8k_selfcons,
    "agentic-gsm8k": load_agentic_gsm8k,
}


TIERS: dict[int, list[str]] = {
    1: ["arc-challenge", "arc-easy", "gsm8k", "mmlu", "truthfulqa-mc1"],
    2: ["hellaswag", "piqa", "winogrande", "boolq", "mmlu-pro"],
    3: ["gsm8k-selfcons", "bfcl-v3-simple"],
    4: ["humaneval-plus", "mbpp-plus"],
    5: ["ifeval"],
    6: ["math-500", "mgsm"],
}


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "arc-challenge"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    tasks = LOADERS[name](limit=n)
    for t in tasks[:n]:
        print(f"--- {t.task_id} ({t.kind}) expected={t.expected!r}")
        if t.prompt:
            print(t.prompt[:300])
        elif t.messages:
            print(t.messages[0]["content"][:300])
        print()
