"""V2 evaluation runner.

Loads a benchmark, dispatches each task by kind, saves per-bench JSON +
aggregate stats. Designed for llama-server backend (Q8_0 GGUF) running with
flash-attn, 4 parallel slots, continuous batching.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from client import LlamaClient
from code_sandbox import humaneval_check, mbpp_check
from extractors import (extract_boxed, extract_numeric, extract_python_code,
                         math_match, numeric_match)
from ifeval_rules import evaluate_response as ifeval_eval
from loaders import LOADERS, TIERS, BenchTask

RESULTS_DIR = Path("D:/AVA/experiments/exp4_finetune/eval_v2/results")


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    rad = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - rad), min(1.0, center + rad))


async def _run_mcq_logprob(client: LlamaClient, task: BenchTask) -> dict:
    score = await client.score_mcq_logprob(task.prompt, task.labels)
    matched = score.selected.upper() == task.expected.upper()
    return {
        "task_id": task.task_id,
        "kind": "mcq_logprob",
        "expected": task.expected,
        "predicted": score.selected,
        "matched": matched,
        "margin": score.margin,
        "label_logprobs": score.label_logprobs,
        "metadata": task.metadata,
    }


async def _run_generate(client: LlamaClient, task: BenchTask) -> dict:
    out = await client.chat_generate(
        task.messages,
        max_tokens=task.max_tokens,
        temperature=0.0,
        stop=task.stop,
    )
    text = out["text"]
    metadata = task.metadata
    matched = False
    extracted = None

    if task.benchmark in ("gsm8k", "gsm8k-selfcons", "mgsm-en", "mgsm-es",
                          "mgsm-fr"):
        extracted = extract_numeric(text)
        matched = numeric_match(extracted, task.expected)
    elif task.benchmark == "math-500":
        extracted = extract_boxed(text) or extract_numeric(text)
        matched = math_match(extracted, task.expected)
    elif task.benchmark == "humaneval-plus":
        code = extract_python_code(text)
        extracted = code
        ok, msg = humaneval_check(
            code,
            metadata["prompt"],
            metadata["test"],
            metadata["entry_point"],
        )
        matched = ok
        metadata = {**metadata, "exec_msg": msg[:300]}
    elif task.benchmark == "mbpp-plus":
        code = extract_python_code(text)
        extracted = code
        ok, msg = mbpp_check(code, metadata["tests"])
        matched = ok
        metadata = {**metadata, "exec_msg": msg[:300]}
    elif task.benchmark == "ifeval":
        ok, per_rule = ifeval_eval(
            text,
            metadata["instruction_id_list"],
            metadata["kwargs"],
        )
        matched = ok
        extracted = "ifeval-pass" if ok else "ifeval-fail"
        metadata = {**metadata, "rules": per_rule}
    elif task.benchmark == "bfcl-v3-simple":
        extracted = text.strip()
        try:
            obj = json.loads(extracted[extracted.find("{"):extracted.rfind("}")+1])
            gt = metadata["ground_truth"]
            matched = obj.get("name") == gt.get("name")
        except Exception:
            matched = False
    else:
        extracted = text
        matched = task.expected.strip() in text.strip()

    return {
        "task_id": task.task_id,
        "kind": "generate",
        "expected": task.expected,
        "extracted": extracted,
        "matched": matched,
        "response": text[:600],
        "usage": out.get("usage", {}),
        "metadata": metadata,
    }


async def _run_selfcons(client: LlamaClient, task: BenchTask) -> dict:
    seeds = task.metadata["seeds"]
    k = task.metadata["k"]
    answers: list[tuple[str | None, str]] = []
    for seed in seeds:
        out = await client.chat_generate(
            task.messages,
            max_tokens=task.max_tokens,
            temperature=0.7,
            top_p=0.95,
            seed=seed,
        )
        ext = extract_numeric(out["text"])
        answers.append((ext, out["text"][:200]))

    counts: Counter[str] = Counter()
    for ans, _ in answers:
        if ans is not None:
            counts[ans] += 1
    if counts:
        majority = counts.most_common(1)[0][0]
    else:
        majority = ""
    matched = numeric_match(majority, task.expected)
    return {
        "task_id": task.task_id,
        "kind": "selfcons",
        "k": k,
        "expected": task.expected,
        "predicted": majority,
        "matched": matched,
        "all_answers": [a[0] for a in answers],
        "metadata": task.metadata,
    }


async def _run_agentic_gsm8k(client: LlamaClient, task: BenchTask) -> dict:
    from agentic import run_agentic_task
    user_msg = task.messages[0]["content"]
    question = user_msg.split("\n\n")[0]
    out = await run_agentic_task(client, question, task.expected)
    return {
        "task_id": task.task_id,
        "kind": "agentic",
        "expected": task.expected,
        "extracted": out["extracted"],
        "matched": out["matched"],
        "used_tools": out["used_tools"],
        "response": out["response"],
        "metadata": task.metadata,
    }


async def _dispatch(client: LlamaClient, task: BenchTask) -> dict:
    if task.benchmark == "agentic-gsm8k":
        return await _run_agentic_gsm8k(client, task)
    if task.benchmark == "gsm8k-selfcons":
        return await _run_selfcons(client, task)
    if task.kind == "mcq_logprob":
        return await _run_mcq_logprob(client, task)
    return await _run_generate(client, task)


def _summarize(results: list[dict], by_keys: list[str] | None = None) -> dict:
    n = len(results)
    correct = sum(1 for r in results if r.get("matched"))
    acc = correct / max(n, 1)
    lo, hi = _wilson_ci(acc, n)
    summary: dict = {
        "n": n,
        "correct": correct,
        "accuracy": round(acc, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
    }
    if by_keys:
        for key in by_keys:
            buckets: dict[str, list[dict]] = defaultdict(list)
            for r in results:
                v = str((r.get("metadata") or {}).get(key, "_"))
                buckets[v].append(r)
            summary[f"by_{key}"] = {
                k: {
                    "n": len(v),
                    "acc": round(sum(1 for x in v if x.get("matched")) / max(len(v), 1), 4),
                }
                for k, v in sorted(buckets.items())
            }
    return summary


async def run_benchmark(name: str, *, limit: int | None = None,
                         offset: int = 0,
                         tier: int | None = None) -> dict:
    loader = LOADERS[name]
    print(f"[{name}] loading...")
    t_load = time.perf_counter()
    if name == "mgsm":
        per_lang = limit // 3 if limit else None
        tasks = loader(per_lang_limit=per_lang)
    else:
        tasks = loader(limit=limit, offset=offset)
    print(f"[{name}] loaded {len(tasks)} tasks in {time.perf_counter()-t_load:.1f}s")
    if not tasks:
        return {"benchmark": name, "skipped": True, "reason": "no_tasks"}

    by_keys = []
    if tasks[0].benchmark.startswith("mmlu") and "subject" in (tasks[0].metadata or {}):
        by_keys.append("subject")
    if tasks[0].benchmark == "mmlu-pro":
        by_keys.append("category")
    if tasks[0].benchmark == "math-500":
        by_keys.extend(["subject", "level"])
    if tasks[0].benchmark.startswith("mgsm"):
        by_keys.append("lang")

    async with LlamaClient(parallel=4) as client:
        await client.health()
        t0 = time.perf_counter()
        sem_tasks = [asyncio.create_task(_dispatch(client, t)) for t in tasks]
        results: list[dict] = []
        report_every = max(10, len(tasks) // 50)
        for i, fut in enumerate(asyncio.as_completed(sem_tasks)):
            try:
                r = await fut
            except Exception as e:
                r = {"task_id": "?", "matched": False, "error": str(e)}
            results.append(r)
            if (i + 1) % report_every == 0 or (i + 1) == len(tasks):
                acc = sum(1 for x in results if x.get("matched")) / len(results)
                elapsed = time.perf_counter() - t0
                rate = len(results) / max(elapsed, 1e-3)
                eta = (len(tasks) - len(results)) / max(rate, 1e-3)
                print(f"[{name}] {len(results)}/{len(tasks)} "
                      f"acc={acc:.3f} ({rate:.2f}/s, eta={eta:.0f}s)")
        elapsed_total = time.perf_counter() - t0

    summary = _summarize(results, by_keys=by_keys)
    summary["benchmark"] = name
    summary["elapsed_s"] = round(elapsed_total, 1)
    summary["throughput"] = round(len(tasks) / max(elapsed_total, 1e-3), 2)
    summary["model"] = "AVA-v2-Q8_0"
    summary["backend"] = "llama-server (flash-attn, Q8 KV)"
    summary["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    out_dir = RESULTS_DIR / "v2_full"
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_name = name.replace("/", "_")
    (out_dir / f"{bench_name}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (out_dir / f"{bench_name}_details.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(f"[{name}] DONE acc={summary['accuracy']:.3f} "
          f"ci=[{summary['ci95'][0]:.3f},{summary['ci95'][1]:.3f}] "
          f"n={summary['n']} time={summary['elapsed_s']:.0f}s")
    return summary


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bench", action="append", help="benchmark names (repeatable)")
    p.add_argument("--tier", type=int, help="run all benchmarks in tier")
    p.add_argument("--limit", type=int, default=None,
                   help="cap items per benchmark (for smoke tests)")
    p.add_argument("--offset", type=int, default=0)
    args = p.parse_args()

    benches: list[str] = []
    if args.tier:
        benches.extend(TIERS[args.tier])
    if args.bench:
        benches.extend(args.bench)
    if not benches:
        p.error("provide --bench or --tier")

    overall: list[dict] = []
    for name in benches:
        s = await run_benchmark(name, limit=args.limit, offset=args.offset)
        overall.append(s)

    out = RESULTS_DIR / "v2_full" / "_aggregate.json"
    existing = []
    if out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))
    existing.extend(overall)
    out.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    for s in overall:
        print(f"  {s.get('benchmark'):20s}  "
              f"acc={s.get('accuracy', 0):.3f}  "
              f"n={s.get('n', 0):5d}  "
              f"t={s.get('elapsed_s', 0):.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
