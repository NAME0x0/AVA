"""Self-consistency benchmark for GSM8K.

Generates K samples per problem and takes majority vote on extracted answers.
This is one of the biggest accuracy boosters at inference time (typically +10-20%
on GSM8K) and requires no additional training.

Reference: Wang et al. 2022 "Self-Consistency Improves Chain of Thought Reasoning"
"""
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from ava.external_benchmarks import load_external_benchmark_tasks

BASE_MODEL = "D:/AVA/experiments/exp4_finetune/models/Qwen3.5-2B"
RESULTS_DIR = Path("D:/AVA/experiments/exp4_finetune/results")


@dataclass
class SCConfig:
    model_path: str = BASE_MODEL
    adapter_path: str | None = None
    gsm8k_limit: int = 50
    k_samples: int = 5  # Number of samples per problem
    temperature: float = 0.7  # Higher temp for diverse samples
    max_new_tokens: int = 768


def load_model(config: SCConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, dtype=torch.bfloat16,
    )
    name = Path(config.model_path).name
    if config.adapter_path:
        ac = Path(config.adapter_path) / "adapter_config.json"
        if ac.exists():
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, config.adapter_path)
            model = model.merge_and_unload()
            name = Path(config.adapter_path).name + " (LoRA)"
    model.eval()
    return model, tokenizer, name


def extract_numeric(text):
    patterns = [
        r"(?:the answer is|the final answer is)\s*\$?([-\d,]+\.?\d*)",
        r"(?:####)\s*\$?([-\d,]+\.?\d*)",
        r"(?:final answer:?)\s*\$?([-\d,]+\.?\d*)",
        r"(?:answer:)\s*\$?([-\d,]+\.?\d*)",
        r"(?:therefore|thus|so|hence)[,:]?\s+\$?([-\d,]+\.?\d*)",
        r"\\boxed\{([-\d,]+\.?\d*)\}",
        r"=\s*\$?([-\d,]+\.?\d*)\s*$",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE | re.MULTILINE)
        if m:
            val = m.group(1).replace(",", "").rstrip(".")
            if val and val != "-":
                return val
    nums = re.findall(r"[-\d,]+\.?\d*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def normalize_number(s):
    """Normalize a number string for comparison."""
    if not s:
        return None
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return f"{f:.2f}"
    except ValueError:
        return s


@torch.no_grad()
def generate_k_samples(model, tokenizer, prompt, config: SCConfig):
    """Generate K diverse samples for the same problem."""
    messages = [
        {"role": "user", "content": prompt + "\n\nSolve this step by step and give the final numeric answer."},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    samples = []
    for _ in range(config.k_samples):
        out = model.generate(
            **inputs, max_new_tokens=config.max_new_tokens,
            temperature=config.temperature, do_sample=True,
            top_p=0.95, pad_token_id=tokenizer.eos_token_id,
        )
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        samples.append(resp)
    return samples


def majority_vote(answers):
    """Take majority vote over normalized answers."""
    normalized = [normalize_number(a) for a in answers if a is not None]
    if not normalized:
        return None
    counter = Counter(normalized)
    return counter.most_common(1)[0][0]


def check_match(predicted, expected):
    if not predicted:
        return False
    try:
        return abs(float(predicted) - float(expected)) < 0.01
    except (ValueError, TypeError):
        return str(predicted) == str(expected)


def run_sc_benchmark(config: SCConfig | None = None):
    config = config or SCConfig()

    print("=" * 60)
    print(f"Self-Consistency GSM8K Benchmark (K={config.k_samples})")
    print("=" * 60)

    model, tokenizer, model_name = load_model(config)
    print(f"Model: {model_name}")
    print(f"GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"K samples: {config.k_samples}, Temperature: {config.temperature}\n")

    tasks = load_external_benchmark_tasks("gsm8k", split=None, limit=config.gsm8k_limit, offset=0)

    greedy_correct = 0
    sc_correct = 0
    results = []

    for i, task in enumerate(tasks):
        expected = task.expected.strip()

        # Generate K samples
        samples = generate_k_samples(model, tokenizer, task.prompt, config)
        answers = [extract_numeric(s) for s in samples]

        # Greedy = first sample (but with temp so not truly greedy)
        greedy_pred = answers[0]
        greedy_ok = check_match(greedy_pred, expected)
        if greedy_ok:
            greedy_correct += 1

        # Self-consistency = majority vote
        sc_pred = majority_vote(answers)
        sc_ok = check_match(sc_pred, expected)
        if sc_ok:
            sc_correct += 1

        results.append({
            "task_id": task.task_id,
            "expected": expected,
            "greedy_pred": greedy_pred,
            "greedy_ok": greedy_ok,
            "sc_pred": sc_pred,
            "sc_ok": sc_ok,
            "all_answers": answers,
            "unique_answers": len(set(normalize_number(a) for a in answers if a)),
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(tasks)}] Greedy: {greedy_correct}/{i+1} ({greedy_correct/(i+1)*100:.0f}%) "
                  f"| SC@{config.k_samples}: {sc_correct}/{i+1} ({sc_correct/(i+1)*100:.0f}%)")

    n = len(tasks)
    greedy_acc = greedy_correct / n
    sc_acc = sc_correct / n

    print(f"\n{'='*60}")
    print(f"RESULTS: {model_name}")
    print(f"{'='*60}")
    print(f"  Greedy GSM8K:     {greedy_correct}/{n} = {greedy_acc*100:.1f}%")
    print(f"  SC@{config.k_samples} GSM8K:      {sc_correct}/{n} = {sc_acc*100:.1f}%")
    print(f"  Delta:            {(sc_acc-greedy_acc)*100:+.1f}pp")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = Path(config.adapter_path).name if config.adapter_path else "base"
    out = RESULTS_DIR / f"sc{config.k_samples}_gsm8k_{tag}.json"
    summary = {
        "model": model_name,
        "k_samples": config.k_samples,
        "temperature": config.temperature,
        "greedy_accuracy": round(greedy_acc, 3),
        "sc_accuracy": round(sc_acc, 3),
        "delta_pp": round((sc_acc - greedy_acc) * 100, 1),
        "results": results,
    }
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved to: {out}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    run_sc_benchmark(SCConfig(
        model_path=args.model,
        adapter_path=args.adapter,
        gsm8k_limit=args.limit,
        k_samples=args.k,
        temperature=args.temperature,
    ))
