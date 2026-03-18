"""Build DPO (Direct Preference Optimization) corpus from GSM8K.

For each math problem:
1. Generate K responses from the SFT model
2. Check which answers are correct
3. Save (chosen, rejected) pairs for DPO training

This requires a trained model checkpoint. Run AFTER SFT training completes.

Usage:
    python scripts/build_dpo_corpus.py --adapter models/Qwen3.5-2B-AVA-v1 --k 4 --limit 500
"""
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from ava.external_benchmarks import load_external_benchmark_tasks

BASE_MODEL = "D:/AVA/experiments/exp4_finetune/models/Qwen3.5-2B"
OUTPUT_DIR = Path("D:/AVA/experiments/exp4_finetune/corpora")


@dataclass
class DPOConfig:
    model_path: str = BASE_MODEL
    adapter_path: str | None = None
    gsm8k_limit: int = 500  # Use more problems for DPO
    k_samples: int = 4  # Generate K responses per problem
    temperature: float = 0.8
    max_new_tokens: int = 768
    output_path: str = str(OUTPUT_DIR / "ava_exp4_dpo_gsm8k.jsonl")


def load_model(config: DPOConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, dtype=torch.bfloat16,
    )
    if config.adapter_path:
        ac = Path(config.adapter_path) / "adapter_config.json"
        if ac.exists():
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, config.adapter_path)
            model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


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


def check_match(predicted, expected):
    if not predicted:
        return False
    try:
        return abs(float(predicted) - float(expected)) < 0.01
    except (ValueError, TypeError):
        return str(predicted) == str(expected)


@torch.no_grad()
def generate_samples(model, tokenizer, prompt, config: DPOConfig):
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


def build_dpo_corpus(config: DPOConfig | None = None):
    config = config or DPOConfig()

    print("=" * 60)
    print("Building DPO Corpus from GSM8K")
    print("=" * 60)

    model, tokenizer = load_model(config)
    print(f"GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"GSM8K problems: {config.gsm8k_limit}")
    print(f"K samples per problem: {config.k_samples}")

    tasks = load_external_benchmark_tasks("gsm8k", split=None, limit=config.gsm8k_limit, offset=0)

    dpo_pairs = []
    stats = {"total_problems": 0, "pairs_generated": 0, "no_correct": 0, "all_correct": 0}

    start = time.perf_counter()
    for i, task in enumerate(tasks):
        expected = task.expected.strip()
        prompt = task.prompt.strip()

        # Generate K responses
        responses = generate_samples(model, tokenizer, prompt, config)
        answers = [extract_numeric(r) for r in responses]
        correct_mask = [check_match(a, expected) for a in answers]

        stats["total_problems"] += 1
        correct_responses = [r for r, c in zip(responses, correct_mask) if c]
        incorrect_responses = [r for r, c in zip(responses, correct_mask) if not c]

        if not correct_responses:
            stats["no_correct"] += 1
            continue
        if not incorrect_responses:
            stats["all_correct"] += 1
            continue

        # Create DPO pairs: each correct paired with each incorrect
        for chosen in correct_responses:
            for rejected in incorrect_responses:
                dpo_pairs.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "expected": expected,
                })
                stats["pairs_generated"] += 1

        if (i + 1) % 20 == 0:
            elapsed = time.perf_counter() - start
            rate = (i + 1) / elapsed * 60
            print(f"  [{i+1}/{len(tasks)}] {stats['pairs_generated']} pairs | "
                  f"{stats['no_correct']} no-correct | {stats['all_correct']} all-correct | "
                  f"{rate:.0f} problems/min")

    elapsed = time.perf_counter() - start

    # Save
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\n{'='*60}")
    print(f"DPO Corpus Complete!")
    print(f"{'='*60}")
    print(f"  Problems processed: {stats['total_problems']}")
    print(f"  DPO pairs: {stats['pairs_generated']}")
    print(f"  No correct response: {stats['no_correct']} ({stats['no_correct']/max(stats['total_problems'],1)*100:.1f}%)")
    print(f"  All correct: {stats['all_correct']} ({stats['all_correct']/max(stats['total_problems'],1)*100:.1f}%)")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Saved to: {output_path}")

    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--output", default=str(OUTPUT_DIR / "ava_exp4_dpo_gsm8k.jsonl"))
    args = parser.parse_args()

    build_dpo_corpus(DPOConfig(
        model_path=args.model,
        adapter_path=args.adapter,
        gsm8k_limit=args.limit,
        k_samples=args.k,
        output_path=args.output,
    ))
