"""Compare all saved checkpoints on a quick ARC + GSM8K subset.

Evaluates each checkpoint-N directory and the final model, producing
a comparison table. Run AFTER training completes (GPU must be free).
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
OUTPUT_DIR = Path("D:/AVA/experiments/exp4_finetune/models/Qwen3.5-2B-AVA-v1")
RESULTS_DIR = Path("D:/AVA/experiments/exp4_finetune/results")

ARC_LIMIT = 25  # Small fast subset
GSM8K_LIMIT = 15


def load_model_with_adapter(adapter_path=None):
    """Load base model + optional adapter. Returns model, tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto",
        trust_remote_code=True, dtype=torch.bfloat16,
    )
    if adapter_path and (Path(adapter_path) / "adapter_config.json").exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


@torch.no_grad()
def eval_arc_quick(model, tokenizer, tasks):
    correct = 0
    for task in tasks:
        messages = [{"role": "user", "content": task.prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        logits = model(**inputs).logits[0, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        scores = {}
        for label, _ in task.choices:
            ids = tokenizer.encode(label, add_special_tokens=False)
            scores[label] = float(log_probs[ids[0]].item()) if ids else float("-inf")
        pred = max(scores, key=scores.get)
        if pred.strip().lower() == task.expected.strip().lower():
            correct += 1
    return correct, len(tasks)


@torch.no_grad()
def eval_gsm8k_quick(model, tokenizer, tasks):
    correct = 0
    for task in tasks:
        prompt = task.prompt.strip()
        if "step by step" not in prompt.lower():
            prompt += "\n\nSolve step by step. Give the final numeric answer."
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs, max_new_tokens=384, temperature=0.1, do_sample=True,
            top_p=0.95, pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        predicted = extract_numeric(response)
        expected = task.expected.strip()
        try:
            if predicted and abs(float(predicted) - float(expected)) < 0.01:
                correct += 1
        except (ValueError, TypeError):
            pass
    return correct, len(tasks)


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


def find_checkpoints():
    """Find all checkpoint-N directories and final model."""
    checkpoints = []
    if OUTPUT_DIR.exists():
        for d in sorted(OUTPUT_DIR.iterdir()):
            if d.is_dir() and d.name.startswith("checkpoint-"):
                step = int(d.name.split("-")[1])
                checkpoints.append((step, str(d)))
        # Check if final adapter exists at top level
        if (OUTPUT_DIR / "adapter_config.json").exists():
            checkpoints.append((999999, str(OUTPUT_DIR)))
    return checkpoints


def main():
    print("=" * 70)
    print("Checkpoint Comparison: ARC-Challenge + GSM8K")
    print("=" * 70)

    checkpoints = find_checkpoints()
    if not checkpoints:
        print("No checkpoints found!")
        return

    print(f"Found {len(checkpoints)} checkpoints")
    print(f"ARC subset: {ARC_LIMIT} | GSM8K subset: {GSM8K_LIMIT}\n")

    # Load benchmark tasks once
    arc_tasks = load_external_benchmark_tasks("arc-challenge", split=None, limit=ARC_LIMIT, offset=0)
    gsm_tasks = load_external_benchmark_tasks("gsm8k", split=None, limit=GSM8K_LIMIT, offset=0)

    results = []

    # Baseline (no adapter)
    print("Evaluating: Base Qwen3.5-2B (no adapter)...")
    model, tokenizer = load_model_with_adapter(None)
    arc_c, arc_t = eval_arc_quick(model, tokenizer, arc_tasks)
    gsm_c, gsm_t = eval_gsm8k_quick(model, tokenizer, gsm_tasks)
    results.append({
        "name": "Base (no adapter)",
        "step": 0,
        "arc": f"{arc_c}/{arc_t} ({arc_c/arc_t*100:.0f}%)",
        "gsm8k": f"{gsm_c}/{gsm_t} ({gsm_c/gsm_t*100:.0f}%)",
        "arc_pct": arc_c / arc_t,
        "gsm_pct": gsm_c / gsm_t,
    })
    print(f"  ARC: {arc_c}/{arc_t} = {arc_c/arc_t*100:.0f}% | GSM8K: {gsm_c}/{gsm_t} = {gsm_c/gsm_t*100:.0f}%")
    del model
    torch.cuda.empty_cache()

    # Checkpoints
    for step, path in checkpoints:
        label = "Final" if step == 999999 else f"checkpoint-{step}"
        print(f"\nEvaluating: {label}...")
        model, tokenizer = load_model_with_adapter(path)
        arc_c, arc_t = eval_arc_quick(model, tokenizer, arc_tasks)
        gsm_c, gsm_t = eval_gsm8k_quick(model, tokenizer, gsm_tasks)
        results.append({
            "name": label,
            "step": step,
            "arc": f"{arc_c}/{arc_t} ({arc_c/arc_t*100:.0f}%)",
            "gsm8k": f"{gsm_c}/{gsm_t} ({gsm_c/gsm_t*100:.0f}%)",
            "arc_pct": arc_c / arc_t,
            "gsm_pct": gsm_c / gsm_t,
        })
        print(f"  ARC: {arc_c}/{arc_t} = {arc_c/arc_t*100:.0f}% | GSM8K: {gsm_c}/{gsm_t} = {gsm_c/gsm_t*100:.0f}%")
        del model
        torch.cuda.empty_cache()

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Model':<25} {'ARC':>15} {'GSM8K':>15}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<25} {r['arc']:>15} {r['gsm8k']:>15}")
    print("=" * 70)

    # Find best
    best_arc = max(results, key=lambda x: x["arc_pct"])
    best_gsm = max(results, key=lambda x: x["gsm_pct"])
    print(f"\nBest ARC: {best_arc['name']} ({best_arc['arc']})")
    print(f"Best GSM8K: {best_gsm['name']} ({best_gsm['gsm8k']})")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "checkpoint_comparison.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
