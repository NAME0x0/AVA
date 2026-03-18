"""Benchmark the fine-tuned Qwen3.5-2B-AVA model on ARC-Challenge.

Compares base model vs fine-tuned (with LoRA adapters).
"""
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from ava.external_benchmarks import load_external_benchmark_tasks

BASE_MODEL_PATH = "D:/AVA/experiments/exp4_finetune/models/Qwen3.5-2B"
ADAPTER_PATH = "D:/AVA/experiments/exp4_finetune/models/Qwen3.5-2B-AVA-v1"
LIMIT = 100
OUTPUT_PATH = "D:/AVA/experiments/exp4_finetune/results/arc_qwen35_2b_ava_v1_100.json"

print(f"Loading ARC-Challenge tasks (limit={LIMIT})...")
tasks = load_external_benchmark_tasks("arc-challenge", split=None, limit=LIMIT, offset=0)
print(f"Loaded {len(tasks)} tasks")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

print("Loading base model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,
)

# Check if adapter exists and load it
adapter_config = Path(ADAPTER_PATH) / "adapter_config.json"
if adapter_config.exists():
    print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model = model.merge_and_unload()
    model_name = "Qwen3.5-2B-AVA-v1"
    print("LoRA adapter loaded and merged!")
else:
    print(f"No adapter found at {ADAPTER_PATH}, using base model")
    model_name = "Qwen3.5-2B-base"

model.eval()
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


@torch.no_grad()
def score_mcq_logprobs(prompt: str, choices: tuple) -> tuple[str, dict]:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    log_probs = torch.log_softmax(logits, dim=-1)

    scores = {}
    for label, _choice_text in choices:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if token_ids:
            scores[label] = float(log_probs[token_ids[0]].item())
        else:
            scores[label] = float("-inf")

    best = max(scores, key=scores.get)
    return best, scores


print(f"\nRunning ARC-Challenge benchmark ({len(tasks)} items)...")
print("=" * 60)

results = []
correct = 0
start_time = time.perf_counter()

for i, task in enumerate(tasks):
    pred, scores = score_mcq_logprobs(task.prompt, task.choices)
    matched = pred.strip().lower().rstrip(".") == task.expected.strip().lower().rstrip(".")
    if matched:
        correct += 1

    results.append({
        "task_id": task.task_id,
        "category": task.category,
        "expected": task.expected,
        "predicted": pred,
        "matched": matched,
    })

    if (i + 1) % 10 == 0:
        elapsed = time.perf_counter() - start_time
        print(f"  [{i+1}/{len(tasks)}] correct={correct}/{i+1} ({correct/(i+1)*100:.1f}%) | {elapsed:.0f}s")

elapsed = time.perf_counter() - start_time

payload = {
    "model": model_name,
    "base_model": BASE_MODEL_PATH,
    "adapter": ADAPTER_PATH if adapter_config.exists() else None,
    "quantization": "4bit-nf4-double",
    "benchmark": "arc-challenge",
    "limit": LIMIT,
    "accuracy": round(correct / len(tasks), 3),
    "correct": correct,
    "total": len(tasks),
    "elapsed_seconds": round(elapsed, 1),
    "results": results,
}

Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(OUTPUT_PATH).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

print(f"\n{'='*60}")
print(f"ARC-Challenge Results ({model_name}):")
print(f"  Accuracy: {correct}/{len(tasks)} = {correct/len(tasks)*100:.1f}%")
print(f"  Time: {elapsed:.0f}s")
print(f"\nComparison:")
print(f"  AVA v3 scratch (14M): 24% baseline")
print(f"  Qwen3.5-2B base:     66% baseline")
print(f"  Qwen3.5-2B-AVA-v1:   {correct/len(tasks)*100:.1f}%")
