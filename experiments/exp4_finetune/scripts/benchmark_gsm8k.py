"""Benchmark Qwen3.5-2B on GSM8K using generation scoring."""
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from ava.external_benchmarks import load_external_benchmark_tasks

MODEL_PATH = "D:/AVA/experiments/exp4_finetune/models/Qwen3.5-2B"
LIMIT = 50
OUTPUT_PATH = "D:/AVA/experiments/exp4_finetune/results/gsm8k_qwen35_2b_baseline_50.json"

print(f"Loading GSM8K tasks (limit={LIMIT})...")
tasks = load_external_benchmark_tasks("gsm8k", split=None, limit=LIMIT, offset=0)
print(f"Loaded {len(tasks)} tasks")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Loading model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,
)
model.eval()
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def extract_number(text: str) -> str:
    """Extract the final number from a response."""
    # Look for numbers, possibly with commas or decimals
    numbers = re.findall(r'-?[\d,]+\.?\d*', text.replace(',', ''))
    if numbers:
        return numbers[-1].strip()
    return text.strip()


@torch.no_grad()
def solve_gsm8k(prompt: str, enable_thinking: bool = False) -> tuple[str, float]:
    """Generate solution and extract answer."""
    messages = [{"role": "user", "content": prompt + "\n\nPlease solve step by step and give the final numerical answer."}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    start = time.perf_counter()
    outputs = model.generate(
        **inputs,
        max_new_tokens=256 if enable_thinking else 128,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    elapsed = time.perf_counter() - start

    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return generated, elapsed


print(f"\nRunning GSM8K benchmark ({len(tasks)} items, no-think mode)...")
print("=" * 60)

results = []
correct = 0
start_time = time.perf_counter()

for i, task in enumerate(tasks):
    generated, item_time = solve_gsm8k(task.prompt, enable_thinking=False)
    pred_number = extract_number(generated)
    expected_number = extract_number(task.expected)
    matched = pred_number == expected_number

    if matched:
        correct += 1

    results.append({
        "task_id": task.task_id,
        "expected": task.expected,
        "expected_number": expected_number,
        "pred_number": pred_number,
        "matched": matched,
        "generated": generated[:200],
        "time": round(item_time, 1),
    })

    if (i + 1) % 10 == 0:
        elapsed = time.perf_counter() - start_time
        print(f"  [{i+1}/{len(tasks)}] correct={correct}/{i+1} ({correct/(i+1)*100:.1f}%) | {elapsed:.0f}s")

elapsed = time.perf_counter() - start_time

payload = {
    "model": "Qwen3.5-2B",
    "quantization": "4bit-nf4-double",
    "benchmark": "gsm8k",
    "mode": "no_think",
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
print(f"GSM8K Results (Qwen3.5-2B, 4-bit, no-think):")
print(f"  Accuracy: {correct}/{len(tasks)} = {correct/len(tasks)*100:.1f}%")
print(f"  Time: {elapsed:.0f}s")
print(f"\nFor comparison: AVA v3 scratch model = 0-2% GSM8K")


# Now try with thinking enabled
print(f"\n\nRunning GSM8K benchmark ({len(tasks)} items, WITH thinking)...")
print("=" * 60)

results_think = []
correct_think = 0
start_time = time.perf_counter()

for i, task in enumerate(tasks):
    generated, item_time = solve_gsm8k(task.prompt, enable_thinking=True)
    # Strip thinking tags if present
    if "</think>" in generated:
        generated_answer = generated.split("</think>")[-1].strip()
    else:
        generated_answer = generated
    pred_number = extract_number(generated_answer)
    expected_number = extract_number(task.expected)
    matched = pred_number == expected_number

    if matched:
        correct_think += 1

    results_think.append({
        "task_id": task.task_id,
        "expected": task.expected,
        "expected_number": expected_number,
        "pred_number": pred_number,
        "matched": matched,
        "generated": generated[:500],
        "time": round(item_time, 1),
    })

    if (i + 1) % 10 == 0:
        elapsed = time.perf_counter() - start_time
        print(f"  [{i+1}/{len(tasks)}] correct={correct_think}/{i+1} ({correct_think/(i+1)*100:.1f}%) | {elapsed:.0f}s")

elapsed = time.perf_counter() - start_time

payload_think = {
    "model": "Qwen3.5-2B",
    "quantization": "4bit-nf4-double",
    "benchmark": "gsm8k",
    "mode": "with_thinking",
    "limit": LIMIT,
    "accuracy": round(correct_think / len(tasks), 3),
    "correct": correct_think,
    "total": len(tasks),
    "elapsed_seconds": round(elapsed, 1),
    "results": results_think,
}

output_think = OUTPUT_PATH.replace("baseline", "thinking")
Path(output_think).write_text(json.dumps(payload_think, indent=2) + "\n", encoding="utf-8")

print(f"\n{'='*60}")
print(f"GSM8K Results (Qwen3.5-2B, 4-bit, WITH thinking):")
print(f"  Accuracy: {correct_think}/{len(tasks)} = {correct_think/len(tasks)*100:.1f}%")
print(f"  Time: {elapsed:.0f}s")
print(f"\nComparison:")
print(f"  No-think: {correct}/{len(tasks)} = {correct/len(tasks)*100:.1f}%")
print(f"  With-think: {correct_think}/{len(tasks)} = {correct_think/len(tasks)*100:.1f}%")
print(f"  AVA v3 scratch: 0-2%")
