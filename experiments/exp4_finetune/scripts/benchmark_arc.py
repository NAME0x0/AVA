"""Benchmark Qwen3.5-2B on ARC-Challenge using log-prob MCQ scoring."""
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Add AVA src to path for benchmark loading
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from ava.external_benchmarks import load_external_benchmark_tasks

MODEL_PATH = "D:/AVA/experiments/exp4_finetune/models/Qwen3.5-2B"
LIMIT = 100
OFFSET = 0
OUTPUT_PATH = "D:/AVA/experiments/exp4_finetune/results/arc_qwen35_2b_baseline_100.json"

print(f"Loading ARC-Challenge tasks (limit={LIMIT})...")
tasks = load_external_benchmark_tasks("arc-challenge", split=None, limit=LIMIT, offset=OFFSET)
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


@torch.no_grad()
def score_mcq_logprobs(prompt: str, choices: tuple) -> tuple[str, dict]:
    """Score MCQ by comparing log-probs of each choice label token."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]  # last token logits
    log_probs = torch.log_softmax(logits, dim=-1)

    scores = {}
    for label, _choice_text in choices:
        # Try encoding just the label (A, B, C, D)
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if token_ids:
            scores[label] = float(log_probs[token_ids[0]].item())
        else:
            scores[label] = float("-inf")

    best = max(scores, key=scores.get)
    return best, scores


@torch.no_grad()
def score_mcq_generation(prompt: str, choices: tuple) -> tuple[str, dict]:
    """Score MCQ by generating a response and checking if it matches a choice."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=8,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    # Check if generated text starts with a valid label
    for label, _choice_text in choices:
        if generated.upper().startswith(label.upper()):
            return label, {"generated": generated, "method": "generation"}

    # Fallback: no match found
    return generated, {"generated": generated, "method": "generation_no_match"}


def normalize_answer(text: str) -> str:
    return text.strip().lower().rstrip(".")


print(f"\nRunning ARC-Challenge benchmark ({len(tasks)} items)...")
print("=" * 60)

results = []
correct_logprob = 0
correct_gen = 0
start_time = time.perf_counter()

for i, task in enumerate(tasks):
    # Method 1: Log-prob scoring
    pred_lp, scores_lp = score_mcq_logprobs(task.prompt, task.choices)
    matched_lp = normalize_answer(pred_lp) == normalize_answer(task.expected)
    if matched_lp:
        correct_logprob += 1

    # Method 2: Generation scoring
    pred_gen, scores_gen = score_mcq_generation(task.prompt, task.choices)
    matched_gen = normalize_answer(pred_gen) == normalize_answer(task.expected)
    if matched_gen:
        correct_gen += 1

    results.append({
        "task_id": task.task_id,
        "category": task.category,
        "expected": task.expected,
        "logprob_pred": pred_lp,
        "logprob_matched": matched_lp,
        "gen_pred": pred_gen,
        "gen_matched": matched_gen,
    })

    if (i + 1) % 10 == 0:
        elapsed = time.perf_counter() - start_time
        print(f"  [{i+1}/{len(tasks)}] logprob={correct_logprob}/{i+1} ({correct_logprob/(i+1)*100:.1f}%) | gen={correct_gen}/{i+1} ({correct_gen/(i+1)*100:.1f}%) | {elapsed:.0f}s")

elapsed = time.perf_counter() - start_time

# Summary
payload = {
    "model": "Qwen3.5-2B",
    "model_path": MODEL_PATH,
    "quantization": "4bit-nf4-double",
    "benchmark": "arc-challenge",
    "limit": LIMIT,
    "offset": OFFSET,
    "logprob_accuracy": round(correct_logprob / len(tasks), 3),
    "logprob_correct": correct_logprob,
    "generation_accuracy": round(correct_gen / len(tasks), 3),
    "generation_correct": correct_gen,
    "total": len(tasks),
    "elapsed_seconds": round(elapsed, 1),
    "results": results,
}

Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
Path(OUTPUT_PATH).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

print(f"\n{'='*60}")
print(f"ARC-Challenge Results (Qwen3.5-2B, 4-bit):")
print(f"  Log-prob scoring: {correct_logprob}/{len(tasks)} = {correct_logprob/len(tasks)*100:.1f}%")
print(f"  Generation scoring: {correct_gen}/{len(tasks)} = {correct_gen/len(tasks)*100:.1f}%")
print(f"  Time: {elapsed:.0f}s ({elapsed/len(tasks):.1f}s/item)")
print(f"  Output: {OUTPUT_PATH}")
print(f"\nFor comparison: AVA v3 scratch model = 24% baseline, 33% with retrieval")
