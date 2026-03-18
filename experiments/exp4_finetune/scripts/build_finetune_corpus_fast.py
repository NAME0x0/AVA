"""Build a focused fast-iteration corpus for quick training experiments.

~5K carefully curated examples for 2-3 hour training runs.
Priority: GSM8K CoT (biggest potential improvement) + tool-use + identity.
"""
import json
import random
from pathlib import Path

random.seed(42)

OUTPUT_PATH = Path("D:/AVA/experiments/exp4_finetune/corpora")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

examples: list[dict] = []
stats: dict[str, int] = {}


def add(name: str, new: list[dict]) -> None:
    examples.extend(new)
    stats[name] = len(new)
    print(f"  [{name}] {len(new)} examples")


# === 1. GSM8K Chain-of-Thought (highest priority — massive potential gain) ===
gsm8k_path = Path("D:/AVA/corpora/gsm8k_train_reasoning_support_v1/examples.jsonl")
if gsm8k_path.exists():
    with open(gsm8k_path, encoding="utf-8") as f:
        raw = [json.loads(l) for l in f if l.strip()]
    cot = []
    for ex in raw:
        r, a, p = ex.get("teacher_rationale_short", ""), ex.get("response", ""), ex.get("prompt", "")
        if not (r and a and p):
            continue
        p_clean = p.strip().replace("Reply with only the final numeric answer.", "").strip()
        cot.append({"prompt": p_clean, "response": f"Let me solve this step by step.\n\n{r}\n\nThe answer is {a}."})
    # Take 3000 — enough to learn CoT pattern
    if len(cot) > 3000:
        cot = random.sample(cot, 3000)
    add("gsm8k_cot", cot)


# === 2. ARC-Challenge training (2nd priority — direct benchmark improvement) ===
arc_path = Path("D:/AVA/corpora/public_benchmark_distill_v1/examples.jsonl")
if arc_path.exists():
    with open(arc_path, encoding="utf-8") as f:
        raw = [json.loads(l) for l in f if l.strip()]
    arc = []
    for ex in raw:
        if ex.get("benchmark") != "arc-challenge":
            continue
        p, label = ex.get("prompt", ""), ex.get("response", "")
        if not (p and label):
            continue
        text = ""
        for line in p.split("\n"):
            if line.strip().startswith(f"{label}.") or line.strip().startswith(f"{label} "):
                text = line.strip()[2:].strip()
                break
        p_clean = p.strip().replace("Reply with only the correct option label.", "").strip()
        resp = f"The answer is {label}. {text}." if text else f"The answer is {label}."
        arc.append({"prompt": p_clean, "response": resp})
    if len(arc) > 1500:
        arc = random.sample(arc, 1500)
    add("arc_challenge", arc)


# === 3. Tool-use + identity + reasoning (from v2 corpus builder) ===
# These are small but important for model personality
from build_finetune_corpus_v2 import (
    tool_use_examples, reasoning_examples, identity_examples, conversation_examples,
)
add("tool_use", tool_use_examples)
add("reasoning", reasoning_examples)
add("identity", identity_examples)
add("conversation", conversation_examples)


# === 4. Math breakthrough distill ===
bt_path = Path("D:/AVA/corpora/ava_v3_breakthrough_distill_v1/examples.jsonl")
if bt_path.exists():
    with open(bt_path, encoding="utf-8") as f:
        raw = [json.loads(l) for l in f if l.strip()]
    bt = []
    for ex in raw:
        r, a, p = ex.get("teacher_rationale_short", ""), ex.get("response", ""), ex.get("prompt", "")
        if not p:
            continue
        p_clean = p.strip().replace("Reply with only the final numeric answer.", "").strip()
        if r:
            bt.append({"prompt": p_clean, "response": f"Let me work through this.\n\n{r}\n\nThe answer is {a}."})
        else:
            bt.append({"prompt": p_clean, "response": f"The answer is {a}."})
    add("breakthrough", bt)


# === Shuffle and write ===
random.shuffle(examples)
output_file = OUTPUT_PATH / "ava_exp4_finetune_fast.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"\nTotal: {len(examples)} examples")
print(f"Written to: {output_file}")
print(f"Size: {output_file.stat().st_size / 1024:.1f} KB")
for name, count in sorted(stats.items(), key=lambda x: -x[1]):
    print(f"  {name}: {count} ({count/len(examples)*100:.1f}%)")

# Estimated training time at 14s/step, batch=2, grad_accum=4
n_steps = len(examples) // 8  # effective batch size 8
print(f"\nEstimated training: {n_steps} steps x 14s = {n_steps*14/3600:.1f} hours")
