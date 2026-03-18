"""Build augmented v2 corpus by combining base v2 with extra tool-use examples.

This creates the best-quality corpus for v2 training.
"""
import json
import random
from pathlib import Path

random.seed(42)

BASE_CORPUS = Path("D:/AVA/experiments/exp4_finetune/corpora/ava_exp4_finetune_v2.jsonl")
TOOL_AUGMENT = Path("D:/AVA/experiments/exp4_finetune/corpora/tool_use_augmented.jsonl")
OUTPUT = Path("D:/AVA/experiments/exp4_finetune/corpora/ava_exp4_finetune_v2_augmented.jsonl")

# Load base corpus
with open(BASE_CORPUS, encoding="utf-8") as f:
    examples = [json.loads(line) for line in f if line.strip()]
print(f"Base v2 corpus: {len(examples)} examples")

# Load tool augmentation
with open(TOOL_AUGMENT, encoding="utf-8") as f:
    tool_examples = [json.loads(line) for line in f if line.strip()]
print(f"Tool augmentation: {len(tool_examples)} examples")

# Combine
examples.extend(tool_examples)
random.shuffle(examples)

# Write
with open(OUTPUT, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"\nAugmented corpus: {len(examples)} examples")
print(f"Written to: {OUTPUT}")
print(f"Size: {OUTPUT.stat().st_size / 1024:.1f} KB")

# Count tool-use examples
tool_count = sum(1 for ex in examples if "tool_call" in ex.get("response", ""))
print(f"Tool-use examples: {tool_count} ({tool_count/len(examples)*100:.1f}%)")
