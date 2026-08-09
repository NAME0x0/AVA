"""Series 1 / Exp B: which layers are dead weight, and does layer TYPE predict it?

Post-hoc, forward-pass only, no training — so it runs on the real donors at real
scale on a 4 GB laptop (same property that made quantization tractable for us).

Metric: angular cosine distance between a block's input and output hidden states
(cheaper than Block Influence: last-token only, per the 2026 pruning literature).
Small distance => the block barely moves the residual stream => prune candidate.

The question that actually matters for OUR models: both donors are HYBRIDS
(Qwen3.5-4B = 24 GDN + 8 full attention; LFM2.5 = 22 short-conv + 8 GQA). The
pruning literature is built on plain transformers. In a 3:1 hybrid the sparse
full-attention layers are deliberately placed and I expect them to be
load-bearing — so this reports redundancy SPLIT BY LAYER TYPE to test that
directly instead of assuming it.
"""
import json
import sys

import torch
import transformers

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-4B"
NCAL = int(sys.argv[2]) if len(sys.argv) > 2 else 16
OUT = f"D:/AVA/ratchet/logs/redundancy_{MODEL.split('/')[-1]}.json"

CAL = [
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n",
    "class LRUCache:\n    def __init__(self, capacity: int):\n        self.capacity = capacity\n",
    "The mitochondria is the powerhouse of the cell, converting nutrients into ATP through",
    "import pandas as pd\ndf = pd.read_csv('data.csv')\ndf.groupby('category')",
    "To prove that the square root of two is irrational, assume the contrary: that",
    "async def fetch_all(urls):\n    async with aiohttp.ClientSession() as session:\n",
    "The French Revolution began in 1789 and fundamentally reshaped European politics by",
    "SELECT customers.name, COUNT(orders.id) FROM customers LEFT JOIN orders ON",
]


def layer_types(model, cfg) -> list[str]:
    """Best-effort per-layer type label (hybrids interleave mixers)."""
    lt = getattr(cfg, "layer_types", None) or getattr(cfg, "layer_type_list", None)
    if lt:
        return [str(x) for x in lt]
    labels = []
    for blk in model.model.layers:
        names = {n for n, _ in blk.named_modules()}
        has_attn = any("self_attn" in n or "attention" in n for n in names)
        has_conv = any("conv" in n.lower() for n in names)
        has_gdn = any("linear_attn" in n or "delta" in n.lower() for n in names)
        labels.append("conv" if has_conv and not has_attn else
                      "linear_attn" if has_gdn else
                      "attention" if has_attn else "?")
    return labels


bnb = transformers.BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map={"": 0}, dtype=torch.bfloat16,
    trust_remote_code=True)
tok = transformers.AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model.eval()
cfg = model.config
types = layer_types(model, cfg)
n_layers = len(types)
print(f"{MODEL} | {n_layers} layers | vram {torch.cuda.memory_allocated()/1e9:.2f}GB")
print(f"layer types: {types}\n", flush=True)

sums = torch.zeros(n_layers, dtype=torch.float64)
count = 0
texts = (CAL * ((NCAL // len(CAL)) + 1))[:NCAL]
for t in texts:
    ids = tok(t, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    hs = out.hidden_states                      # [n_layers+1] x (1, seq, d)
    for i in range(n_layers):
        a = hs[i][0, -1].float()                # last token (cheap variant)
        b = hs[i + 1][0, -1].float()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).clamp(-1, 1)
        sums[i] += torch.arccos(cos).item()     # angular distance
    count += 1

ang = (sums / count).tolist()
rows = sorted(range(n_layers), key=lambda i: ang[i])
print(f"{'rank':>4} {'layer':>6} {'type':>14} {'angular_dist':>13}   (low = redundant)")
for rank, i in enumerate(rows):
    print(f"{rank:>4} {i:>6} {types[i]:>14} {ang[i]:>13.4f}")

by_type: dict[str, list[float]] = {}
for i, t in enumerate(types):
    by_type.setdefault(t, []).append(ang[i])
print("\n=== mean angular distance by layer type (higher = more load-bearing) ===")
for t, v in sorted(by_type.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
    print(f"  {t:>14}: {sum(v)/len(v):.4f}   (n={len(v)})")

k = max(1, n_layers // 4)
cand = rows[:k]
print(f"\ntop-{k} prune candidates (25%): {sorted(cand)}")
print("  types:", [types[i] for i in sorted(cand)])
json.dump({"model": MODEL, "angular": ang, "types": types,
           "ranked_redundant": rows, "candidates_25pct": sorted(cand)},
          open(OUT, "w", encoding="utf-8"), indent=2)
print(f"\nwrote {OUT}")
