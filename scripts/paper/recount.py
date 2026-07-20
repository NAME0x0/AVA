import json
from pathlib import Path
D = Path("D:/AVA/experiments/exp4_finetune/eval_v2/results/v2_full")
rows = {}
for f in sorted(D.glob("*_summary.json")):
    s = json.loads(f.read_text())
    rows[f.stem.replace("_summary", "")] = s.get("n")
total_n = sum(v for v in rows.values() if isinstance(v, int))
print("per-benchmark n:", json.dumps(rows, indent=2))
print("sum of n column:", total_n)
