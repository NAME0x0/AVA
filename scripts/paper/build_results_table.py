"""Generate the headline results table (paper/tables/results_main.tex) from the
per-benchmark summary JSON, so the numbers cannot drift from source."""
import json
from pathlib import Path

D = Path("D:/AVA/experiments/exp4_finetune/eval_v2/results/v2_full")
OUT = Path("D:/AVA/paper/tables/results_main.tex")
OUT.parent.mkdir(parents=True, exist_ok=True)

# (json stem, pretty label)
ROWS = [
    ("arc-challenge", "ARC-Challenge"),
    ("arc-easy", "ARC-Easy"),
    ("mmlu", "MMLU (5-shot)"),
    ("mmlu-pro", "MMLU-Pro"),
    ("hellaswag", "HellaSwag"),
    ("piqa", "PIQA"),
    ("winogrande", "WinoGrande"),
    ("boolq", "BoolQ"),
    ("truthfulqa-mc1", "TruthfulQA-MC1"),
    ("gsm8k", "GSM8K"),
    ("gsm8k-selfcons", "GSM8K self-cons.\\ ($k{=}5$)"),
    ("math-500", "MATH-500"),
    ("mgsm", "MGSM (en/es/fr)"),
    ("humaneval-plus", "HumanEval+"),
    ("mbpp-plus", "MBPP+"),
    ("ifeval", "IFEval"),
    ("agentic-gsm8k", "Agentic GSM8K"),
]

lines = [r"\begin{tabular}{lrrr}", r"\toprule",
         r"Benchmark & $n$ & Acc.\ (\%) & 95\% CI \\", r"\midrule"]
for stem, label in ROWS:
    s = json.loads((D / f"{stem}_summary.json").read_text())
    lo, hi = s["ci95"]
    lines.append(f"{label} & {s['n']:,} & {s['accuracy']*100:.1f} & "
                 f"[{lo*100:.1f}, {hi*100:.1f}] \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {OUT}")
