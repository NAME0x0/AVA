"""Generate MMLU-by-subject, MATH-by-level, and MGSM-by-language breakdown
tables from source JSON."""
import json
from pathlib import Path

D = Path("D:/AVA/experiments/exp4_finetune/eval_v2/results/v2_full")
T = Path("D:/AVA/paper/tables")
T.mkdir(parents=True, exist_ok=True)


def rows(items):
    return "\n".join(
        f"{k.replace('_', ' ')} & {n} & {a*100:.1f} \\\\" for k, n, a in items
    )


# MMLU top-10 / bottom-10 subjects (exclude the "_" ctx-overflow bucket)
mmlu = json.loads((D / "mmlu_summary.json").read_text())["by_subject"]
subs = [(k, v["n"], v["acc"]) for k, v in mmlu.items() if k != "_"]
subs.sort(key=lambda x: x[2], reverse=True)
lines = [r"\begin{tabular}{lrr}", r"\toprule",
         r"MMLU subject & $n$ & Acc.\ (\%) \\", r"\midrule",
         r"\multicolumn{3}{l}{\emph{Strongest 10}}\\", rows(subs[:10]), r"\midrule",
         r"\multicolumn{3}{l}{\emph{Weakest 10}}\\", rows(subs[-10:]),
         r"\bottomrule", r"\end{tabular}"]
(T / "mmlu_subjects.tex").write_text("\n".join(lines), encoding="utf-8")

# MATH-500 by difficulty level
lv = json.loads((D / "math-500_summary.json").read_text())["by_level"]
lines = [r"\begin{tabular}{lrr}", r"\toprule", r"Level & $n$ & Acc.\ (\%) \\", r"\midrule"]
for k in sorted(lv):
    lines.append(f"{k} & {lv[k]['n']} & {lv[k]['acc']*100:.1f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(T / "math_levels.tex").write_text("\n".join(lines), encoding="utf-8")

# MGSM by language
lg = json.loads((D / "mgsm_summary.json").read_text())["by_lang"]
lines = [r"\begin{tabular}{lrr}", r"\toprule", r"Language & $n$ & Acc.\ (\%) \\", r"\midrule"]
for k in ("en", "es", "fr"):
    if k in lg:
        lines.append(f"{k} & {lg[k]['n']} & {lg[k]['acc']*100:.1f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(T / "mgsm_lang.tex").write_text("\n".join(lines), encoding="utf-8")

print("wrote mmlu_subjects.tex, math_levels.tex, mgsm_lang.tex")
