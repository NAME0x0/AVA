# Task-count reconciliation (Task 3) — 2026-07-20

The v2 report and model card carried three different totals (16,872 / 52,027 / 52,045). This defines one taxonomy; the paper and card use it consistently.

## Canonical taxonomy
- **Benchmarks (configurations): 17.**
- **Evaluation instances: 52,027.** The sum of the per-benchmark `n` across the 17 `*_summary.json` files (verified by `scripts/paper/recount.py`). This is the source-of-truth headline count. Note it counts *benchmark rows*, not distinct underlying questions: GSM8K items recur across `gsm8k` (1,319), `gsm8k-selfcons` (a 200-item subset), `agentic-gsm8k` (1,319), and `mgsm` (en, 250). Phrase as "evaluation instances across 17 benchmark configurations."
- **Model calls (throughput): ≈52,045.** The runner's internal generation counter (MCQ single-token forward passes 47,193 + generative decodes 4,852). Includes self-consistency samples and minor retries, so it slightly exceeds the instance count. Report only in the appendix as "≈52k model calls," never as the headline.

## Deprecated
- **16,872** — the stale header figure in `RESULTS_REPORT_V2_FULL.md`. It is not reproducible from the summary JSONs (which sum to 52,027) and must be removed from the report header and the model card. Do not use it anywhere.

## Canonical phrasing (paper + card)
> "AVA-v2 was evaluated across 17 benchmark configurations comprising 52,027 evaluation instances (≈52k model calls)."
