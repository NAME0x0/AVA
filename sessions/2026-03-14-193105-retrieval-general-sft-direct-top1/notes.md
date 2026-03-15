# Retrieval Session: retrieval-general-sft-direct-top1

## Command

`ava session retrieval retrieval-general-sft-direct-top1 sessions/2026-03-14-184859-failure-patch-v2-rerun-11m-96/checkpoints/ava-11m-failure-patch-v2.pt corpora/general_sft --device cuda --max-new-tokens 48 --retrieval-top-k 1 --mode direct`

## Inputs

- Checkpoint: `sessions/2026-03-14-184859-failure-patch-v2-rerun-11m-96/checkpoints/ava-11m-failure-patch-v2.pt`
- Support corpus: `corpora/general_sft`
- Support files: `2`
- Support examples: `59`
- Retrieval top_k: `1`
- Retrieval mode: `direct`
- Category gated: `True`

## Accuracy Delta

- Benchmark: `9/10` -> `8/10`
- Tool eval: `6/6` -> `6/6`
- Compliance: `4/4` -> `4/4`

## Changed Benchmark Rows

- [science] prompt=`What planet is known as the Red Planet?` baseline=`[car 493` -> retrieval=`Mars`
- [tool] prompt=`Use the calculator tool for 144 / 12.` baseline=`12` -> retrieval=`[/ /cavvvvvvvvvvvvvvvvvvvinnothelcalcalculelcalc`
- [tool] prompt=`Use the calculator tool for sqrt(81).` baseline=`9` -> retrieval=`[/ calcannor /calcathelp /cavvvvvvvvvvvis`

## Changed Tool Rows

- none

## Changed Compliance Rows

- none

## Focus Prompt

- `What planet is known as the Red Planet?`
- Baseline trace: `results/focus_trace_baseline.json`
- Retrieval trace: `results/focus_trace_retrieval.json`

## Artifacts

- Baseline benchmark/tool/compliance under `results/`
- Retrieval benchmark/tool/compliance under `results/`
