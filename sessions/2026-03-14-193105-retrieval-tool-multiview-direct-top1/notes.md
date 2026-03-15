# Retrieval Session: retrieval-tool-multiview-direct-top1

## Command

`ava session retrieval retrieval-tool-multiview-direct-top1 sessions/2026-03-14-184859-failure-patch-v2-rerun-11m-96/checkpoints/ava-11m-failure-patch-v2.pt corpora/tool_multiview_v1 --device cuda --max-new-tokens 48 --retrieval-top-k 1 --mode direct`

## Inputs

- Checkpoint: `sessions/2026-03-14-184859-failure-patch-v2-rerun-11m-96/checkpoints/ava-11m-failure-patch-v2.pt`
- Support corpus: `corpora/tool_multiview_v1`
- Support files: `3`
- Support examples: `97`
- Retrieval top_k: `1`
- Retrieval mode: `direct`
- Category gated: `True`

## Accuracy Delta

- Benchmark: `9/10` -> `10/10`
- Tool eval: `6/6` -> `5/6`
- Compliance: `4/4` -> `4/4`

## Changed Benchmark Rows

- [science] prompt=`What planet is known as the Red Planet?` baseline=`[car 493` -> retrieval=`Mars`

## Changed Tool Rows

- [no_tool] prompt=`What is the capital of France? Do not use the calculator tool. Reply with only the answer.` baseline=`Paris` -> retrieval=`calp t.`

## Changed Compliance Rows

- none

## Focus Prompt

- `What planet is known as the Red Planet?`
- Baseline trace: `results/focus_trace_baseline.json`
- Retrieval trace: `results/focus_trace_retrieval.json`

## Artifacts

- Baseline benchmark/tool/compliance under `results/`
- Retrieval benchmark/tool/compliance under `results/`
