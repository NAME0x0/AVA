# Session Workflow

Every meaningful change starts in a session under `sessions/`. AVA also keeps an append-only activity ledger under `sessions/activity/` for CLI invocations, snapshots, and wrapped external commands.

## Bootstrap

```bash
ava session bootstrap karpathy-reset
```

This creates:

- `session.json`
- `results/budget_sweep.json`
- `results/prompt_protocol_sweep.json`
- `results/benchmark.json`
- `results/tool_benchmark.json`
- `results/compliance_benchmark.json`
- `results/benchmark_registry.json`
- `notes.md`

## HF Research Session

```bash
ava session hf-research hf-march-2026
```

This creates a reproducible paper-review packet:

- `results/papers.json`
- `results/hypotheses.json`
- `results/relevant_benchmarks.json`
- `results/recommendation.json`
- `results/benchmark.json`
- `results/tool_benchmark.json`
- `results/compliance_benchmark.json`
- `results/benchmark_registry.json`
- `notes.md`

## Inspection Session

```bash
ava inspect checkpoint path/to/checkpoint.pt --prompt "Question: Use the calculator tool for 144 / 12. Return a compact calculator trace followed by the final answer.\nAnswer: "
```

This creates an inspectable activation packet:

- `results/trace.json`
- `artifacts/command.txt`
- `artifacts/prompt.txt`
- `notes.md`

The trace records:

- chosen token at each generation step
- top next-token logits
- per-layer top MLP neuron activations for the last token
- per-head attention summaries for the last token
## Activity Ledger

```bash
ava activity snapshot --label before-tokenizer-change
ava activity run -- python -m pytest -q
```

This creates:

- `activity/YYYY-MM-DD.jsonl`
- `activity/snapshots/<timestamp>-<label>.json`
- `activity/commands/<timestamp>-<label>/command.txt`
- `activity/commands/<timestamp>-<label>/stdout.txt`
- `activity/commands/<timestamp>-<label>/stderr.txt`
- `activity/commands/<timestamp>-<label>/result.json`

## Proof-of-Concept Capture

This is not implemented as a first-class AVA command yet, but it is the current proof-of-concept plan for human-facing evidence:

- Record terminal-native sessions with `asciinema` as the primary replay artifact.
- Keep the `.cast` file as the authoritative visual replay because it is smaller and more faithful than a raw screen video for terminal work.
- Optionally render the cast to `webm` or `mp4` with `ffmpeg` for easier sharing.
- Treat structured AVA artifacts as the actual source of truth: session JSON, notes, hashes, stdout, stderr, checkpoints, and the activity ledger.
- If this is implemented later, store casts under `sessions/<session>/artifacts/proof/` and log their hashes in the session notes.

## Transparent Training Session

```bash
ava session train tool-sft-smoke configs/experiments/ava-11m-tool-sft.yaml corpora/tool_sft --max-steps 128
```

This creates a fully inspectable packet:

- `results/environment.json`
- `results/corpus.json`
- `results/budget.json`
- `results/config.json`
- `results/training.json`
- `results/evaluation.json`
- `results/tool_eval.json`
- `results/compliance.json`
- `results/benchmark.json`
- `results/tool_benchmark.json`
- `results/compliance_benchmark.json`
- `results/benchmark_registry.json`
- `results/next_actions.json`
- `artifacts/command.txt`
- `artifacts/<config>.yaml`
- `checkpoints/<model>.pt`
- `notes.md`

## Transparency Rules

- Record the exact command line.
- Append every `ava` command to `sessions/activity/YYYY-MM-DD.jsonl`.
- Snapshot the exact config file used.
- Record Python, torch, CUDA, GPU, and repo state.
- Record corpus file paths, sizes, and hashes.
- Save raw loss history, not just a final score.
- Save per-task benchmark outputs, not just a single aggregate metric.
- Save separate tool-behavior outputs for trace generation, no-tool abstention, and tool-boundary behavior.
- Save inspection traces when interpretability or debugging is the goal.
- Save separate compliance outputs for formatting obedience, refusal behavior, and tool-boundary policy.
- Save the external benchmark registry so future scope does not disappear from the session record.
- Save checkpoint hashes so later analysis can point to the exact artifact.
- End each session with one concrete next move.
- Use `ava activity snapshot` before or after major edits when you need an explicit repo-change checkpoint outside a session packet.
- Use `ava activity run -- ...` for tests, evals, or auxiliary commands that should have archived stdout and stderr.

## Benchmark Layers

AVA uses four benchmark layers:

1. Smoke eval: tiny internal checks for train or eval plumbing across language, math, science, coding, and tool use.
2. Tool eval: compact tool-behavior checks for trace generation, no-tool abstention, and tool boundaries.
3. Compliance eval: format obedience, refusal quality, and tool-policy behavior.
4. Registry eval: the larger text, science, coding, multilingual, multimodal, and agentic portfolio AVA should eventually target.

## Parallel Sweeps

The bootstrap session runs two cheap sweeps in parallel:

- model budget sweep
- tool prompt protocol sweep

These are intended to reject bad ideas before spending GPU time.

## Recommended Order

1. Budget sweep
2. Tokenizer experiment
3. Small pretraining run
4. Tool-use distillation
5. Compact tool curriculum build-out
6. Compliance tuning and evaluation
7. Multilingual transfer evaluation
8. Planning benchmark and skill instrumentation
9. Memory ablation
10. Multimodal scaffold work
11. Broader language, math, science, and coding pretraining run
12. Verifiable post-training
13. Tool RL and later RL infrastructure




