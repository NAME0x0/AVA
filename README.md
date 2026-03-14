# AVA

AVA is the product.

This repository is the research and engineering stack used to build AVA into a compact assistant that fits on a 4 GB VRAM GPU and improves through data quality, tool use, and careful evaluation. Near-term training remains text-first, but the scaffold now targets a broader useful model: language, math, science, coding, multilingual transfer, agentic planning, and later multimodal reasoning.

## Product Goal

AVA is being rebuilt as a compact general-purpose model for:

- everyday language tasks
- math reasoning
- science reasoning
- practical coding assistance
- reliable calculator-style tool use
- strong instruction following, refusal quality, and tool-boundary compliance
- multilingual transfer evaluation
- long-horizon planning and skill use
- future multimodal reasoning evaluation
- long-horizon interaction through external memory

The ambition is still to build a model family that scales well, but the repo is explicit about current constraints: a 4 GB local-first training line cannot honestly target frontier-everything performance today. The right move is to set the scaffolding now so later scale-ups are measured on the correct benchmark mix.

## Repo Role

The repo is the research program behind AVA:

- minimal GPT-2 style training code
- paper-backed experiment sessions
- append-only activity ledger for commands, snapshots, and wrapped test runs
- tool-use, memory, planning, compliance, and activation-trace inspection
- benchmark registry for text, science, coding, multilingual, multimodal, and agentic evaluation
- 4 GB VRAM budget checks
- fast tests for each new idea

## Design Rules

- `4 GB VRAM first`: every baseline must fit the RTX A2000 Laptop GPU budget.
- `Text-first, multimodal-ready`: near-term training stays text-centric, but benchmark and session scaffolding must support later vision-language work.
- `Versatility is earned in stages`: language, math, science, and coding should all appear in the benchmark and data plans before scale claims are made.
- `Tool use over raw params`: calculator and structured tool traces are core training targets.
- `Planning claims require benchmarks`: DeepPlanning-class tasks belong in the registry before AVA markets agentic planning.
- `Compliance must be measured`: formatting obedience, safe refusals, and tool-policy behavior must show up in session artifacts.
- `Benchmark scaffolding must stay ahead of model scale`: the registry should already include multilingual, multimodal, science, coding, and planning targets before the model can run them all well.
- `Session logs are the source of truth`: every sweep, decision, and inspection trace lives in `sessions/`.
- `Infinite context is not literal`: AVA uses an external memory path, not a magic unbounded Transformer context window.
- `Reasoning should exist, but be cheap`: compact scratchpads, verifiable rewards, and selective test-time scaling matter more than verbose chain-of-thought everywhere.
- `Sparse MoE is a branch, not the mainline`: use it later for ablations or deployment experiments, not as the primary answer to the 4 GB product constraint.

## Current Repo Contents

- `src/ava/`
  GPT-2 style model code, tokenizer, benchmark registry, tool and memory modules, session orchestration, and training utilities.
- `configs/`
  Baseline and sweep configs sized for a 4 GB workflow.
- `docs/`
  Architecture, benchmark strategy, experiment workflow, and data strategy notes.
- `sessions/`
  Generated on demand when session commands run; contains experiment packets, notes, next actions, and the `activity/` audit ledger.
- `tests/`
  Fast validation for the current research core.


## Quick Start

```bash
python -m pip install -e .[dev]
ava session bootstrap karpathy-reset
ava session sota arxiv-march-2026
ava session hf-research hf-march-2026
ava session moe-feasibility moe-march-2026
ava benchmark registry --modality code
ava benchmark registry --stage agentic
ava train dry-run configs/base.yaml
ava session train tool-sft-smoke configs/experiments/ava-11m-tool-sft.yaml corpora/tool_sft --max-steps 128
ava inspect checkpoint path/to/checkpoint.pt --prompt "Question: Use the calculator tool for 144 / 12. Return a compact calculator trace followed by the final answer.\nAnswer: "
ava activity snapshot --label before-change
ava activity run -- python -m pytest -q
```

The session commands write paper-backed experiment packets. `ava benchmark registry` prints the external benchmark scaffold that AVA should eventually target. Every `ava` command also appends an event to `sessions/activity/YYYY-MM-DD.jsonl`. Use `ava activity snapshot` to capture repo-change state and `ava activity run -- ...` to log external test or eval commands with stdout and stderr artifacts.

## Evaluation Stack

AVA now has four evaluation layers plus an inspection path:

1. Internal smoke benchmarks for fast iteration.
2. Tool-behavior benchmarks for trace generation, no-tool abstention, and boundary refusals.
3. Compliance benchmarks for format obedience, refusal quality, and tool-boundary behavior.
4. External benchmark registry for future text, science, coding, multilingual, multimodal, and agentic evaluation.

For inspection, AVA can also emit per-step traces with top logits, attention summaries, and top MLP neuron activations for a concrete prompt and checkpoint.

See `docs/BENCHMARKS.md` and `docs/EXPERIMENTS.md`.

## Activity Ledger

AVA now keeps an append-only activity ledger under `sessions/activity/`:

- `YYYY-MM-DD.jsonl` records every `ava` CLI invocation and session lifecycle event.
- `snapshots/` stores explicit repo-state snapshots taken with `ava activity snapshot`.
- `commands/` stores wrapped external command artifacts such as `stdout.txt`, `stderr.txt`, and `result.json` from `ava activity run -- ...`.
- Future proof-of-concept: terminal-first evidence capture with `asciinema`, with optional `ffmpeg` rendering for shareable video exports.

This does not magically intercept arbitrary shell edits made outside AVA, so repo-change transparency still relies on git state plus explicit snapshots when needed.

## First Practical Plan

1. Stabilize the dense text-first baseline on language, math, science, and coding.
2. Improve tool-use and compliance behavior with compact supervised data.
3. Build a stronger compact tool curriculum before any tool RL branch.
4. Add explicit planning and skill evaluation before any agentic product claims.
5. Replace the byte tokenizer with a compact BPE or SentencePiece branch.
6. Add multilingual transfer eval before claiming language generalization.
7. Add stronger science and coding eval before scaling architecture claims.
8. Add multimodal benchmark scaffolding before building a vision encoder path.
9. Explore compression, tool RL, and tiny MoE branches only after the dense teacher is stable.

## Source-Backed Direction

The current direction is anchored in:

- GPT-2 and Karpathy-style minimal training loops for clean small-model iteration.
- LIMO and Phi/Textbooks-style evidence that data quality can matter more than brute-force scale.
- DeepSeek-R1 and DAPO-style evidence that verifiable RL can improve reasoning.
- Toolformer and ToolACE-R-style evidence that small numbers of tool demonstrations plus iterative refinement can materially improve tool use.
- ICRL, ToRL, and ReTool-style evidence that tool RL is promising after compact tool supervision already works.
- DeepPlanning and SkillNet-style evidence that planning and skill claims should be benchmarked explicitly.
- Penguin-VL and InternVL-U as future multimodal references, with compact VLM design favored for AVA's first vision branch.
- s1-style evidence that selective test-time scaling can help if it is tightly budgeted.
- Titans-style external memory for long-horizon product behavior.
- Common modern evaluation sets such as MMLU-Pro, GPQA, HumanEval+, BFCL, DeepPlanning, MMMU, MathVista, M-IFEval, and FLORES-class transfer checks.

See `docs/ARCHITECTURE.md`, `docs/BENCHMARKS.md`, `docs/RESEARCH_ROADMAP.md`, and `docs/EXPERIMENTS.md`.






