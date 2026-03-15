# AVA

AVA is a compact AI model project built under hard constraints: one 4 GB VRAM GPU, limited data, and no large-cluster research budget.

This repository is the engineering and experimentation stack behind AVA. It is where the model, training loop, datasets, evaluation harnesses, and session logs live. The goal is not to imitate frontier labs with brute force. The goal is to find a small-model path that wins through better data, tighter experiments, stronger tool use, and a more transparent training process.

## Current Verified State

As of March 15, 2026, the strongest verified AVA result in this repo is a small-model plus explicit-memory system, not a raw-weight-only win.

- Base checkpoint: `sessions/2026-03-14-184859-failure-patch-v2-rerun-11m-96/checkpoints/ava-11m-failure-patch-v2.pt`
- Harder held-out stress suite: `87/87` with nearest-memory routing using the `21`-example bank in `corpora/tool_memory_minimal_v3`
- Same stress suite baseline without memory: `17/87`
- Harder held-out expanded suite: `40/40` with the `23`-example bank in `corpora/tool_repair_nano_v1`

The important result is not just that AVA improves with retrieval. It is that bank quality matters more than bank size. In the current stack, a carefully shaped external memory bank is the main capability multiplier for the `11M` model.

Representative tracked sessions:
- `sessions/2026-03-14-202211-stress-tool-minimal-v3-rerun/`
- `sessions/2026-03-14-202119-expanded-transfer-tool-repair-nano-v1/`
- `sessions/2026-03-14-184859-failure-patch-v2-rerun-11m-96/`

## What This Repository Is

AVA is the product. This codebase is the machinery used to build it.

Today the stack is text-first and focused on:
- language
- math
- science
- coding
- tool use
- compliance and instruction following
- planning and memory scaffolding
- multilingual and multimodal evaluation scaffolding

The long-term ambition is broader, but the repo is explicit about current reality: AVA has to earn capability through measured iteration, not marketing claims.

## Why AVA Exists

Most small-model projects fail in one of two ways: they either stay toy-sized and never become useful, or they copy large-model research directions that do not fit local hardware.

AVA takes a different route:
- keep the stack simple enough to inspect end to end
- run short, well-documented experiment cycles
- bias toward methods that improve quality per token, per parameter, and per unit of compute
- treat tool use, evaluation, and transparency as first-class parts of the model

## Current Approach

The active research line is a compact GPT-2 style decoder with:
- tight 4 GB VRAM budget checks
- supervised and synthetic training curricula
- compact calculator-style tool protocols
- compliance and tool-boundary evaluation
- session-based experiment logging
- checkpoint inspection with activation traces
- an append-only activity ledger for AVA-managed commands

Recent work in this repo has focused on improving tool-use behavior, tokenizer experiments, warm-start curriculum stages, and making every experiment easier to audit.

## Repository Layout

- `src/ava/` — model code, tokenizers, training loop, evaluation, sessions, tools, memory, and inspection
- `configs/` — baseline and experiment configs sized for local hardware
- `corpora/` — tracked training and synthetic experiment corpora
- `docs/` — architecture, data, benchmark, experiment, and roadmap notes
- `sessions/` — generated experiment packets, metrics, notes, and activity logs
- `tests/` — fast validation for the research core

## Quick Start

Install the package and dev tools:

```bash
python -m pip install -e .[dev]
```

Add training dependencies when needed:

```bash
python -m pip install -e .[train]
```

Inspect the benchmark scaffold:

```bash
ava benchmark registry --modality code
ava benchmark registry --stage agentic
```

Run a dry-run budget check:

```bash
ava train dry-run configs/base.yaml
```

Start a tracked training session:

```bash
ava session train tool-sft-smoke configs/experiments/ava-11m-tool-sft.yaml corpora/tool_sft --max-steps 128
```

Run the test suite with archived command output:

```bash
ava activity run -- python -m pytest -q
```

Replay the current stress result on the best 11M checkpoint:

```bash
ava session memory-transfer stress-tool-minimal-v3-rerun sessions/2026-03-14-184859-failure-patch-v2-rerun-11m-96/checkpoints/ava-11m-failure-patch-v2.pt corpora/tool_memory_minimal_v3 --device cuda --nearest-threshold 0.45 --nearest-margin 0.0 --suite stress
```

## How Experiments Work

AVA is session-first. Meaningful research work is recorded under `sessions/` with:
- config snapshots
- corpus manifests
- environment metadata
- training curves and evaluation outputs
- checkpoint artifacts
- written notes and next-step decisions

The intent is simple: no silent fallbacks, no hidden training state, and no black-box project history.

## Evaluation Philosophy

AVA uses multiple evaluation layers on purpose.

- Smoke evals catch broken training and prompt plumbing quickly.
- Tool evals measure trace generation, abstention, and boundary behavior.
- Compliance evals measure formatting obedience, refusal quality, and policy adherence.
- The benchmark registry tracks the larger target surface for science, coding, multilingual, multimodal, and agentic work.

This repo treats benchmark scaffolding as part of model design. If AVA is supposed to grow into a broader product, the evaluation surface has to be explicit before the scale-up happens.

## Transparency

Transparency is a design constraint, not a nice-to-have.

Every serious experiment should leave behind enough evidence for someone else to answer:
- what changed
- why it changed
- what was run
- what improved
- what failed
- what should happen next

For command-level provenance, AVA also keeps an activity ledger under `sessions/activity/` for CLI invocations, snapshots, and wrapped external commands.

## Further Reading

- [Architecture](docs/ARCHITECTURE.md)
- [Benchmarks](docs/BENCHMARKS.md)
- [Data Strategy](docs/DATA.md)
- [Experiment Workflow](docs/EXPERIMENTS.md)
- [Research Roadmap](docs/RESEARCH_ROADMAP.md)
- [Teacher Distillation SOP](docs/TEACHER_DISTILLATION_SOP.md)

## Status

AVA is still in active experimentation. The stack is intentionally compact, the claims are intentionally conservative, and the repo is designed so that each iteration leaves behind a usable research trail.


