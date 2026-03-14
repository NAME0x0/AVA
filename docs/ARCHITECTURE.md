# AVA Architecture

AVA is the product. This codebase is the research and training stack that builds it.

## Product Shape

AVA now has two layers of scope:

- near-term training target: language, math, science, coding, dependable tool use, and external memory on a 4 GB GPU
- long-term evaluation scaffold: multilingual transfer and multimodal reasoning once the base stack is stable

That distinction matters. The current model code is still text-first, but the benchmark and session system should already know how AVA will be judged later.

## Core Loop

1. Curate high-quality language, math, science, and code text.
2. Tokenize with a simple baseline tokenizer.
3. Train a compact GPT-2 style causal decoder.
4. Add compact tool traces for calculator use.
5. Add verifiable post-training on math, science, coding, and tool tasks.
6. Add compliance tuning for formatting obedience, refusal quality, and tool boundaries.
7. Add external memory writes gated by surprise.
8. Track every decision in a dated session.
9. Expand evaluation before expanding claims: multilingual first, multimodal after that.

## Modules

- `ava.tokenizer`
  A byte-level baseline tokenizer. It is intentionally simple and fully local.
- `ava.model`
  A minimal GPT-2 style decoder-only Transformer.
- `ava.train`
  Dry-run sizing plus a simple from-scratch training loop for raw text corpora.
- `ava.eval`
  Internal smoke eval plus compliance eval.
- `ava.benchmarks`
  External benchmark registry covering text, science, code, multilingual, multimodal, and agentic targets.
- `ava.tools`
  A safe calculator and compact tool trace formats.
- `ava.memory`
  A Titans-inspired external memory buffer with surprise-gated writes.
- `ava.research`
  Curated paper registry and AVA-specific hypotheses from current arXiv work.
- `ava.experiments`
  Parallel sweep utilities for budgets, tool protocols, memory thresholds, and test-time strategy.
- `ava.sessions`
  Session creation, result persistence, benchmark manifests, and next-step notes.

## Benchmark Layers

AVA uses three benchmark layers on purpose:

- internal smoke checks
  Tiny cheap tasks for debugging train/eval plumbing across language, math, science, code, and tool use.
- external near-term targets
  Text, science, and coding benchmarks that can influence training choices soon.
- external future targets
  Multilingual, vision, and agentic benchmarks that define the scaling roadmap before the model can run them all well.

This avoids a common failure mode: building a narrow stack that later has no honest path to broader reasoning or product evaluation.

## What “Titans” Means Here

The Titans paper motivates long-horizon memory through learned test-time updates. On this hardware budget AVA uses a lighter interpretation:

- write only when surprise is high
- retrieve compact summaries into the prompt
- keep the core model small

This is an engineering approximation, not a claim of literal infinite context.

## Product Strategy

- Train dense first.
- Keep the model small enough to fit comfortably on 4 GB VRAM.
- Use data quality and post-training to punch above parameter count.
- Measure compliance explicitly instead of assuming it emerges.
- Add science and coding benchmarks before making versatility claims.
- Add multilingual transfer benchmarks before making language-general claims.
- Add multimodal benchmarks before building a larger vision-language stack.
- Use tool calling and selective test-time compute before reaching for large model size.
- Treat aggressive quantization and BitNet-style ideas as deployment branches, not as the first training line.

## What AVA Is Not Claiming Yet

- The current text-first baseline is not already multimodal.
- The current stack does not prove instant zero-shot acquisition of unseen languages.
- A tiny 4 GB baseline will not beat frontier systems everywhere.

The scaffolding now supports broader scope, but the model still has to earn it through data, architecture, and evaluation.
