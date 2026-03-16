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
4. Use the decoder as the control branch while AVA-v2 tests a cleaner recurrent-depth student.
5. Add compact tool traces for calculator use.
6. Add verifiable post-training on math, science, coding, and tool tasks.
7. Add compliance tuning for formatting obedience, refusal quality, and tool boundaries.
8. Add external memory writes gated by surprise.
9. Track every decision in a dated session.
10. Expand evaluation before expanding claims: multilingual first, multimodal after that.

## Modules

- `ava.tokenizer`
  A byte-level baseline tokenizer plus artifact-backed open-tokenizer imports. The byte path is still the stable control; the best compression result so far comes from a Qwen-family tokenizer.
- `ava.model`
  A minimal GPT-2 style decoder-only Transformer plus an early looped branch. The current looped implementation is only a scaffold, not yet a faithful recurrent-depth architecture.
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


## AVA-v2 Direction

The next serious architecture branch is no longer just “scale the same stack a bit.”

The current evidence points toward three coordinated upgrades:

- open tokenizer first
  The best measured tokenizer on AVA's real corpora is a Qwen-family tokenizer, cutting token counts to roughly one quarter of the byte baseline on the largest tracked corpus.
- recurrent-depth student core
  The strongest recent architecture lead for “more reasoning without more parameters” is latent recurrent depth: a prelude, a shared recurrent block, and a coda. AVA's current looped path is only a rough placeholder and should not be confused with that paper's design.
- structured retrieval and memory
  AVA's public benchmark wins are coming faster from better retrieval than from checkpoint surgery. The next retrieval target is not another flat nearest-neighbor tweak, but a more structured memory layer that can combine sparse, dense, and graph-style propagation.

## Architecture Lessons So Far

- Late tokenizer migration is not working on the current byte-trained checkpoint. Tokenizer improvement belongs early in AVA-v2, not late in AVA-v1.
- Naive loop repetition is not enough. The failed looped runs show that AVA needs a cleaner recurrent design and curriculum if loops are going to matter.
- Retrieval quality matters more than small weight patches on public science MCQ. That is why the best current public path is a routed sparse+dense system rather than a newly fine-tuned checkpoint.
