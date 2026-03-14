# Benchmark Strategy

AVA uses two different concepts on purpose:

- `smoke benchmarks`
  Tiny internal checks used on every short training run, including a dedicated tool-behavior packet.
- `external benchmark registry`
  The larger benchmark portfolio AVA should eventually target.

Do not confuse the two. The smoke suite is for debugging. The registry defines the longer-term product and research contract.

## Principles

- Add the benchmark before making the claim.
- Separate near-term training targets from future scale targets.
- Keep science, coding, multilingual, multimodal, and agentic evaluation visible even while the base model is text-first.
- Use benchmarks that are standard enough to compare against other labs and open models.
- Prefer benchmarks with public evaluation code or clear official references.

## Foundation Text And Reasoning Benchmarks

These matter earliest for AVA's current stack:

- `IFEval`
  Instruction following and constrained formatting.
- `MMLU-Pro`
  Broad knowledge and reasoning.
- `GPQA Diamond`
  Hard science reasoning.
- `MATH`
  Competition-style symbolic math.
- `BFCL`
  Function or tool calling correctness.

## Coding Benchmarks

These make coding part of the real scope instead of a hand-wavy future promise:

- `HumanEval+`
  Function synthesis with stronger tests.
- `MBPP+`
  Small practical Python programming tasks.
- `LiveCodeBench`
  Contamination-aware coding evaluation for later scale stages.

## Multilingual Transfer Benchmarks

These are the right scaffold for strong language generalization. They measure transfer. They do not prove magical instant language acquisition.

- `M-IFEval`
  Multilingual instruction following.
- `MGSM`
  Multilingual grade-school math.
- `MMLU-ProX`
  Multilingual reasoning.
- `Belebele`
  Multilingual reading comprehension.
- `FLORES-200`
  Translation and transfer.

## Multimodal Vision Benchmarks

These define the future vision-language branch:

- `MMMU`
  Broad expert-level multimodal reasoning.
- `MathVista`
  Visual math reasoning.
- `ScienceQA`
  Multimodal science reasoning.
- `AI2D`
  Diagram understanding.
- `ChartQA`
  Chart and plot reasoning.
- `DocVQA`
  Document image understanding.

## Agentic Planning Benchmarks

These should exist in the registry before AVA makes strong planning claims:

- `DeepPlanning`
  Long-horizon planning with verifiable constraints.
- `tau2-bench`
  Multi-turn, policy-constrained agentic tool use.

## Later Scale Benchmarks

These should be in the registry now, but not allowed to distort the first 4 GB training recipe:

- `LiveCodeBench`
  Harder contamination-aware coding evaluation.
- `SWE-bench Verified`
  Software issue resolution.

## Session Contract

Every session packet should contain:

- `results/benchmark.json`
  Internal smoke tasks.
- `results/tool_benchmark.json`
  Tool-behavior smoke tasks.
- `results/compliance_benchmark.json`
  Compliance smoke tasks.
- `results/benchmark_registry.json`
  External benchmark portfolio.

HF research sessions may also include:

- `results/relevant_benchmarks.json`
  The subset of the registry that a paper review changes immediately.

That way the repo stays honest about three things at once:

- what was actually evaluated today
- what compliance means today
- what the future model family will eventually be judged on

