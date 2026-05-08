# Experiments — what we tried, what shipped, what's next

Status legend: ✅ done · 📦 released · ⏸ paused · 🚧 active · ❌ failed

```
Exp 1-3 ✅          Exp 4 📦              Exp 5 ⏸                Exp 6 🚧
Scratch baseline →  Qwen 3.5 QLoRA →    Gemma 4 inference  →   AVA v3 ternary MoE
14M params          AVA v1 → AVA v2     26B / E4B / E2B        + MCP tools
24% ARC ceiling     82% ARC ceiling     ~6.4 tok/s peak        scaffolding (P0)
```

The repo keeps every branch. Closed branches stay so the failure modes remain visible. Below: each experiment, what it set out to prove, what it actually showed, and why we did or didn't keep building on it.

## Exp 1-3 — Scratch AVA systems · ✅ done

**Goal**: prove useful AI behavior can emerge on a tiny local model trained from scratch.

**Built**: 11M-99M parameter scratch configs, byte-level tokenizer experiments, retrieval ensembles, memory-transfer flows, tool/compliance scaffolds.

**Best results**:
- ~24% ARC-Challenge (random = 25%) for the bare scratch baseline
- 91/299 on ARC with a sparse science retrieval ensemble
- 87/87 on internal memory-transfer stress suites
- Qwen tokenizer cuts tokens to ~0.24× the byte baseline on the largest tracked corpus

**Why we stopped**: 14M parameters is too small to store enough world knowledge for ARC-style reasoning. The ceiling is data, not architecture. The retrieval, memory, and tooling code is retained — those ideas survive into v3.

**Files**: `src/ava/`, `corpora/`, `tests/test_*.py`.

## Exp 4 / AVA v1 — first QLoRA attempt · ✅ done

**Goal**: prove QLoRA works on 4 GB VRAM with a real open base model.

**Built**: Qwen3.5-2B NF4 loading, LoRA pipeline, evaluation harness, agentic harness with calculator/python/memory tools.

**Best result**: 66% ARC-C (matches base — i.e. zero lift), 40% GSM8K on the fast v1 run with a 5K corpus.

**Lesson**: 5K examples in 1 epoch isn't enough corpus to move ARC at all. The corpus has to scale.

**Files**: `experiments/exp4_finetune/scripts/finetune_qlora.py`, `experiments/exp4_finetune/EXPERIMENT_LOG.md`.

## Exp 4 / AVA v2 — released · 📦

**Goal**: turn AVA into a publishable local assistant with a real eval.

**Built**: 20K curated corpus (math + science + reasoning + tools), Triton/SDPA training speedup (10.7× per-step on Windows), full 17-benchmark eval pipeline, HF model card, GGUF export path.

**Best results** (full 17-benchmark eval, 16,872 tasks, Q8_0 GGUF):
- 82.0% ARC-Challenge (full 1,172) — ahead of Llama 3.2 3B-Instruct (78.6%)
- 92.0% ARC-Easy (2,376)
- 75.9% PIQA, 75.0% BoolQ
- 59.2% MMLU 5-shot (14,042)
- 35.3% / 44.0% GSM8K (greedy / k=5 self-consistency)
- 30.9% MMLU-Pro, 18.8% MATH-500, 35.7% MBPP+, 19.5% HumanEval+
- 42 MB adapter, 1.81 GB peak training VRAM, 100 minute wall

**What's released**:
- LoRA adapter weights at [`huggingface.co/NAME0x0/AVA-v2`](https://huggingface.co/NAME0x0/AVA-v2)
- Pre-quantized GGUFs at [`huggingface.co/NAME0x0/AVA-v2-GGUF`](https://huggingface.co/NAME0x0/AVA-v2-GGUF)
- Full eval report at [`experiments/exp4_finetune/eval_v2/RESULTS_REPORT_V2_FULL.md`](../experiments/exp4_finetune/eval_v2/RESULTS_REPORT_V2_FULL.md)

**What remains for v2 alone**: longer context, stronger math, RL post-training, better tool specialization, student distillation. Most of these roll forward into v3 instead of getting bolted onto v2.

**Files**: `experiments/exp4_finetune/scripts/finetune_v2_full.py`, `experiments/exp4_finetune/eval_v2/`.

## Exp 5 — Gemma 4 local inference research · ⏸ paused

**Goal**: make Gemma 4 (26B-A4B / E4B / E2B) run usefully on 4 GB VRAM + 32 GB RAM, exploring streaming int4 loaders, TurboQuant bit-packing, MoE offload, YaRN long-context, and a two-tier fast/deep runtime.

**Best measured results**:

| Track | Role | Practical context | Best decode | Notes |
|---|---|---|---|---|
| `26B-A4B` | feasibility | 256K → 1M experimental | ~0.50 tok/s warm | proves fit, not speed |
| `E4B` deep | reasoning | 512K practical | ~0.85 tok/s | dense deep-path |
| `E2B` fast (Transformers) | chat | 512K | ~2.13 tok/s | mid-tier fallback |
| `E2B` fast (llama.cpp) | chat | 512K | **~6.39 tok/s decode, ~17.74 tok/s total** | best fast path |

**Why paused**: even the best fast-path numbers (~6.4 tok/s) didn't justify a default switch from AVA v2 + ongoing v3 work. The two-tier runtime (`quick:` / `deep:` / `reason:`) was working but the deep-branch escalation cost was still too high. The v3 plan reuses Exp 5's streaming int4 loader and TurboQuant cache, so the engineering doesn't go to waste.

**Files retained**: `experiments/exp5_gemma4/` (not deleted, not active). See `experiments/exp5_gemma4/RESULTS.md` for the detailed write-up.

## Exp 6 / AVA v3 — ternary MoE student + MCP tools · 🚧 active

**Goal**: push capacity-per-VRAM by ~5× and fix v2's structural weaknesses (math, tool routing, instruction following).

**Plan**:
- **Teacher**: Qwen 3.6 35B-A3B (active 3B / total 35B MoE).
- **Student**: 6-8B ternary MoE — ternary FFN experts (~16× smaller than BF16) + BF16 router + 1 BF16 shared expert (the "MoTE" pattern).
- **Attention**: native 3:1 hybrid Gated DeltaNet (linear-time decode) instead of full attention.
- **Distillation**: BitDistiller 3-stage QAT — BF16 warmup → mixed-precision intermediate → ternary QAT distillation from teacher logits.
- **Tools**: MCP-based via FastMCP 3.0 + XGrammar constrained decoding. The model learns to call external tools through a protocol, not by memorizing JSON syntax inside SFT.
- **Post-training**: SFT then DPO on a curated mix.

**Status**: P0 (scaffolding) — design doc, configs, engine stubs, MCP catalog, scripts in place. P1-P11 (download teacher → ternary linear → MoTE-FFN wiring → BF16 warmup → QAT → SFT/DPO → MCP server → GGUF export → full eval) are queued.

**Files**: `experiments/exp6_v3/DESIGN.md`, `experiments/exp6_v3/README.md`, `experiments/exp6_v3/configs/`, `experiments/exp6_v3/engine/`, `experiments/exp6_v3/mcp/`.

## Why each branch stays in the repo

We don't delete failed or paused branches. The repo's value is partly **the trail of decisions** — if you can see why we stopped scratch training, why v1's 5K corpus didn't move ARC, and why Gemma 4 didn't hit our latency bar, you can build on it without re-running those mistakes.

If you only want the released model, use [QUICKSTART.md](QUICKSTART.md). Everything else is optional reading.
