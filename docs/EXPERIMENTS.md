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

**Goal**: strict dominance over v2 on every benchmark v2 measured, plus native 32 K context (256 K via YaRN, 1 M target), MCP tool routing, and ~13× capacity-per-VRAM. The architecture stacks three 2026 research lines on top of the original distillation backbone.

**Plan**:
- **Teacher**: Qwen 3.6 35B-A3B (active 3B / total 35B MoE).
- **Student**: ~3.26B ternary MoE — ternary routed FFN experts + BF16 router + ternary shared expert (the "MoTE" pattern, with PrismML-packed storage).
- **Reasoning**: **HRM-Text dual recurrence** — per-block H-module (slow planner) + L-module (fast computer, up to 6 iterations) + halting head. Latent multi-step reasoning in one forward pass, no CoT tokens emitted. Source: Sapient HRM-Text (arXiv:2506.21734, May 2026).
- **Attention**: **Mamba-3 (MIMO, complex SSM) in 3 : 1 hybrid** with gated softmax attention. +1.8 pp over Gated DeltaNet at 1.5 B; half the state size of Mamba-2 (ICLR 2026, arXiv:2603.15569). Gated DeltaNet kept as fallback.
- **Storage**: ternary experts trained with **group-256 QAT**, exported to **stock upstream llama.cpp `TQ1_0` (1.6875 bpw) / `TQ2_0` (2.0625 bpw)** + Q8_0 embeddings — zero fork dependency. PrismML Bonsai (Mar–Apr 2026) is the existence proof for sub-2-bpw quality; its `BB1_0` format stays a stretch lane until upstreamed.
- **Distillation**: BitDistiller 3-stage QAT — BF16 warmup → HRM halting curriculum (ACT ponder loss) → ternary QAT distillation from teacher logits.
- **Tools**: MCP-based via FastMCP 3.0 + XGrammar constrained decoding. The model learns to call external tools through a protocol, not by memorizing JSON syntax inside SFT.
- **Post-training**: SFT then DPO then tool-discrimination SFT (300 "when NOT to call" examples).

**Status**: P2 (student implementation) — design doc, configs, MCP catalog in place; engine modules **implemented June 2026** (group-256 ternary QAT linear, MoTE FFN, Mamba-3 reference mixer + FLA wrapper, HRM refinement recurrence with convergence-aware halting and latent-restart escape, full 3.24 B-param `AVAv3ForCausalLM` assembly, 8 passing CPU smoke tests). P1 (teacher fetch) and the rest of P2 (sparse-upcycle init) next; P3-P12 (BF16 warmup → HRM halting curriculum → ternary QAT distillation → SFT / DPO / tool-SFT → TQ packing → MCP wiring → full eval → release) queued.

**Files**: `experiments/exp6_v3/DESIGN.md`, `experiments/exp6_v3/README.md`, `experiments/exp6_v3/configs/`, `experiments/exp6_v3/engine/`, `experiments/exp6_v3/mcp/`, and the full v3 doc set under [`docs/v3/`](v3/INDEX.md).

## Why each branch stays in the repo

We don't delete failed or paused branches. The repo's value is partly **the trail of decisions** — if you can see why we stopped scratch training, why v1's 5K corpus didn't move ARC, and why Gemma 4 didn't hit our latency bar, you can build on it without re-running those mistakes.

If you only want the released model, use [QUICKSTART.md](QUICKSTART.md). Everything else is optional reading.
