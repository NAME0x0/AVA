# Why v3 — One-Page Motivation

## What AVA v2 cannot do

AVA v2 (Qwen 3.5 2B + QLoRA, 42 MB adapter) is the strongest 2 B-class model on ARC-Challenge that any small lab has shipped on a 4 GB laptop. The full 17-benchmark / 16 872-task eval at Q8_0 GGUF is in [`docs/RESULTS.md`](../RESULTS.md). Headline:

| Strength | v2 score | The number to beat |
|---|---|---|
| ARC-Challenge | 82.0 % | Phi-4-mini 3.8 B (83.7 %) |
| MMLU | 59.2 % | Mistral 7B v0.2 (60.1 %) |
| ARC-Easy | 92.0 % | nothing at the 2 B scale |

But the same eval exposed five hard ceilings that no amount of further QLoRA fine-tuning on Qwen 3.5 2B will fix:

| Weakness | v2 measured | Why fine-tuning alone cannot fix it |
|---|---|---|
| Math (GSM8K greedy) | 35.3 % | Base model has weak symbolic reasoning; the 384-token training cap truncates CoT |
| Math (MATH-500) | 18.8 % | Same; competition math needs multi-step latent reasoning |
| Tool routing (agentic GSM8K) | 0.6 % call rate | 55 tool examples vs 20 K math examples — corpus bias is permanent |
| Multilingual (MGSM fr) | 28.4 % | Base tokenizer + 2 B capacity ceiling |
| Long context | n/a, 384-token train cap | Quadratic attention + no long-ctx data |

v2 also has structural limits the eval did not stress: the 2 B dense base means there is no head-room for new knowledge without forgetting old knowledge, and the 4 GB residency budget already forces 4-bit serving.

## The 2026 research that makes v3 possible

Between January and May 2026 three independent groups published the missing pieces:

1. **HRM-Text (Sapient Intelligence, May 2026).** A 1 B brain-inspired reasoner with **two interdependent recurrent stacks** — a high-level slow planner and a low-level fast computer — that performs multi-step reasoning in latent space, **without chain-of-thought tokens**. Trained on ~40 B effective tokens (1000× less than typical pre-training). Independent verification: **MMLU 60.7, ARC-C 81.9, MATH 56.2, DROP 82.2**. The MATH number is the smoking gun: it is **+37 pp over AVA v2 at half the parameter count**.
2. **Mamba-3 (ICLR 2026).** Generation-3 subquadratic attention with **complex-valued state updates** and a **MIMO formulation**. At 1.5 B scale, **+1.8 pp over Gated DeltaNet** on downstream tasks, with **half the hidden state size of Mamba-2**. Drop-in compatible with the FLA toolkit kernels already targeted by [`experiments/exp6_v3/`](../../experiments/exp6_v3/).
3. **PrismML Bonsai (Caltech, March–April 2026).** First commercially-viable 1-bit LLM family. **1.125 bpw** packed storage (1-bit weight ±1, 128 weights share one FP16 scale). Bonsai 8 B fits in **1.15 GB**. Ternary Bonsai (1.58-bit) follow-on family in 1.7 B / 4 B / 8 B sizes. Kernels merged into `llama.cpp` head and Apple's MLX.

Each one targets a different v2 ceiling. None of them is sufficient alone. **AVA v3 stacks all three** on top of the existing Qwen 3.6 35B-A3B teacher distillation plan.

## What v3 will deliver

In one line: **a model that wins on every benchmark v2 measured, runs faster, holds longer context, and uses tools natively — all on the same 4 GB laptop.**

Concretely (full target list in [`PERF_TARGETS.md`](PERF_TARGETS.md)):

- MMLU ≥ 70 % (v2: 59.2 %)
- GSM8K ≥ 70 % greedy (v2: 35.3 %)
- MATH-500 ≥ 50 % (v2: 18.8 %)
- ARC-C ≥ 85 % (v2: 82.0 %)
- HumanEval+ ≥ 50 % (v2: 19.5 %)
- IFEval ≥ 65 % (v2: 31.6 %)
- BFCL v3 tool acc ≥ 70 % (v2: not in eval)
- RULER 128 K recall ≥ 90 % (v2: untrained beyond 384 tokens)
- Decode throughput ≥ 30 tok/s on RTX A2000 (v2: ~45 tok/s at 1.4 GB)
- Resident GPU memory ≤ 3.6 GB at full context (v2: ~1.4 GB at 8 K ctx)

If any single target is missed, the model does not ship as v3.

## What v3 is not

- Not a fine-tune. It is a new student model trained from scratch on warm-started weights.
- Not a frontier-scale pretraining run. It rides on a 35 B teacher, not on its own trillions of tokens.
- Not a cloud-only path. Every phase up to stage 2 distillation must be reproducible on the same laptop; stage 2 may rent A100s for ~50–90 wall-days but the resulting weights run unchanged on 4 GB.
- Not closed weights. Same licensing posture as v2.

Continue to [ARCHITECTURE_V3.md](ARCHITECTURE_V3.md) for the block-level design.
