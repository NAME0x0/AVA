# AVA v3 — Architecture

> **Companions:** [HRM_TEXT.md](HRM_TEXT.md) · [SUBQUADRATIC.md](SUBQUADRATIC.md) · [PRISMML.md](PRISMML.md). This document specifies the block diagram, dimensions, and memory math. Each cited mechanism is explained in detail in the companion file.

The v3 student is a single network that wears three hats at once:

1. A **ternary MoE student** distilled from the Qwen 3.6 35B-A3B MoE teacher (already established in [`experiments/exp6_v3/DESIGN.md`](../../experiments/exp6_v3/DESIGN.md)).
2. A **HRM-style dual-recurrent reasoner**: the top half of each block is a "high-level" planner that runs slowly, the bottom half is a "low-level" computer that runs at every micro-step inside the same forward pass.
3. A **Mamba-3 / Gated DeltaNet hybrid** for subquadratic context handling, deployed in a **3:1 ratio** with a gated softmax-attention layer.

The packed binary for serving uses **stock upstream GGUF ternary types** — `TQ1_0` (1.6875 bpw, smallest) and `TQ2_0` (2.0625 bpw, fastest) with group-256 scales baked into QAT — so the release runs on unmodified `llama.cpp`. PrismML's fork-only 1.125 bpw `BB1_0` is a stretch lane, adopted only if their upstream PR lands (June 2026 status check — see [PRISMML.md](PRISMML.md)).

---

## 1. Block diagram

```
                  ┌───────────────────────────────────────────────────────┐
                  │             input tokens (Qwen 3.6 tokenizer)         │
                  └─────────────────────────┬─────────────────────────────┘
                                            │ embeddings (BF16, tied)
                                            ▼
                    ╔═══════════════════════════════════════╗
                    ║   24 × HRM-augmented hybrid block     ║
                    ║                                       ║
                    ║   ┌─────────────────────────────────┐ ║
                    ║   │  H-module (slow, planning)      │ ║   runs once per
                    ║   │  - Gated Attention + RoPE       │ ║   N L-steps
                    ║   │  - MoE FFN (ternary, top-4/32)  │ ║   (N=2..6 adaptive)
                    ║   └────────────────┬────────────────┘ ║
                    ║                    │ z_H               ║
                    ║                    ▼                   ║
                    ║   ┌─────────────────────────────────┐ ║
                    ║   │  L-module (fast, computing)     │ ║   runs N times
                    ║   │  - Mamba-3 (MIMO, complex SSM)  │ ║   per H-step
                    ║   │  - MoE FFN (ternary, top-4/32)  │ ║
                    ║   │  - reads z_H as conditioning    │ ║
                    ║   └────────────────┬────────────────┘ ║
                    ║                    │ z_L               ║
                    ║                    ▼                   ║
                    ║          [ halting head, σ ]           ║   per-token
                    ║          continue L if σ < 0.5         ║   halting
                    ║          else emit token               ║
                    ╚═══════════════════════════════════════╝
                                            │
                                            ▼
                                  RMSNorm → LM head (BF16, tied)
```

24 blocks total. Inside each block the H-module is conventional (one attention sublayer + one MoE FFN). The L-module is the new piece: a Mamba-3 sublayer + MoE FFN, **conditioned on the latest H-output**, that is allowed to fire 2–6 times before the next H-step. Halting is decided by a per-token **convergence-aware** sigmoid head — input `[z_L ; z_H ; ‖z_L − z_L_prev‖]`, so the head sees the fixed-point residual — with a hard budget of 6 L-steps and an optional perturbed-restart escape in adaptive mode (June 2026 revision; rationale in [HRM_TEXT.md](HRM_TEXT.md) §1b).

This gives v3 the property that motivated HRM-Text: **latent multi-step reasoning inside a single forward pass, with no chain-of-thought tokens emitted.** The model still emits a normal token stream — the recurrence is invisible to the inference API.

---

## 2. Dimensions

| Field | Value | Rationale |
|---|---|---|
| Layers (blocks) | 24 | Down from 28 in original v3 design — saves ~1.2 GB; recurrence makes up for depth |
| Hidden size | 1792 | Unchanged from v3 P0 |
| Head dim | 128 | Unchanged |
| Heads (Gated Attention) | 14 | 14 × 128 = 1792 |
| KV heads (GQA) | 4 | 3.5× compression on KV cache |
| Layer ratio | 3 × L-block + 1 × H-block per 4 layers | Inherits Qwen 3.6 3:1 hybrid; H-blocks are softmax+RoPE, L-blocks are Mamba-3 |
| MoE FFN per layer | 32 routed (ternary) + 1 shared (BF16) | MoTE pattern |
| Top-k routing | 4 | Inherits Qwen 3.6 |
| Routed expert intermediate | 768 | Unchanged |
| Shared expert intermediate | 4096 | Unchanged |
| Mamba-3 state size | 64 per head (MIMO) | Half of Mamba-2's 128; relies on Mamba-3's perplexity-at-half-state result |
| Max H-steps per token | 1 (always) | One H pass per token |
| Max L-steps per token | 6 (adaptive halting, mean ~2.5 budget-target) | Reasoning slider — see [HRM_TEXT.md](HRM_TEXT.md) |
| RoPE base | 1 000 000 | Matches Qwen 3.6 |
| Context length (native) | 32 768 | Trained natively |
| Context length (YaRN extension) | 256 K → 1 M | Same as Qwen 3.6 |
| Tokenizer | Qwen 3.6 (vocab 248 320) | Unchanged |
| Vocabulary | tied embedding/LM-head | Saves ~0.7 GB |

Parameter count (active + total):

| Component | Quant | Per-layer | Total (24 layers) | Footprint |
|---|---|---|---|---|
| Routed experts (32 × 2 × 1792 × 768) | TQ1_0, 1.6875 bpw (group 256) | 88.0 M params | 2.11 B params | **0.45 GB** |
| Shared expert (2 × 1792 × 4096) | TQ1_0, 1.6875 bpw (group 256) | 14.7 M params | 0.35 B params | **0.07 GB** |
| Router (32 × 1792) | BF16 | 0.06 M | 1.4 M | 3 MB |
| Attention Q/K/V/O (H-blocks, 6 of 24) | BF16 | 13.5 M | 81 M | 0.16 GB |
| Mamba-3 projections (L-blocks, 18 of 24) | BF16 | 14.8 M | 266 M | 0.53 GB |
| Embedding + LM head (tied) | Q8_0 at export (BF16 in training) | — | 445 M | 0.47 GB |
| RMSNorm + biases | BF16 | — | ~5 M | 10 MB |
| **Total (storage, TQ1_0 build)** | | | **~3.26 B** | **~1.70 GB** |
| KV cache (TurboQuant K4/V2, 32 K ctx) | mixed | — | — | ~0.7 GB |
| Activations + workspace | BF16 | — | — | ~0.9 GB |
| **GPU resident at 32 K context** | | | | **~3.3 GB ✅** |

The 4 GB ceiling is met with ~0.7 GB headroom for OS + driver overhead. The TQ2_0 speed build adds ~0.10 GB on the expert tensors (still ≤ 3.4 GB resident). If PrismML's `BB1_0` lands upstream, the routed experts drop a further ~0.30 GB (stretch lane).

For comparison: AVA v2 at Q8_0 GGUF takes ~1.4 GB at 8 K context with no MoE and no Mamba state. v3 has **~13× more parameters per VRAM** (3.26 B vs 1.89 B Qwen 3.5 base × 0.5 quant savings) while staying inside the same 4 GB envelope.

---

## 3. Why this is "better than v2 in every regard"

The architecture below answers the v2 failure modes from [`V2_GAP_ANALYSIS.md`](V2_GAP_ANALYSIS.md):

| v2 limitation | v3 mechanism |
|---|---|
| Math/MATH-500 capped by single-pass dense forward | HRM-style L-module recurrence — multi-step reasoning before token emission |
| GSM8K capped by 384-token CoT cap | Native 32 K context + recurrence does CoT in latent space (no emitted tokens) |
| Tool routing 0.6 % call rate | MCP + constrained decoding + 300-example discrimination SFT (already in v3 P6) |
| Multilingual MGSM degradation | Larger active capacity (~1.4 B active vs 1.89 B all-active) + Qwen 3.6 tokenizer |
| HellaSwag 56.8 % below peers | Distilled from 35 B teacher; more capacity + balanced SFT mix |
| Long context untrained | Native 32 K + YaRN to 256 K; Mamba-3 subquadratic decode |
| Throughput ceiling | Mamba-3 linear decode + stock TQ2_0 2-bit-aligned kernels (PrismML demonstrated up to 8× FP16 at sub-2 bpw) |
| Memory ceiling | 1.6875 bpw packed experts + Q8_0 embeddings + Q4/Q2 KV cache |

---

## 4. Comparison to the original v3 design

The original v3 design in [`experiments/exp6_v3/DESIGN.md`](../../experiments/exp6_v3/DESIGN.md) had 28 layers, Gated DeltaNet 3:1 hybrid, MoTE pattern, and a teacher-distillation pipeline. This document **extends** that design — it does not replace it.

| Field | Original v3 | New v3 (this doc) | Reason |
|---|---|---|---|
| Layers | 28 | 24 | HRM recurrence trades depth for adaptive compute |
| Subquadratic layer | Gated DeltaNet | Mamba-3 (MIMO) | +1.8 pp at 1.5 B (ICLR 2026 result) |
| FFN inside L-block | Same MoE | Same MoE | Unchanged |
| Recurrence | None | H/L dual loop with halting | HRM-Text architecture |
| Routed expert packing | ternary BF16 weights | TQ1_0 1.6875 bpw, group 256 (BB1_0 1.125 bpw stretch) | Stock llama.cpp kernels, no fork dependency |
| KV cache | TurboQuant K4/V2 | TurboQuant K4/V2 | Unchanged |
| Routing | top-4/32 | top-4/32 | Unchanged |
| Shared expert | 1 BF16 | 1 ternary TQ1_0 (1.6875 bpw) | Saves ~5 GB on CPU resident set |
| Tokenizer | Qwen 3.6 | Qwen 3.6 | Unchanged |

All other choices (BitDistiller 3-stage, teacher = Qwen 3.6 35B-A3B, FLA kernel toolkit, MCP tool stack) are inherited unchanged.

---

## 5. Training implication summary

The architecture above maps to one extra training requirement on top of the v3 plan in [`experiments/exp6_v3/DESIGN.md`](../../experiments/exp6_v3/DESIGN.md):

- **Halting curriculum.** The L-module halting sigmoid is trained with ACT-style ponder loss (mean target = 2.5 L-steps/token, max 6). This is bolted onto Stage 2 of BitDistiller.
- **Mamba-3 stability tricks.** Complex-valued state at BF16 needs careful init; reuse the recipe in the Mamba-3 paper (5.1 of the ICLR 2026 PDF).
- **Ternary packer.** Stage 4 (export) packs experts to stock `TQ1_0`/`TQ2_0` (group 256 — the same grouping the QAT forward used), embeddings to Q8_0. No llama.cpp patch needed (June 2026 revision — see [PRISMML.md](PRISMML.md) §2).
- **Halting curriculum addition (June 2026).** Deep supervision on intermediate L-iterates (2/4/final, weights 0.2/0.3/0.5) + convergence-aware halting inputs — see [HRM_TEXT.md](HRM_TEXT.md) §1b.

Full pipeline in [RECIPE.md](RECIPE.md).
