# Subquadratic Strategy

> **Source papers:**
> - Yang et al., *Gated Delta Networks: Improving Mamba2 with Delta Rule*, OpenReview r8H7xhYPwz (Gated DeltaNet)
> - Mamba-3 authors, *Mamba-3: Improved Sequence Modeling using State Space Principles*, arXiv:2603.15569 (ICLR 2026)
> - `fla-org/flash-linear-attention` — Triton kernel library, RWKV7 added Jan 2025, Mamba-3 PR open as of May 2026

AVA v3 must deliver native 32 K context (256 K via YaRN, 1 M target) on a 4 GB laptop. Softmax attention is impossible at those lengths under our memory budget — the KV cache alone for 256 K context with 14 heads × 128 dim × BF16 would be ~14 GB. We therefore need **subquadratic** sequence modeling on most layers, with a small fraction of softmax-attention layers preserved for tasks that require exact recall.

This document explains why we choose **Mamba-3** over Gated DeltaNet, in what **3 : 1 hybrid ratio**, and how the kernels integrate with the FLA toolkit.

---

## 1. The 2026 subquadratic landscape

| Generation | Representative model | State update | Best public number (1.5 B) |
|---|---|---|---|
| Gen-1 | RetNet, GLA | Linear, gated read-out | Lags RWKV-6 on LongBench |
| Gen-2 | Mamba-2, RWKV-6 | Data-dependent scalar gate; selective SSM | Baseline |
| Gen-3 | DeltaNet, **Gated DeltaNet** | Delta rule + gating | +X over Mamba-2 |
| Gen-3+ | **Mamba-3**, MIMO variant | SSM-discretized complex update, MIMO read-out | **+1.8 pp** over Gated DeltaNet, **half the state size** of Mamba-2 |

Mamba-3 is the inference-first revision of the SSM family: it preserves the linear-decode property, adds a more expressive recurrence via SSM discretization, and reduces hidden state size by half at equal perplexity. The MIMO formulation extracts more capacity per state without adding decode latency.

Source: arXiv:2603.15569 abstract and benchmark table at 1.5 B scale.

---

## 2. Why Mamba-3 specifically (and not just Gated DeltaNet)

Two reasons, in priority order.

### 2a. Half the state pairs with HRM's L-loop

The HRM L-loop (see [HRM_TEXT.md](HRM_TEXT.md)) iterates 2–6 times per token. Each iteration writes through the subquadratic state. With Mamba-2-sized state, the loop's effective working set is ~14 heads × 128 dim × 6 iterations = ~10 KB per token per layer of "scratchpad". With Mamba-3's half-state, it's ~5 KB — exactly enough to keep the L-loop fitting in L2 cache on consumer GPUs.

This is not just a memory win, it's a wall-clock win. The L-loop is the per-token critical path; halving its state halves the bandwidth pressure.

### 2b. Complex-valued state for partial-reasoning accumulation

Mamba-3's complex-valued update rule is `s' = α · s + β · u`, where `α, β ∈ ℂ`. The complex rotation lets the state carry **phase information** across iterations, which empirically beats real-valued accumulation for state-tracking and retrieval tasks (Mamba-3 §3.2).

For HRM-style reasoning, where each L-step refines the same problem, this is exactly the right inductive bias. Real-valued state under-attributes to "I've seen this evidence already"; complex phase tracks it.

We are not arguing either of these effects is decisive on its own. We are arguing that Mamba-3 dominates Gated DeltaNet on the published benchmark, costs us nothing extra (same FLA toolkit, same training pipeline), and aligns mechanically with the L-loop. It is a free upgrade.

---

## 3. Layer ratio: 3 L-blocks : 1 H-block

```
Block 1   L  Mamba-3 + MoE                ← subquadratic, no RoPE
Block 2   L  Mamba-3 + MoE
Block 3   L  Mamba-3 + MoE
Block 4   H  Gated Attention + RoPE + MoE ← softmax, RoPE
Block 5   L  Mamba-3 + MoE
Block 6   L  Mamba-3 + MoE
Block 7   L  Mamba-3 + MoE
Block 8   H  Gated Attention + RoPE + MoE
... repeat × 3 → 24 blocks total (18 L, 6 H)
```

The 3 : 1 ratio is inherited from Qwen 3.6, where it was selected as the empirical sweet-spot between pure-subquadratic (RULER-recall collapse beyond 128 K) and pure-attention (KV cache blows the budget).

Compute split per forward pass at 32 K context:

| Layer type | Count | FLOPs per token | Mem traffic |
|---|---|---|---|
| L (Mamba-3) | 18 | O(state_dim²) | constant per token |
| H (Gated Attention) | 6 | O(seq_len × head_dim) | grows with context |
| Total at 32 K | 24 | ~70 % L, ~30 % H | ~50/50 |

At 256 K context, H-block time dominates because softmax attention is quadratic. Mitigations: flash-attention 3 kernels (already in v3 inference plan), and the KV cache uses TurboQuant K4/V2 from Exp 5.

---

## 4. RoPE placement

RoPE applies only to H-blocks (softmax attention). L-blocks use Mamba-3's data-dependent gating instead — there is no position to rotate.

This is the same convention Qwen 3.6 uses, and was validated on RULER (Long-context recall ≥ 90 % at 128 K with the 3:1 hybrid).

---

## 5. FLA kernel plan

We rely on the `fla-org/flash-linear-attention` Triton kernel library.

| Kernel | FLA status | v3 dependency |
|---|---|---|
| Gated DeltaNet (`fused_recurrent`, `chunkwise`) | Released | Fallback if Mamba-3 kernel not ready |
| Mamba-3 (MIMO) | PR open (May 2026) | **Primary kernel; vendor or contribute upstream** |
| Flash Attention 3 (for H-blocks) | Stable | Used as-is |

Risk: if the Mamba-3 PR doesn't land in FLA by P3 (warmup), we fall back to Gated DeltaNet. The architecture change is one-line — the L-block sublayer type is configurable.

---

## 6. Native context length training

Stage 1 warmup trains at **32 K context** with packed sequences. Stage 2 and 3 hold at 32 K (cheaper for distillation throughput). YaRN extension is applied as a post-process to push the effective context to 256 K, with the standard `mscale` adjustment for RoPE base 1 000 000.

For 1 M context, we do a 0.5 B-token additional fine-tune on ProLong + LongAlpaca-12k at 64 K → 128 K → 256 K curriculum.

---

## 7. Long-context eval

| Bench | Target |
|---|---|
| RULER 8 K | recall ≥ 95 % |
| RULER 32 K | recall ≥ 92 % |
| RULER 128 K | recall ≥ 90 % |
| RULER 256 K | recall ≥ 80 % |
| LongBench v2 | within 3 pp of Qwen 3.6 27B |
| Passkey 1 M | retrieve passkey ≥ 80 % |

If any 32 K / 128 K target misses by > 5 pp, we drop YaRN to 128 K-only and accept the smaller advertised context.

---

## 8. References

- Yang et al., *Gated DeltaNet* (OpenReview r8H7xhYPwz).
- arXiv:2603.15569, *Mamba-3*.
- arXiv:2402.04691, *YaRN: Efficient Context Window Extension*.
- `fla-org/flash-linear-attention` GitHub.

Full citation list: [REFERENCES.md](REFERENCES.md).
