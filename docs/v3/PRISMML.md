# PrismML Deployment

> **Source:** PrismML (Caltech spin-out). Bonsai 8 B (1-bit) launched 31 March 2026; Ternary Bonsai 1.7 B / 4 B / 8 B (1.58-bit) launched 16 April 2026. Backends: `llama.cpp` head and Apple MLX. The PrismML kernels were upstreamed into both.

PrismML matters to AVA v3 because they did the **deployment** work that v3's training pipeline can drop into. We do not train *with* PrismML; we **export to** the PrismML packed format for inference.

This is the bridge between the BitDistiller-trained ternary student and the GGUF artifact a user downloads.

---

## 1. What PrismML actually shipped

| Detail | Value |
|---|---|
| Base architecture | Qwen3-8B dense decoder (36 layers, GQA 32 q / 8 kv, SwiGLU, RoPE, 65 K ctx) |
| Bonsai 8 B weight format | each weight = ±1 (1 bit); 128 weights share one FP16 scale → **1.125 bpw** effective |
| Bonsai 8 B footprint | 1.15 GB (vs 16.38 GB FP16) — **14×** reduction |
| Bonsai 8 B speed | up to **8×** FP16 throughput on supported hardware |
| Bonsai 8 B energy | 75–80 % less than FP16 |
| Ternary Bonsai (Apr 16) | 1.58-bit weights, sizes 1.7 B / 4 B / 8 B |
| Backends | `llama.cpp` head build, Apple `MLX` |
| License | Open weights for the Ternary Bonsai family |

What we are inheriting: **the storage layout, the group-scale convention, and the merged kernel code in upstream llama.cpp**. None of this requires a license payment.

---

## 2. Storage layout in detail

For each weight tensor `W ∈ R^{m × n}`:

1. Reshape to row-major blocks of 128 consecutive weights (the "group").
2. For each group:
   - Compute `s = scale` (one FP16 value = absmax / 1.0 for 1-bit, or scaled magnitude for ternary).
   - For 1-bit (Bonsai 8 B): each weight is stored as a single bit, `bit_i = 1` if `w_i > 0` else `0`. Dequant: `w_i ≈ (2 × bit_i - 1) × s`.
   - For ternary (1.58-bit, Ternary Bonsai): each weight is stored as a `trit` ∈ {−1, 0, +1}. Five trits packed in 8 bits (since 3^5 = 243 < 256). Dequant: `w_i ≈ trit_i × s`.
3. The packed binary is one contiguous stream: `[scales_FP16] [packed_bits_or_trits]`.

For AVA v3:

| Tensor class | PrismML pack | bpw |
|---|---|---|
| Routed expert weights (32 × 2 × 1792 × 768 per layer) | **1-bit** with group 128 | **1.125** |
| Shared expert weights (2 × 1792 × 4096 per layer) | **Ternary (1.58-bit)** with group 128 | **1.58** |
| Router weights | BF16, unpacked | 16 |
| Attention Q/K/V/O | BF16, unpacked | 16 |
| Mamba-3 projections | BF16, unpacked | 16 |
| Embedding + LM head (tied) | BF16, unpacked | 16 |

The 1-bit choice for routed experts is the aggressive lane. We are bet-hedging on the shared expert with 1.58-bit because the shared expert carries cross-task knowledge and its quality is more sensitive to quantization (validated by MoTE ablations).

If the 1-bit routed lane fails Stage 2 distillation (i.e. > 3 pp MMLU regression vs the 1.58-bit lane), we downgrade all routed weights to 1.58-bit. See [RISKS.md](RISKS.md) §3.

---

## 3. GGUF integration

PrismML's contribution to upstream `llama.cpp` (merged March–April 2026) added two new quant types:

- `TQ1_0` — ternary 1.58-bit with FP16 group scales of 128 (already present pre-PrismML for BitNet; PrismML extended it with optimized kernels).
- `BB1_0` — 1-bit binary with FP16 group scales of 128 (introduced by PrismML; the "BB" prefix is the upstream name).

For AVA v3, the GGUF metadata distinguishes routed (BB1_0) from shared (TQ1_0) tensors at the per-tensor level. The same llama-server build can serve both.

A patch list lives at [`experiments/exp6_v3/scripts/export_gguf.py`](../../experiments/exp6_v3/scripts/export_gguf.py) (currently a stub). The export script will:

```
1. Load BF16 student checkpoint.
2. For each routed expert weight:
     groups = reshape(weight, [-1, 128])
     scales = max(abs(groups), axis=-1)        # FP16
     bits   = (groups > 0).to(uint8)            # bit-pack 8 weights per byte
     append to TENSOR_DATA[BB1_0]
3. For each shared expert weight:
     groups = reshape(weight, [-1, 128])
     scales = scale_optimal(groups)             # ternary RTN with PrismML scale rule
     trits  = round(groups / scales).clip(-1, 1)
     pack 5 trits per byte
     append to TENSOR_DATA[TQ1_0]
4. Write GGUF header + tensors.
```

The packer is ~200 LOC. The kernels do the heavy lifting; we only need to produce correctly laid-out bytes.

---

## 4. Why this is "free" compared to inventing our own format

Three reasons.

1. **Kernels are upstream and tuned.** The `llama.cpp` matmul kernels for `BB1_0` and `TQ1_0` were written by PrismML engineers with full per-arch tuning (Apple Silicon, x86 AVX-512, CUDA). Reimplementing would burn weeks for worse throughput.
2. **GGUF is the v2 distribution channel.** The AVA v2 GGUF release (`NAME0x0/AVA-v2-GGUF`) was the lever that got Ollama / LM Studio adoption. v3 inherits that channel by reusing the same format.
3. **Hugging Face's `huggingface.js` and `llama-cpp-python` recognize the quant tags.** No bespoke loader needed in the ecosystem.

---

## 5. What we do not adopt from PrismML

| PrismML feature | v3 take |
|---|---|
| Qwen3-8B dense base | We have Qwen 3.6 35B-A3B MoE teacher → 3.26 B student. Different problem, different base. |
| 1-bit on attention Q/K/V/O | **No.** Attention is outlier-sensitive; 1-bit there causes catastrophic loss (BiLLM WikiText ppl 27+). v3 keeps attention BF16. |
| Direct post-training quantization | **No.** v3 uses BitDistiller 3-stage QAT. PTQ at 1.125 bpw on MoE has no published track record. |
| MLX backend as primary | Secondary. v3's primary serving target is `llama.cpp` on x86/CUDA. MLX is a bonus for Mac users. |

We are taking the *output format* from PrismML, not the *training method*. The training method stays BitDistiller + HRM halting curriculum + Mamba-3 stability (see [RECIPE.md](RECIPE.md)).

---

## 6. Verification checklist

Before P10 release, all of the following must pass on the laptop:

- [ ] `llama.cpp` head build serves the v3 GGUF with no crashes at 32 K context.
- [ ] Decode throughput ≥ 30 tok/s on RTX A2000 at `reasoning_budget=auto`.
- [ ] Resident GPU memory ≤ 3.6 GB at 32 K context.
- [ ] MMLU at Q-packed weights within 1.5 pp of BF16 baseline.
- [ ] ARC-C at Q-packed weights within 1.5 pp of BF16 baseline.
- [ ] BB1_0 + TQ1_0 round-trip identical to the unpacked checkpoint (deterministic packer test).

If MMLU regresses by > 1.5 pp, escalate to: (a) downgrade routed to TQ1_0 (1.58-bit), or (b) widen group size to 256 to give more headroom per scale.

---

## 7. References

- *PrismML Launches World's First 1-Bit AI Model*, PR Newswire, 31 March 2026.
- *PrismML Introduces Ternary Bonsai Model Family*, PR Newswire, 16 April 2026.
- *PrismML Exits Stealth With First Commercially Viable 1-Bit LLMs*, Machine Herald, 3 April 2026.
- `ggerganov/llama.cpp` PRs introducing `BB1_0` (search "PrismML" in PR history).
- Yan et al., *MoTE: Mixture-of-Ternary-Experts*, NeurIPS 2025 (ternary MoE pattern v3 inherits).

Full citation list: [REFERENCES.md](REFERENCES.md).
