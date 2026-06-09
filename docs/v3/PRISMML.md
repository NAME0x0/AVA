# PrismML Deployment

> **Source:** PrismML (Caltech spin-out). Bonsai 8 B (1-bit) launched 31 March 2026; Ternary Bonsai 1.7 B / 4 B / 8 B (1.58-bit) launched 16 April 2026.
> **Kernel status (verified 9 June 2026):** Bonsai kernels live in the **`PrismML-Eng/llama.cpp` fork (`prism` branch)** — upstream PRs are pending, *not merged*. Apple MLX serves Bonsai natively via its 2-bit path. Upstream `llama.cpp` has had its own ternary types `TQ1_0` / `TQ2_0` since 2024, but they use **group size 256** while Bonsai uses **group size 128**, so the formats are not interchangeable (see [llama.cpp discussion #22019](https://github.com/ggml-org/llama.cpp/discussions/22019)).

PrismML matters to AVA v3 because they proved the **deployment math**: an 8 B model in 1.15 GB at competitive quality. We do not train *with* PrismML and — after the June 2026 status check — we no longer depend on their fork either.

**The v3 pivot:** because v3 trains its ternary weights ourselves (BitDistiller QAT), we control the group size. We train with **group-256 scales from day one**, which makes our checkpoints export directly to **stock upstream `TQ1_0` / `TQ2_0`** — zero fork dependency, kernels that already ship in every llama.cpp / Ollama / LM Studio build. PrismML's 1.125 bpw `BB1_0` becomes a stretch goal we adopt only if their upstream PR lands.

---

## 1. What PrismML actually shipped

| Detail | Value |
|---|---|
| Base architecture | Qwen3-8B dense decoder (36 layers, GQA 32 q / 8 kv, SwiGLU, RoPE, 65 K ctx) |
| Bonsai 8 B weight format | each weight = ±1 (1 bit); 128 weights share one FP16 scale → **1.125 bpw** effective |
| Bonsai 8 B footprint | 1.15 GB (vs 16.38 GB FP16) — **14×** reduction |
| Bonsai 8 B speed | up to **8×** FP16 throughput on supported hardware |
| Ternary Bonsai (Apr 16) | 1.58-bit weights, sizes 1.7 B / 4 B / 8 B; packed 2-bit-aligned ("Q2_0"-style), group 128 |
| Backends (June 2026) | `PrismML-Eng/llama.cpp` fork (`prism` branch); Apple MLX 2-bit native. **Not upstream llama.cpp yet** |
| License | Open weights for the Ternary Bonsai family |

What we take from PrismML: **the existence proof and the quality data points**. What we deliberately do *not* take anymore: their group-128 storage layout, because it strands us on a fork.

---

## 2. The v3 storage decision: train group-256, export stock TQ

Upstream `llama.cpp` ternary types (merged 2024, originally for TriLM / BitNet b1.58):

| Type | Layout per 256-weight block | bpw | Kernel note |
|---|---|---|---|
| `TQ1_0` | trits packed base-3 (5 per byte) + 1 FP16 scale | **1.6875** | smallest; decode-bound friendly |
| `TQ2_0` | 2 bits per trit + 1 FP16 scale | **2.0625** | 2-bit aligned → fastest mat-vec on AVX-512 / CUDA |

v3 QAT (see [RECIPE.md](RECIPE.md) P5) computes its ternary fake-quant with **one scale per 256 consecutive weights**, matching this layout exactly. The exported GGUF is then bit-exact with what the QAT forward pass saw — no post-hoc re-grouping error, no custom kernel, no fork.

Per-tensor assignment for v3:

| Tensor class | Format | bpw | Why |
|---|---|---|---|
| Routed expert weights (32 × 2 × 1792 × 768 per layer) | **TQ1_0** (ship) / **TQ2_0** (speed build) | 1.6875 / 2.0625 | bulk of parameters; ternary QAT-trained |
| Shared expert weights (2 × 1792 × 4096 per layer) | **TQ1_0** | 1.6875 | ternary QAT-trained, group 256 |
| Router weights | BF16 | 16 | tiny, routing-critical |
| Attention Q/K/V/O | BF16 | 16 | outlier-sensitive; never sub-8-bit |
| Mamba-3 projections + SSM params | BF16 | 16 | complex-state stability |
| Embedding + LM head (tied) | **Q8_0 at export** (BF16 in training) | 8.5 | standard practice; ~0.42 GB saved, negligible quality cost |

Two GGUF builds ship at P9: `ava-v3-TQ1_0` (smallest) and `ava-v3-TQ2_0` (fastest). Same checkpoint, two packings.

### Where this leaves BB1_0 (1.125 bpw)

| Condition | Action |
|---|---|
| PrismML upstream PR lands in llama.cpp before P9 | Re-pack routed experts as BB1_0 → saves ~0.30 GB on disk; requires a 1-bit QAT branch at P5 (extra ablation) |
| PR not landed (expected) | Ship TQ1_0/TQ2_0 only. Revisit in v3.1 |

The 1-bit lane is now strictly opportunistic. The release gate ([PERF_TARGETS.md](PERF_TARGETS.md)) is computed against the TQ path.

---

## 3. GGUF export

The export script ([`experiments/exp6_v3/scripts/export_gguf.py`](../../experiments/exp6_v3/scripts/export_gguf.py)) becomes *simpler* than the original plan — it emits only formats stock `gguf-py` already knows:

```
1. Load BF16 student checkpoint (QAT master weights).
2. For each ternary tensor (routed + shared experts):
     groups  = reshape(weight, [-1, 256])
     scales  = group_scale(groups)               # same rule as QAT forward (mean-abs)
     trits   = round(groups / scales).clip(-1, 1)
     pack as TQ1_0 (5 trits/byte) or TQ2_0 (2 bits/trit)
3. Embedding/LM head → Q8_0. Router, attention, Mamba-3 → BF16/F16.
4. Write GGUF with stock tensor-type tags. Serve with unmodified llama.cpp.
```

Determinism requirement: step 2 must reuse the *identical* scale rule as the QAT forward pass, so packed inference reproduces training-time quantization exactly (round-trip test in §5).

---

## 4. What we do not adopt from PrismML

| PrismML feature | v3 take |
|---|---|
| Group-128 storage layout (`BB1_0` / Bonsai "Q2_0") | **No (changed June 2026).** Fork-only; group-size-incompatible with upstream TQ types. We train group-256 and export stock TQ1_0/TQ2_0 |
| Qwen3-8B dense base | We have Qwen 3.6 35B-A3B MoE teacher → 3.26 B student. Different problem, different base |
| 1-bit on attention Q/K/V/O | **No.** Attention is outlier-sensitive; 1-bit there causes catastrophic loss (BiLLM WikiText ppl 27+). v3 keeps attention BF16 |
| Direct post-training quantization | **No.** v3 uses BitDistiller 3-stage QAT. PTQ at ≤ 2 bpw on MoE has no published track record |
| MLX backend as primary | Secondary. v3's primary serving target is `llama.cpp` on x86/CUDA. MLX is a bonus for Mac users |

We are taking the *evidence* from PrismML (sub-2-bpw works at 8 B), not their format or training method.

---

## 5. Verification checklist

Before P10 release, all of the following must pass on the laptop:

- [ ] **Stock** `llama.cpp` (no fork, no patch) serves the v3 GGUF with no crashes at 32 K context.
- [ ] Decode throughput ≥ 30 tok/s on RTX A2000 at `reasoning_budget=auto` (TQ2_0 build).
- [ ] Resident GPU memory ≤ 3.6 GB at 32 K context.
- [ ] MMLU at TQ-packed weights within 1.5 pp of BF16 baseline.
- [ ] ARC-C at TQ-packed weights within 1.5 pp of BF16 baseline.
- [ ] TQ1_0 round-trip identical to QAT fake-quant outputs (deterministic packer test).

If MMLU regresses by > 1.5 pp at TQ1_0: ship TQ2_0 only (kernel rounding differences are smaller there) and file the delta in release notes.

---

## 6. References

- *PrismML Launches World's First 1-Bit AI Model*, PR Newswire, 31 March 2026.
- *PrismML Introduces Ternary Bonsai Model Family*, PR Newswire, 16 April 2026.
- `ggml-org/llama.cpp` discussion [#22019](https://github.com/ggml-org/llama.cpp/discussions/22019) — Bonsai group-128 vs upstream TQ group-256; fork status.
- `PrismML-Eng/llama.cpp` fork, `prism` branch — Bonsai Q2_0-style kernels (not upstream as of 9 June 2026).
- Compilade et al., upstream `TQ1_0` / `TQ2_0` quant types in `llama.cpp` (2024) — the formats v3 ships.
- Yan et al., *MoTE: Mixture-of-Ternary-Experts*, NeurIPS 2025 (ternary MoE pattern v3 inherits).

Full citation list: [REFERENCES.md](REFERENCES.md).
