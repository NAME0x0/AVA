# Exp5: Gemma 4 Inference Optimization Engine — Results

## Overview

Exp5 builds inference optimizations for running Gemma 4 models on constrained
hardware: **4 GB VRAM (RTX A2000 Laptop GPU) + 32 GB RAM**. The target models
are Gemma 4 E4B (8B dense) and Gemma 4 26B-A4B (MoE with 128 experts).

Four optimizations were implemented:

| # | Optimization | Status | Key Result |
|---|-------------|--------|------------|
| 1 | GPU Placement | Done | 2.0 tok/s (43% faster than CPU-only) |
| 2 | TurboQuant Bit-Packing | Done | 4.8x KV cache compression (was 1.6x) |
| 3 | Streaming Quantization | Done | 50 GB model loads in 8.9 GB peak RAM, correct inference |
| 4 | Quantized Weight Cache | Done | 11.4x faster reload (23.8s vs 271s) |

---

## 1. GPU Placement (Manual Layer Movement)

**Problem:** `accelerate`'s `infer_auto_device_map` treats Gemma4ForConditionalGeneration
as one unsplittable block, placing everything on CPU. Even with a manual device map,
`dispatch_model` uses meta tensors for CPU-offloaded parameters. Gemma 4's multimodal
forward path directly accesses `embed_tokens.weight` (bypassing accelerate hooks),
causing `RuntimeError: Tensor on device meta`.

**Solution:** Skip `dispatch_model` entirely. Load all weights as real CPU tensors,
then manually `.to('cuda:0')` specific decoder layers. Install pre/post forward
hooks at the GPU/CPU boundary for automatic device transfers.

**Implementation:** `engine/loader.py` — `_move_layers_to_gpu()` and `_install_device_boundary_hooks()`

**Key design decisions:**
- Pre-hooks on EVERY GPU layer (not just the first) because the text model's
  forward loop passes fresh CPU tensors (position_embeddings, attention_mask)
  to each layer call from its own local scope
- Post-hook on the last GPU layer moves hidden_states back to CPU for remaining layers
- `non_blocking=True` on device transfers for better overlap
- 1.5 GB headroom reserved for KV cache + activations + intermediate tensors

**Results on Gemma 4 E4B-it (bf16):**

| Metric | CPU-only | 11 layers GPU |
|--------|----------|---------------|
| Speed | 1.4 tok/s | 2.0 tok/s |
| Peak VRAM | 0 MB | 2092 MB |
| Layers on GPU | 0/42 | 11/42 |
| GPU weight | 0 MB | 2074 MB |
| Output quality | Correct | Correct (identical) |

---

## 2. TurboQuant Bit-Packing

**Problem:** TurboQuant V3 stores quantized KV cache indices as `uint8` (1 byte per
index), even though 4-bit indices need only 4 bits and 2-bit indices need 2 bits.
Actual compression was 1.6x vs theoretical 5.3x.

**Solution:** Pack multiple indices per byte:
- 4-bit: 2 indices per byte (high nibble + low nibble)
- 2-bit: 4 indices per byte (6-4-2-0 bit shifts)

**Implementation:** `engine/turboquant.py` — `pack_indices()` and `unpack_indices()`

**Key design decisions:**
- Vectorized pack/unpack using torch bitwise ops (no Python loops)
- Auto-detection of packed format in `decompress()` from last dimension size
- Always returns 4-tuple (indices, norms, mins, scales) — packed format is transparent
- `pack=True` default on `MSECompressor` and `TurboQuantV3`

**Results (K4/V2 asymmetric on bf16 KV cache):**

| Metric | Before (uint8) | After (bit-packed) |
|--------|---------------|-------------------|
| Key compression | ~2x | 3.7x |
| Value compression | ~2x | 6.9x |
| Combined K+V | 1.6x | 4.8x |
| Reconstruction error | Same | Same (packing is lossless) |

Validated on real Gemma 4 E4B inference: 4 global attention layers compressed
with correct answers (Tokyo, Jupiter, quantum computing).

---

## 3. Streaming Quantization (Any-Size Model Loader)

**Problem:** The Gemma 4 26B-A4B model is 50 GB in bf16, which doesn't fit in 32 GB
RAM. Standard loading (`from_pretrained`) tries to materialize the full model.

**Solution:** Stream safetensors weight-by-weight, quantize each to int4 on-the-fly,
never hold more than 1 layer of bf16 at a time.

**Implementation:** `engine/streaming.py`

### Architecture

```
streaming_load()
  |
  +-- Step 1: Create model shell on meta device (0 RAM)
  |
  +-- Step 2: Parse safetensors index, resolve shard paths
  |
  +-- Step 3: For each layer group:
  |     |
  |     +-- Load tensor from safetensors (direct file read, no mmap)
  |     +-- If 2D linear weight -> StreamingInt4Quantizer -> QuantizedLinear
  |     +-- If 3D expert weight -> collect, quantize per-expert -> QuantizedMoEExperts
  |     +-- If non-quantizable (norms, embeds) -> assign as-is in bf16
  |     +-- gc.collect() after each layer
  |     
  +-- Step 4: Post-processing
        +-- Materialize non-persistent meta buffers (RoPE, embed_scale)
        +-- Re-tie weights (lm_head = embed_tokens)
```

### Key Components

**`StreamingInt4Quantizer`** — Per-group asymmetric int4 quantization:
- Group size 128, per-group min/max -> zero point + scale
- Pack 2 values per byte (high nibble, low nibble)
- No calibration data needed (round-to-nearest)

**`QuantizedLinear`** — Drop-in nn.Linear replacement:
- Stores packed int4 weights + fp16 scales/zeros
- Dequantizes to input dtype on every forward call
- ~3.8x memory reduction per linear layer

**`QuantizedMoEExperts`** — Drop-in Gemma4TextExperts replacement:
- Stores 128 experts as individual packed int4 buffers
- During forward, dequantizes ONLY the selected experts (typically 2 per token)
- Avoids `grouped_mm` which requires float tensors
- Uses eager expert-by-expert dispatch (matching the base implementation)
- ~3.7x memory reduction for expert weights

**Direct Safetensors Reader** — Bypasses `safe_open` mmap:
- `safe_open` mmaps the entire file; on 32 GB RAM with a 50 GB file, reading
  any tensor causes a segfault (virtual address space exhaustion on Windows)
- `_read_tensor_from_safetensors()` reads the JSON header, seeks to the
  exact byte offset, reads only the bytes needed, creates a tensor
- Peak RAM = size of ONE tensor being read, not the entire shard

### Key Challenges Solved

1. **Meta device buffers:** Non-persistent buffers (RoPE `inv_freq`, `embed_scale`)
   created during `__init__` remain on meta after weight loading. Fixed by
   re-instantiating RotaryEmbedding modules with correct sub-configs
   (VisionConfig vs TextConfig) and computing embed_scale from config.

2. **Weight tying:** Gemma 4 ties `lm_head.weight = embed_tokens.weight`.
   After streaming weights, the tie is broken (lm_head still points to
   meta parameter). Fixed by calling `model.tie_weights()` post-load.

3. **50 GB mmap crash:** `safetensors.safe_open` memory-maps the entire file.
   On Windows with 32 GB RAM, accessing any page in the 50 GB mapping crashes.
   Fixed with direct file I/O using byte offsets from the safetensors header.

4. **MoE expert compatibility:** `grouped_mm` requires float tensors; packed
   uint8 causes `RuntimeError: Expected Float32/BFloat16/Float16, got Byte`.
   Fixed by creating `QuantizedMoEExperts` with eager per-expert dispatch
   and on-the-fly dequantization.

### Results

**Gemma 4 E4B-it (8B dense) — Streaming Load:**

| Metric | Value |
|--------|-------|
| Load time | 42s |
| bf16 size | 16.0 GB |
| Quantized size | 9.4 GB |
| Compression | 1.7x overall (3.8x per linear) |
| Peak RAM | 14.0 GB |
| Linears quantized | 625 |
| Inference | Correct ("Paris" for capital of France) |

**Gemma 4 26B-A4B-it (MoE 128 experts) — Streaming Load:**

| Metric | Value |
|--------|-------|
| Load time | 244s (~4 min) |
| bf16 size | 51.6 GB |
| Quantized size | 15.2 GB (with expert quantization) |
| Compression | 3.4x |
| Peak RAM | 17.7 GB |
| Linears quantized | 8,104 |
| Inference | Correct ("Paris" for capital of France) |
| Inference speed | 0.09 tok/s (CPU dequant overhead — loading mechanism, not runtime) |
| Inference peak RAM | 12.6 GB |
| QuantizedLinear modules | 424 |
| QuantizedMoEExperts modules | 30 |

**QuantizedMoEExperts synthetic validation (128 experts, 26B dims):**

| Metric | Value |
|--------|-------|
| Expert memory (all layers) | 5,776 MB int4 packed |
| Per-expert compression | 3.7x (bf16 -> int4) |
| Forward output | Correct shape (4, 2304), no NaN/Inf |
| Output norm | 107.1 (healthy, non-degenerate) |
| Routing | top-2 experts per token (matches 26B config) |

---

## 4. Quantized Weight Cache (Fast Reload)

**Problem:** Streaming int4 quantization takes 271s for 26B — every experiment
costs 4.5 minutes just to load. Need fast iteration.

**Solution:** Save quantized model (packed int4 buffers + bf16 embeddings/norms)
to sharded safetensors on disk. Reload reads pre-quantized weights directly.

**Implementation:** `engine/streaming.py` — `save_quantized()` and `load_quantized()`

**Key design decisions:**
- Sharded safetensors (4 GB per shard) for safe mmap on 32 GB systems
- Module map JSON records which modules are QuantizedLinear/QuantizedMoEExperts
  with their config (original_in_features, group_size, num_experts, etc.)
- Reload creates model on meta device, replaces modules with quantized placeholders,
  then streams tensors one-by-one from safetensors (no double-memory)
- Auto-cached via `load_model()` in loader.py: first call is slow, subsequent are fast

**Results on Gemma 4 26B-A4B-it:**

| Metric | Streaming Load | Cached Reload |
|--------|---------------|---------------|
| Load time | 271s | 23.8s |
| Speedup | 1x | 11.4x |
| Peak RAM (inference) | 12.6 GB | 8.9 GB |
| Inference | Correct ("Paris") | Correct ("Paris") |
| Disk usage | 0 (reads from HF cache) | 16.7 GB (4 shards) |

---

## File Structure

```
experiments/exp5_gemma4/
  engine/
    loader.py        — Model loading, GPU placement, device boundary hooks
    turboquant.py    — TurboQuant V3 with bit-packed indices
    tq_cache.py      — KV cache compression validation
    yarn.py          — YaRN RoPE context extension (256K -> 1M)
    moe_offload.py   — Expert LRU cache for on-demand expert loading
    patches.py       — Model patching (YaRN + TurboQuant attachment)
    benchmark.py     — Speed/quality/memory benchmarking
    streaming.py     — Streaming quantization loader + save/load cache
  scripts/
    run_baseline.py  — Baseline benchmark runner
    run_optimized.py — Full optimization pipeline benchmark
    test_engine.py   — 19 unit tests for all engine modules
    test_gpu_placement.py — GPU placement integration test
    test_streaming.py     — Streaming quantizer + MoE experts tests
    test_26b_inference.py — End-to-end 26B inference (RAM-guarded)
    test_save_reload_26b.py — Save/reload cycle + inference validation
  results/
    optimized_gemma-4-e4b-it_*.json — Benchmark results
```

---

## Hardware

- **GPU:** NVIDIA RTX A2000 Laptop GPU (4.3 GB VRAM)
- **RAM:** 32 GB
- **OS:** Windows 11 Pro
- **Python:** 3.13.5
- **PyTorch:** 2.x with CUDA
- **Transformers:** Latest (with Gemma 4 support)

---

## Known Limitations

1. **Streaming int4 quality:** Round-to-nearest int4 without calibration data
   has lower quality than GPTQ/AWQ. Suitable for initial loading; consider
   saving quantized weights and using GPTQ for production.

2. **Inference speed:** `QuantizedLinear` dequantizes on every forward call
   on CPU, adding overhead (~13s/tok on E4B). This is a loading mechanism,
   not a runtime optimization. For fast inference, use GPU placement + bf16.

3. **QuantizedMoEExperts uses eager dispatch:** The base `grouped_mm` path
   is faster but requires float tensors. The eager path processes experts
   one-by-one, which is slower but compatible with packed int4.

4. **26B inference speed:** With QuantizedMoEExperts on CPU, inference is
   ~0.10 tok/s due to per-expert dequantization overhead. Combine with GPU
   placement for speed improvement.

5. **Quantized cache disk usage:** The 26B cached weights use 16.7 GB on disk
   (4 shards). This is in addition to the HuggingFace bf16 cache.

---

## 5. Practical Dense Branches (E4B and E2B)

After the 26B feasibility branch was proven, the work pivoted to dense Gemma 4
models for practical local serving.

### E4B Practical Branch

Goal:

- keep a stronger dense model available locally
- stay within the `4 GB VRAM / 32 GB RAM` machine budget
- preserve TurboQuant and YaRN in the stack

Configuration:

- model: `google/gemma-4-E4B-it`
- dtype: bf16
- manual CPU/GPU split
- TurboQuant enabled
- YaRN target: `512K`
- thinking disabled by default for latency

Representative result:

| Metric | Value |
|--------|-------|
| Load time | ~5.0s |
| Layers on GPU | 11/42 |
| Decode speed | ~0.79–0.85 tok/s |
| Total throughput | ~2.1–2.2 tok/s |
| Peak VRAM | ~2.1 GB |
| Output quality | Correct on sanity checks |

Takeaway:

- E4B is the best deep local dense branch tested so far.
- It is substantially more practical than 26B, but still not fast enough to
  be the default interactive branch.

### E2B Fast Branch (Transformers)

Goal:

- build a genuinely fast local branch using the same Gemma 4 family

Configuration:

- model: `google/gemma-4-E2B-it`
- dtype: bf16
- manual CPU/GPU split
- static KV cache
- thinking disabled by default

Best measured Transformers result:

| Metric | Value |
|--------|-------|
| Layers on GPU | 22/35 |
| Decode speed | ~2.13 tok/s |
| Total throughput | ~5.92 tok/s |
| Latency | ~4.22s |
| Peak VRAM | ~2.05 GB |
| Output quality | Correct ("Paris") |

Takeaway:

- E2B is the first Gemma 4 branch that feels meaningfully fast on this box.
- It became the natural fast-path candidate for a tiered local runtime.

---

## 6. Two-Tier Local Runtime

With E2B and E4B both working locally, a two-tier local runtime was built:

- **fast branch:** `E2B`
- **deep branch:** `E4B`

Routing controls:

- `quick:` / `fast:` force E2B
- `deep:` force E4B
- `reason:` / `think:` force E4B with thinking enabled

This keeps the fast path snappy while preserving a higher-quality local branch
for harder prompts.

### Observed behavior

The architecture works, but model switch/load overhead dominates mixed-path
requests. A representative mixed E2B→E4B run showed:

| Metric | Value |
|--------|-------|
| Combined decode speed | ~1.79 tok/s |
| Average request latency | ~23.4s |
| Switch/load overhead | ~14.9s |
| Peak VRAM | ~2.09 GB |

Takeaway:

- the tiered architecture is the correct product direction
- the main remaining cost is branch switching, not fast-path E2B generation

---

## 7. llama.cpp Fast E2B Backend

To push the fast branch further, an alternate E2B backend was implemented using
`llama.cpp` and the Gemma 4 E2B GGUF.

### Key fixes

- source-built `llama-server` used instead of older incompatible Windows bundle
- explicit `--reasoning off` on the fast path
- response cleanup for Gemma 4 channel / thought markup

### Best measured quick-path result

| Metric | Value |
|--------|-------|
| Backend | `llama.cpp` |
| Model | Gemma 4 E2B Q8_0 GGUF |
| Decode speed | ~6.39 tok/s |
| Total throughput | ~17.74 tok/s |
| Latency | ~1.41s |
| Output quality | Correct ("Paris") |

Takeaway:

- this is the current best fast local serving path in the repo
- it should be treated as the default fast branch while E4B remains the deep
  escalation branch

---

## Current Recommendation

For practical local use on this exact machine:

1. **Fast path:** `E2B` via `llama.cpp`
2. **Deep path:** `E4B` via Transformers at `512K`
3. **Research path:** keep `26B` as the streamed feasibility branch

This preserves the original research work while acknowledging the real hardware
limits of a `4 GB VRAM / 32 GB RAM` laptop.
