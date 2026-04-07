# Exp5 Progress Log

## Goal

Build a practical Gemma 4 local-serving stack for a **4 GB VRAM / 32 GB RAM**
machine while preserving as much model quality as possible.

The work split into three tracks:

1. `26B feasible locally`
2. `dense smaller practical branches`
3. `fast local serving`

## Chronology

### 1. 26B feasibility branch

The first milestone was making `google/gemma-4-26B-A4B-it` fit and run at all.

Key changes:

- manual CPU/GPU layer placement in `engine/loader.py`
- streaming int4 loader and cached reload path in `engine/streaming.py`
- real TurboQuant bit-packing in `engine/turboquant.py`
- MoE caching / offload scheduling in `engine/moe_offload.py`

Representative outcomes:

- cached 26B reload: about `8.6s`
- first working 26B local answer: `The capital of France is **Paris**.`
- initial feasible decode speed: about `0.11 tok/s`

This branch proved that `26B feasible locally` is real, but not practical for
daily use.

### 2. Exact runtime tuning on 26B

Several exact optimizations improved the 26B path without changing model
behavior:

- packed and dequantized expert caches
- grouped MoE routing
- CPU dequantized linear cache
- GPU compartment / residency budgeting
- decode-oriented MoE admission and hot-set protection

Representative outcomes:

- 26B warm decode improved from roughly `0.11 tok/s` to about `0.45-0.50 tok/s`
- peak working RSS stayed within the local machine envelope

Takeaway:

- useful research result
- still too slow to be the default interactive path

### 3. Exact speculative decoding scaffold

An exact draft/verifier scaffold was added so the existing 26B runtime could be
tested with assisted decoding.

Result:

- correctness was fine
- `E4B` as a CPU-side draft was slower than plain decode on this machine

Takeaway:

- the scaffold is valuable
- this draft choice is not the practical speed path here

### 4. Practical E4B branch

The next pivot was to make a practical dense branch around `google/gemma-4-E4B-it`.

Configuration:

- bf16
- manual CPU/GPU split
- TurboQuant enabled
- YaRN extended to `512K`
- thinking disabled by default for latency

Representative result:

- decode speed: about `0.79-0.85 tok/s`
- total throughput: about `2.1-2.2 tok/s`
- peak VRAM: about `2.1 GB`

Takeaway:

- E4B is a sane deep local branch
- still not fast enough as the default responder

### 5. Fast E2B branch with Transformers

The fast-path pivot moved to `google/gemma-4-E2B-it`.

Representative best Transformers result:

- decode speed: about `2.13 tok/s`
- total throughput: about `5.92 tok/s`
- latency: about `4.22s`

Takeaway:

- this is the first branch that feels meaningfully faster
- still leaves a lot of speed on the table

### 6. Two-tier local runtime

A local router was added:

- `E2B` for the default fast path
- `E4B` for explicit deep/reasoning escalation

Routing controls:

- `quick:` / `fast:` force E2B
- `deep:` force E4B
- `reason:` / `think:` force E4B with thinking enabled

Takeaway:

- this architecture is the right product direction on this hardware
- model switch/load cost is the main remaining pain point

### 7. llama.cpp fast branch

The fastest local E2B branch now uses a source-built `llama.cpp` server plus
the local Gemma 4 E2B GGUF.

Important fixes:

- explicit `--reasoning off` on the fast branch
- safer response cleanup for Gemma 4 channel / thought markup
- persistent `llama.cpp` fast branch in the two-tier runtime

Best measured quick-path result:

- decode speed: about `6.39 tok/s`
- total throughput: about `17.74 tok/s`
- latency: about `1.41s`
- sanity output: `The capital of France is **Paris**.`

Takeaway:

- this is the current recommended fast path

## Current Recommendation

### Default practical stack

- fast path: `E2B` via `llama.cpp`
- deep path: `E4B` via Transformers
- routing: explicit keyword / heuristic two-tier local runtime

### Research stack

- 26B streamed int4 feasibility branch
- TurboQuant and YaRN research continues there

## Current Limits

1. `26B` remains a feasibility/research branch, not a fast branch.
2. `E4B` is useful as a deep branch, but branch switching still costs seconds.
3. `1M` context is still experimental on this machine; `512K` is the practical
   dense target today.
4. The remaining system bottleneck is no longer fast-path E2B inference; it is
   escalation cost into the deep branch.
