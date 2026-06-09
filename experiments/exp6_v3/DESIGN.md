# AVA v3 — Design Document

**Status:** P2 implementation, June 2026 (engine modules + smoke tests landed; June revision: refinement-first HRM per ARC Prize / arXiv:2601.10679, group-256 ternary → stock TQ1_0/TQ2_0 export).
**Goal:** Build a SOTA-class assistant on a 4 GB VRAM laptop by combining
(1) a Qwen 3.6 35B-A3B MoE teacher → ternary student distillation,
(2) **HRM-Text-style dual-recurrent reasoning** (H-module + L-module + halting head, latent multi-step inference, no emitted CoT tokens),
(3) **Mamba-3 (MIMO, complex SSM) in a 3 : 1 hybrid** with gated softmax attention for subquadratic decode + long context,
(4) **MoTE-pattern ternary FFN experts** with BF16 router and BF16 → ternary shared expert,
(5) **sub-2-bpw packed storage** — group-256 ternary QAT exported to stock upstream llama.cpp `TQ1_0`/`TQ2_0` (PrismML Bonsai is the existence proof, not a dependency),
(6) MCP-based tool use via constrained decoding + targeted "when NOT to call" SFT,
(7) frontier-distilled CoT corpus.

**Hardware budget:** RTX A2000 4 GB VRAM, 32 GB RAM, single-laptop training for every phase except P5 (rented A100s for the distillation token budget).
**Targets (strict dominance gate):** every AVA v2 benchmark non-regressed plus MMLU ≥ 70 %, GSM8K greedy ≥ 70 %, MATH-500 ≥ 50 %, MMLU-Pro ≥ 50 %, HumanEval+ ≥ 50 %, IFEval ≥ 65 %, BFCL v3 ≥ 70 %, RULER 128 K ≥ 90 %, ≥ 30 tok/s decode at `reasoning_budget=auto`. Full row-by-row matrix in [`../../docs/v3/PERF_TARGETS.md`](../../docs/v3/PERF_TARGETS.md).

**Companion documentation:** the v3 doc set in [`../../docs/v3/`](../../docs/v3/) is now the canonical home for the architecture, gap analysis, training recipe, and risk register. This file is the design-doc summary that sits next to the engine stubs.

This document supersedes the AVA v3 plan in `plans/linked-growing-mccarthy.md` (Gemma 4 E4B / E2B). That direction is paused. Exp5 files retained.

---

## 1. Executive summary

| Pillar | Choice | Why |
|---|---|---|
| Base / teacher | `Qwen/Qwen3.6-35B-A3B` (256 experts, 8+1 active, native Gated DeltaNet, 256 K ctx, Apache 2.0) | Already MoE + sub-quadratic; matches every requirement; latest Qwen team release |
| Student size | ~3.26 B params total (~1.4 B active routed + 0.4 B shared), ternary MoE | Tightest size with strict dominance over v2 at 4 GB; see `docs/v3/ARCHITECTURE_V3.md` memory math |
| **Reasoning** | **HRM-Text dual recurrence** — H-module (slow planner, softmax + MoE) + L-module (fast computer, Mamba-3 + MoE), per-token halting (max 6 L-steps, mean ~2.5) | Latent multi-step reasoning in one forward pass; closes v2 MATH/GSM8K gap. Source: Sapient HRM-Text, arXiv:2506.21734, April 2026 numbers MMLU 60.7 / MATH 56.2 |
| **Subquadratic attention** | **Mamba-3 (MIMO, complex SSM)** in **3 : 1 hybrid** with Gated Attention; YaRN to 256 K → 1 M | +1.8 pp over Gated DeltaNet at 1.5 B; half the state size of Mamba-2 (ICLR 2026, arXiv:2603.15569). Gated DeltaNet kept as fallback if Mamba-3 kernel ships late |
| Quantization | MoTE pattern: ternary routed FFN experts + BF16 router + ternary (1.58-bit) shared expert | Best published 1-bit + MoE combination; saves ~5 GB on the CPU-resident set vs BF16 shared |
| **Storage / packing** | **Stock upstream `TQ1_0` (1.6875 bpw) / `TQ2_0` (2.0625 bpw)**, FP16 group scales of 256 baked into QAT; Q8_0 embeddings | Zero fork dependency (June 2026: PrismML BB1_0 confirmed fork-only, group-128 incompatible — see `docs/v3/PRISMML.md`). PrismML demonstrated up-to-8× FP16 decode at sub-2 bpw |
| Training method | BitDistiller 3-stage QAT + distillation from Qwen 3.6 35B-A3B teacher + HRM halting curriculum (ACT ponder loss) | PTQ to 1-bit fails catastrophically; QAT-from-scratch needs trillions of tokens; HRM curriculum is added in Stage 2a |
| Corpus | SYNTHETIC-1 + Bespoke-Stratos-17k + Tulu-3 + Opus-4.7 distill + OpenMathInstruct-2 + Hermes-FC + Skywork-Reward | ~755 K SFT + 75 K DPO, all Apache-2.0 / MIT |
| Tool use | TOOLS.md + FastMCP 3.0 server + XGrammar constrained decoding + small discrimination SFT (300 examples) | BFCL evidence: prompting + constraints beats heavy FT for small models |
| Inference | **Stock** llama.cpp (TQ1_0/TQ2_0 kernels upstream since 2024; MCP client merged Mar 2026) + llama-server | Reuses Exp4-eval infrastructure; native MCP loop; no fork pin |
| Evaluation | BFCL v3 Prompt-mode, MCP-Bench, Tau-Bench, full Exp4-v2 17-bench suite, RULER long-context | Regression-gated CI; strict dominance over v2 |

---

## 2. Architecture

### 2.1 Base / teacher selection

`Qwen/Qwen3.6-35B-A3B` is the right starting point because the user's research goals (sparse MoE, sub-quadratic attention) are already satisfied in the base. Skipping Drop-Upcycling saves weeks of training. We use the teacher exclusively to produce distillation targets — we do not run it at inference time.

| Variant | Use |
|---|---|
| `Qwen/Qwen3.6-35B-A3B` (full BF16) | Teacher, runs CPU/streaming via the Exp5 streaming int4 loader during distillation |
| `unsloth/Qwen3.6-35B-A3B-GGUF` (Q8) | Backup teacher for fast judge calls during DPO data generation |
| `Qwen/Qwen3.6-27B` (dense) | Fallback teacher if the MoE is unstable as a teacher |

Why not 27B dense as the primary teacher? The MoE has stronger SWE-bench Verified (86.0 vs 77.2) and the 3 B active footprint matches the student's compute envelope, making representation alignment easier.

### 2.2 Student architecture

```
AVA v3 student (target 6–8 B total, ~1 B active routed + 0.4 B shared + 0.6 B attn/embed)
─────────────────────────────────────────────────────────────────────────────────
- 28 layers, hidden_size 1792, head_dim 128
- Layout per layer: (3 × Gated DeltaNet block + MoE-FFN, 1 × Gated Attention block + MoE-FFN), 7 repeats
- MoE-FFN per layer: 32 routed (ternary), 1 shared (BF16), top-4 active out of 32
- Routed expert intermediate dim: 768
- Shared expert intermediate dim: 4096
- Vocab size: 248,320 (inherited Qwen tokenizer)
- RoPE base 1,000,000 (matches Qwen3.6); YaRN extension to 256K → 1M
- Quantization: ternary {-1, 0, +1} for routed expert weights; BF16 for router, shared expert,
  attention Q/K/V/O, embeddings, normalization
- ParetoQ-style unified ternary QAT scheduler

Memory math at 1.58-bit:
  routed experts: 28 layers × 32 experts × 2 × 1792 × 768 × 1.58/8 = ~2.5 GB
  shared experts: 28 × 2 × 1792 × 4096 × 16/8                    = ~5.1 GB → keep on CPU
  attention + embeds (BF16):                                       ~1.0 GB
  KV cache (Q8 turboquant, 256K ctx):                              ~1.4 GB
GPU resident set (router + active routed experts + attn + KV):    ~3.6 GB → fits
```

Active params per token (after top-4 routing): ~1.4 B routed + 0.4 B shared + attention = ~2 B effective, but with full 6-8 B knowledge capacity.

### 2.3 Sub-quadratic attention

Already present in Qwen 3.6 base. Student inherits the (3 × Gated DeltaNet + 1 × Gated Attention) hybrid via FLA toolkit kernels (`fla-org/flash-linear-attention`).

Implementation path:
1. Use `transformers` 4.51+ which has Qwen3.6 model class registered
2. Replace student attention sublayers with FLA-provided Gated DeltaNet + GQA kernels
3. Triton kernels from FLA give linear-time decode + chunkwise prefill
4. RoPE applied only inside Gated Attention layers (Gated DeltaNet is content-based, not position-based)

### 2.4 Quantization (MoTE pattern)

| Component | Precision | Reason |
|---|---|---|
| Routed expert FFN weights | Ternary {-1, 0, +1}, 1.58 bit | MoTE; bulk of param count |
| Shared expert FFN weights | BF16 | Carries common knowledge; compresses badly |
| Router (gate) weights | BF16 | Tiny but routing-sensitive |
| Attention Q / K / V / O | BF16 | Outlier-sensitive; ternary QAT here causes catastrophic loss |
| Embeddings | BF16 | Tied with LM head; preserves token rank ordering |
| KV cache at inference | TurboQuant K4 / V2 (4-bit / 2-bit) | Reuse Exp5 tq_cache module |

We do not pursue true 1-bit (binary). PTQ at 1-bit collapses quality (BiLLM WikiText 27+ ppl). Ternary via BitDistiller distillation lands within 2-3 MMLU points of FP16 in published Qwen-class results.

### 2.4b HRM-Text recurrence inside each block

Each of the 24 blocks now contains a two-stack recurrence inspired by the Sapient HRM-Text architecture:

```
For each token at each block:
  z_H ← H-module(h_prev, token_embed)        # softmax attn + MoE, RoPE applied here, runs once
  z_L ← z_H
  for k in 1..6:
      z_L ← L-module(z_L, z_H, token_embed)  # Mamba-3 + MoE
      if halting_head(z_L) ≥ 0.5 and k ≥ 1: break
  block_out ← z_L
```

Halting is trained with an **ACT-style ponder loss** with weight 0.05; target mean L-steps/token ≈ 2.5, hard cap 6. Inference exposes a `reasoning_budget ∈ {1..6, "auto", "adaptive"}` knob. Full integration detail: [`../../docs/v3/HRM_TEXT.md`](../../docs/v3/HRM_TEXT.md).

The recurrence is the structural fix for AVA v2's math weakness (GSM8K 35 %, MATH-500 19 %). It lets the model perform multi-step reasoning in latent space without emitting chain-of-thought tokens. Mamba-3's complex-valued state in the L-module is what carries partial reasoning across iterations — see [`../../docs/v3/SUBQUADRATIC.md`](../../docs/v3/SUBQUADRATIC.md) §2 for why Mamba-3 specifically pairs with HRM.

### 2.5 Training pipeline (BitDistiller 3-stage)

```
Stage 0: Initialize student
  - Sparse-upcycle attention from Qwen 3.6 27B dense (drop-upcycling row/col init)
  - Random-init MoE FFN experts at BF16
  - Insert SubLN normalization before each ternary layer
  - Total budget: 0 GPU-hours, single-shot init

Stage 1: BF16 warmup (continued pretraining)
  - 2-3 B tokens of high-quality web + code + math
  - All weights BF16; 4 GB VRAM → must use gradient-checkpointing + LoRA + CPU optim
  - LoRA-only on routed experts; full fine-tune on router + shared expert
  - Loss: cross-entropy on next-token

Stage 2: Ternary QAT + teacher distillation
  - Quantize routed experts to ternary, keep BF16 master weights for gradient
  - ParetoQ unified QAT scheduler (1.58-bit cell)
  - Loss: 0.5 × CE + 0.3 × KL(teacher_logits || student_logits) + 0.2 × MiniLM-attn-distill
  - Teacher: Qwen 3.6 35B-A3B served via streaming int4 loader (Exp5 reuse), batched
  - 5-10 B tokens of SYNTHETIC-1 + Bespoke-Stratos + Tulu-3
  - 4 GB VRAM: ZeRO-2 CPU optim, micro-batch 1, grad-accum 32, seq 4096

Stage 3: SFT + DPO
  - SFT on the curated mix (700 K examples) at ternary
  - DPO on 75 K Skywork + UltraFeedback + SYNTHETIC-1-DPO pairs
  - Tool-discrimination SFT layer: 300 examples ("when NOT to call")
```

Estimated total token budget: 7-12 B tokens. At ~1500 tok/s on the laptop with QLoRA-style training on the active subset, ~50-90 wall days. Realistic only with reduced scope or rented A100s for stage 2.

---

## 3. Tool use (MCP)

Hypothesis (validated): high-quality TOOLS.md + JSON schema in system prompt + constrained decoding > heavy fine-tuning on tool-call traces. Reserve fine-tune budget for "when NOT to call."

### 3.1 Architecture

```
User → llama-server (chat completion w/ MCP client, merged Mar 2026)
     → Model emits <tool_call>{...}</tool_call>
     → XGrammar-style constrained decoding inside the tag (JSON-schema-enforced)
     → MCP client routes to FastMCP server via stdio
     → Server runs tool in sandbox (subprocess + Job Object on Windows)
     → Result injected as tool_result block
     → Model continues until no tool_call emitted → return
```

### 3.2 Folder layout

```
experiments/exp6_v3/mcp/
├── TOOLS.md                       # human source-of-truth tool catalog
├── README.md                      # MCP setup, security notes
├── server/
│   ├── main.py                    # FastMCP 3.0 stdio server entrypoint
│   ├── requirements.txt           # fastmcp>=3.0, RestrictedPython, httpx, ...
│   ├── tools/
│   │   ├── calc.py                # math expression evaluator
│   │   ├── python_exec.py         # RestrictedPython sandbox
│   │   ├── web_fetch.py           # httpx + domain allowlist
│   │   ├── file_ops.py            # path-scoped read / write
│   │   ├── memory_store.py        # JSON KV persistence
│   │   ├── code_exec.py           # subprocess + Windows Job Object
│   │   └── shell_exec.py          # restricted shell (deny-list)
│   └── audit.py                   # structured JSON logger
├── client/
│   ├── mcp_client.py              # stdio client wrapper
│   ├── tool_loop.py               # generate → detect → route → inject loop
│   ├── constrained.py             # llguidance / XGrammar schema-switch
│   └── tool_retriever.py          # ToolScope RAG (activate at >20 tools)
├── schemas/
│   ├── generate_schemas.py        # auto-export FastMCP tools → JSON-schema
│   └── generated/                 # do-not-edit machine output
├── eval/
│   ├── bfcl_runner.py             # BFCL v3 Prompt-mode regression gate
│   ├── mcp_bench_adapter.py       # MCP-Bench harness adapter
│   ├── tau_bench_adapter.py       # tau-bench / tau2-bench
│   └── fixtures/
└── sandbox/
    ├── Dockerfile                 # isolated container
    └── job_object_win.py          # Windows Job Object wrapper
```

### 3.3 Starter tool catalog

| Tool | Purpose | Sandbox |
|---|---|---|
| `calculator` | Math expressions, unit conversion | Pure Python, no I/O |
| `python_exec` | Untrusted Python | RestrictedPython |
| `code_exec` | General code | Subprocess + Job Object, CPU + memory cap |
| `web_fetch` | URL → markdown | httpx, domain allowlist |
| `file_read` / `file_write` | Path-scoped FS access | `root_paths` allowlist |
| `memory_get` / `memory_set` | JSON KV store | Per-session file |
| `shell_exec` | Safe shell | Command deny-list, no network |

Each tool entry in TOOLS.md follows:

```markdown
## tool: calculator
**When to use**: arithmetic, unit conversion, financial math.
**When NOT to use**: questions answerable from knowledge.
**Arguments**:
- `expression` (string, required): math expression. Supports +−×÷, **, sqrt, log, e.
**Returns**: string with numeric result.
**Example**: expression="(72 - 32) * 5 / 9" → "22.22"
```

JSON schemas auto-generated by `schemas/generate_schemas.py` from FastMCP type hints.

### 3.4 Constrained decoding

XGrammar (default for vLLM, SGLang) has < 40 µs/token overhead. llguidance is the llama-server-compatible alternative at ~50 µs/token. Implement state-machine wrapper:

1. Free decoding in normal generation
2. Detect `<tool_call>` token → switch to JSON-schema-constrained mask
3. After `</tool_call>` → switch back to free decoding

This gives 100 % valid JSON tool calls at zero parameter cost.

---

## 4. Training data

| Layer | Dataset | Size | License |
|---|---|---|---|
| Reasoning core | `lordx64/reasoning-distill-opus-4-7-max-sft` (Opus 4.7 traces) | 8 K | Apache 2.0 |
| Reasoning bulk | SYNTHETIC-1 (R1 distill) — math/code/STEM | 400-500 K | Apache 2.0 |
| Reasoning ultra | `bespokelabs/Bespoke-Stratos-17k` | 17 K | CC-BY-NC-4.0 |
| Math | OpenMathInstruct-2 (NVIDIA, Llama-3.1-405B traces) | sample 100 K | permissive |
| General SFT | `allenai/tulu-3-sft-mixture` | 150 K | Apache 2.0 |
| Tool use | `NousResearch/hermes-function-calling-v1` + BFCL v4 train | 50 K | MIT / CC-BY-4.0 |
| Long context | LongAlpaca-12k + ProLong-train | 30 K | Apache 2.0 |
| **SFT total** | | **~755 K** | mixed |
| DPO bulk | UltraFeedback binarized, Skywork curated | 60 K | CC-BY-NC-4.0 / Apache 2.0 |
| DPO reasoning | SYNTHETIC-1-DPO | 11 K | Apache 2.0 |
| **DPO total** | | **~75 K pairs** | mixed |
| Tool discrimination | hand-curated "when NOT to call" | 300 | local |

We deliberately exclude NuminaMath (GPT-4o output, restricted) from the commercial-permissive lane. We may include it in a separate research-only branch.

---

## 5. Inference stack

Reuse the Exp4 eval-v2 stack:

- llama.cpp head build (`experiments/exp5_gemma4/tools/llama.cpp-head/build/bin/llama-server.exe`) — already has Qwen 3.6 / qwen35 architecture support and MCP client (March 2026 PR)
- Q1_0 / Q2_K / TQ1_0 GGUF format for ternary weights
- TurboQuant K4 / V2 KV compression (Exp5 `tq_cache.py`)
- Flash-attention auto, Q8 KV, 4 parallel slots, continuous batching
- llama-server `--mcp-config` flag wires our FastMCP server at startup

Two-tier expert offload (FineMoE / FloE pattern):
- GPU resident: router + attention + currently-routed top-4 experts
- CPU resident: all 32 routed experts at int4, paged in on demand
- SpecMoEOff-style speculative prefetch for next layer

---

## 6. Evaluation

Reuse Exp4 eval-v2 runner; add new benchmarks. CI gates on every checkpoint:

| Bench | Frequency | Gate |
|---|---|---|
| BFCL v3 (Prompt-mode) | every checkpoint | acc ≥ 70 % |
| Full Exp4 v2 17-bench suite | weekly | no >2 pp regression on any |
| MCP-Bench (28 servers, 250 tools) | monthly | acc ≥ 60 % |
| Tau-Bench / tau2-bench | quarterly | task-success ≥ 50 % |
| RULER (8 K – 256 K) | every checkpoint | recall ≥ 90 % at 128 K |
| LongBench v2 | weekly | within 3 pp of Qwen 3.6 27B |

---

## 7. Risks and open problems

1. **Distillation token budget.** 5-10 B tokens at single-laptop throughput is 50-90 wall days. Likely needs rented A100s for stage 2 or aggressive corpus subsampling.
2. **Ternary + MoE quality cliff.** No published 1-bit MoE; ternary MoE (MoTE) is at 7 B scale. We are pushing into research territory.
3. **Teacher availability.** Qwen 3.6 35B-A3B teacher serving needs streaming-int4 + Exp5 infrastructure. Slow logit generation may dominate wall time.
4. **Sub-quadratic + ternary interaction.** Gated DeltaNet kernels were not designed for ternary weights; FLA toolkit may need a fork.
5. **MCP server security.** Code-exec and shell tools will run on the user's laptop. Any prompt-injection vulnerability lets a hostile prompt run arbitrary code. Strict sandbox + audit log is mandatory.
6. **Tool-discrimination overfit.** 300-example "when not to call" SFT can over-generalize to "never call." Need balanced positive/negative ratio + real-tool eval.
7. **Long-context recall under 1.58-bit.** Empirical, not yet measured. RULER + NIAH are the first checks.

---

## 8. Roadmap (numbered phases)

```
P0  Skeleton + design doc — DONE (May 2026)
P1  Pull Qwen 3.6 27B + 35B-A3B weights, register HF tokenizer + chat template
P2  Implement student architecture — IN PROGRESS (June 2026): engine/ modules landed
    (TernaryLinear group-256 QAT, MoTEFFN, Mamba-3 reference + FLA wrapper, HRM
    refinement recurrence w/ convergence-aware halting, AVAv3ForCausalLM, 8 smoke
    tests passing, 3.24 B params verified on meta device). Next: sparse-upcycle init
    from teacher + FLA kernel wiring on CUDA
P3  Stage 1 BF16 warmup pipeline (LoRA + CPU optim, 2-3 B tokens, CE only)
P4  Stage 2a HRM halting curriculum — train L-loop + halting head with ACT ponder loss (1-1.5 B tokens)
P5  Stage 2b ternary QAT + teacher distillation harness (5-8 B tokens, rented A100s)
P6  Stage 3a SFT on the curated mix (LoRA-only on routed; full FT on shared + router)
P7  Stage 3b DPO alignment on Skywork + UltraFeedback + SYNTHETIC-1-DPO
P8  Stage 3c tool-discrimination SFT (300-example "when NOT to call" set)
P9  Stage 4 ternary packing — export stock-llama.cpp GGUF (TQ1_0 + TQ2_0, Q8_0 embed)
P10 MCP wiring — FastMCP 3.0 + XGrammar constrained decoding + sandboxed tools
P11 Full eval — 17-bench v2 suite + BFCL v3 + MCP-Bench + RULER + LongBench v2
P12 Public release — HF model card, GGUF artifacts, README update, blog post
```

P0-P2 are AVA v3 design + scaffolding. P3-P5 are the heavy training (P5 is the rented-compute phase).
P6-P12 reuse / extend Exp4 eval and Exp5 inference infrastructure.

Detailed token budgets, hardware splits, and per-phase gates: [`../../docs/v3/RECIPE.md`](../../docs/v3/RECIPE.md). Risk register and fallback ladder: [`../../docs/v3/RISKS.md`](../../docs/v3/RISKS.md). Final pass/fail matrix: [`../../docs/v3/PERF_TARGETS.md`](../../docs/v3/PERF_TARGETS.md).

---

## 9. What this enables

- A 6-8 B ternary MoE student that runs on 4 GB VRAM at > 30 tok/s with > 256 K usable context
- Tool use via MCP (calc, python, web-fetch, file-ops, memory, shell, code-exec) without burning student capacity on call-format memorization
- Inheritance of Qwen 3.6 27B / 35B-A3B reasoning quality through KL distillation
- A reproducible, single-laptop training recipe other constrained labs can fork

If the targets are hit (MMLU 70, GSM8K 70, MMLU-Pro 50), AVA v3 would be the strongest sub-2 GB inference-footprint model on the public leaderboards.
