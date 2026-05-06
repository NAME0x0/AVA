# AVA v3 — Design Document

**Status:** Planning, May 2026
**Goal:** Build a SOTA-class assistant on a 4 GB VRAM laptop by combining (1) a Qwen 3.6 MoE teacher → ternary student distillation, (2) Gated DeltaNet hybrid sub-quadratic attention, (3) MoTE-style ternary expert weights with full-precision routers + shared expert, (4) MCP-based tool use via prompting + constrained decoding (no heavy tool fine-tuning), (5) frontier-distilled CoT corpus.
**Hardware budget:** RTX A2000 4 GB VRAM, 32 GB RAM, single-laptop training.
**Targets (stretch):** ≥ MMLU 70 %, ≥ GSM8K 70 %, ≥ MMLU-Pro 50 %, ≥ HumanEval+ 50 %, ≥ IFEval 65 %, ≥ 256 K context, ≥ 30 tok/s decode.

This document supersedes the AVA v3 plan in `plans/linked-growing-mccarthy.md` (Gemma 4 E4B / E2B). That direction is paused. Exp5 files retained.

---

## 1. Executive summary

| Pillar | Choice | Why |
|---|---|---|
| Base / teacher | `Qwen/Qwen3.6-35B-A3B` (256 experts, 8+1 active, native Gated DeltaNet, 256 K ctx, Apache 2.0) | Already MoE + sub-quadratic; matches every requirement; latest Qwen team release |
| Student size | 6 B–8 B params, ternary (1.58-bit) MoE | Only viable size at 4 GB VRAM after Bonsai/BitNet research |
| Attention | Gated DeltaNet (3:1 hybrid w/ Gated Attention) | Inherited from Qwen 3.6, validated long-context recall, beats Mamba-2/RWKV-7 on RULER |
| Quantization | MoTE pattern: ternary FFN experts + BF16 router + 1 BF16 shared expert per layer | Best published 1-bit + MoE combination |
| Training method | BitDistiller 3-stage QAT + distillation from Qwen3.6-35B-A3B teacher | PTQ to 1-bit fails catastrophically; QAT-from-scratch needs trillions of tokens |
| Corpus | SYNTHETIC-1 + Bespoke-Stratos-17k + Tulu-3 + Opus-4.7 distill + OpenMathInstruct-2 + Hermes-FC + Skywork-Reward | ~725 K SFT + 75 K DPO, all Apache-2.0 / MIT |
| Tool use | TOOLS.md + FastMCP 3.0 server + XGrammar constrained decoding + small discrimination SFT (300 examples) | BFCL evidence: prompting + constraints beats heavy FT for small models |
| Inference | llama.cpp head build (already has MCP client merged Mar 2026) + llama-server | Reuses Exp4-eval infrastructure; native MCP loop |
| Evaluation | BFCL v3 Prompt-mode, MCP-Bench, Tau-Bench, full Exp4-v2 17-bench suite, RULER long-context | Regression-gated CI |

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
P0  Skeleton + design doc (this PR)
P1  Pull Qwen 3.6 27B + 35B-A3B weights, register HF tokenizer + chat template
P2  Implement student architecture in src/ava/v3/ (FLA Gated DeltaNet swap, MoTE-FFN, SubLN)
P3  Stage 1 BF16 warmup pipeline (LoRA + CPU optim)
P4  Stage 2 ternary QAT + distillation harness (teacher logit cache, KL loss, ParetoQ scheduler)
P5  Stage 3 SFT + DPO (existing trl pipeline + corpus mix)
P6  Tool-discrimination SFT (300-example set hand-curated)
P7  MCP server + client + TOOLS.md (FastMCP 3.0)
P8  Eval harness extensions: BFCL v3, MCP-Bench, RULER long context
P9  llama.cpp Q1_0 / TQ1_0 export + Modelfile
P10 RC eval pass (full 17-bench + new gates)
P11 Public release: HF model card, GGUF artifacts, README update, blog post
```

P0-P2 are AVA v3 design + scaffolding. P3-P5 are the heavy training.
P6-P10 reuse / extend Exp4 eval and Exp5 inference infrastructure.

---

## 9. What this enables

- A 6-8 B ternary MoE student that runs on 4 GB VRAM at > 30 tok/s with > 256 K usable context
- Tool use via MCP (calc, python, web-fetch, file-ops, memory, shell, code-exec) without burning student capacity on call-format memorization
- Inheritance of Qwen 3.6 27B / 35B-A3B reasoning quality through KL distillation
- A reproducible, single-laptop training recipe other constrained labs can fork

If the targets are hit (MMLU 70, GSM8K 70, MMLU-Pro 50), AVA v3 would be the strongest sub-2 GB inference-footprint model on the public leaderboards.
