# Training Recipe

> **Read first:** [ARCHITECTURE_V3.md](ARCHITECTURE_V3.md), [HRM_TEXT.md](HRM_TEXT.md), [SUBQUADRATIC.md](SUBQUADRATIC.md), [PRISMML.md](PRISMML.md).

End-to-end pipeline for AVA v3, phase by phase. Each phase has: inputs, outputs, training objective, token budget, hardware, success gate, and known risks. The pipeline supersedes the 11-phase plan in [`experiments/exp6_v3/DESIGN.md`](../../experiments/exp6_v3/DESIGN.md) §8 by adding phases for HRM halting-head training, Mamba-3 init, and PrismML packing.

---

## P0 — Scaffolding (in progress)

| | |
|---|---|
| Inputs | Empty `experiments/exp6_v3/` directory |
| Outputs | Configs, engine stubs, MCP catalog, scripts (already done) |
| Token budget | 0 |
| Hardware | None (CPU edit) |
| Gate | All Python imports resolve; `mypy src tests` green |
| Status | **Done** as of May 2026 |

## P1 — Teacher acquisition

| | |
|---|---|
| Inputs | Hugging Face credentials |
| Outputs | `Qwen/Qwen3.6-35B-A3B` weights cached locally; tokenizer + chat template registered |
| Token budget | 0 |
| Hardware | 60 GB free disk; 32 GB RAM (BF16 streamed) |
| Gate | Teacher emits one valid token on a smoke prompt |
| Script | `experiments/exp6_v3/scripts/fetch_teacher.py` |
| Risks | HF rate limit, disk space |

## P2 — Student initialization

| | |
|---|---|
| Inputs | Qwen 3.6 27B dense weights; v3 student config |
| Outputs | BF16 student checkpoint with sparse-upcycled attention, random MoE FFN, Mamba-3 init, HRM halting head (random init) |
| Token budget | 0 |
| Hardware | Laptop; single-shot |
| Gate | Forward pass produces valid logits; no NaN; halting head outputs σ ∈ (0,1) |
| Notes | **Mamba-3 init:** complex state initialized with phase ∼ U[-π, π]; magnitude ∼ U[0, 0.01]. **HRM halting head:** zero-init bias, small Gaussian weights — initially halts every token at k=1, so the L-loop is a no-op until trained |

## P3 — Stage 1: BF16 warmup (continued pretraining)

| | |
|---|---|
| Inputs | Student from P2; 2–3 B tokens of high-quality web + code + math (DCLM-baseline + StarCoder + OpenWebMath) |
| Outputs | Stable BF16 student trained on next-token CE |
| Loss | `CE` only (no halting loss yet) |
| Token budget | 2–3 B |
| Sequence length | 32 768 |
| Hardware | 4 GB VRAM — uses LoRA on routed experts, full FT on router + shared expert + Mamba-3 projections; ZeRO-2 CPU optim, micro-batch 1, grad-accum 32 |
| Wall time | ~30 wall-days on laptop, ~1 day on 4× A100 |
| Gate | Train-loss converges below baseline; ARC-E zero-shot ≥ 85 % |
| Risks | Mamba-3 instability (complex state needs careful normalization — Mamba-3 paper §5.1 recipe) |

## P4 — Stage 2a: HRM halting curriculum

| | |
|---|---|
| Inputs | Stage 1 student |
| Outputs | Student with active L-loop; halting head trained |
| Loss | `CE + 0.05 · ponder_loss(halting_probs)` |
| Token budget | 1–1.5 B |
| Sequence length | 32 768 |
| Hardware | Same as P3 |
| Wall time | ~10 wall-days laptop |
| Gate | Mean L-steps/token converges to ~2.5; MATH-500 zero-shot ≥ 25 % (vs v2 18.8 %) at `reasoning_budget=auto` |
| Notes | First 10 % of P4 keeps halting head frozen at "always halt at k=1"; gradually unfreezes. This protects Stage 1's pre-trained CE loss from being destabilized by an unbounded recurrence on day one |
| Risks | Ponder collapse (always k=1 or always k=max). Mitigation: clip ponder loss; clamp budget |

## P5 — Stage 2b: Ternary QAT + teacher distillation

| | |
|---|---|
| Inputs | P4 student |
| Outputs | Ternary-aware student; routed + shared experts at fake-quant ternary with **group-256 scales** (bit-exact with the TQ1_0/TQ2_0 export — see [PRISMML.md](PRISMML.md) §2) |
| Loss | `0.5·CE + 0.3·KL(teacher_logits || student_logits) + 0.15·MiniLM_attn_distill + 0.05·ponder_loss` |
| Token budget | **5–8 B** |
| Sequence length | 4 096 (shorter than P3/P4 to fit batched teacher logits) |
| Hardware | Stage 2 is the gating phase. Laptop is ~50–90 wall days; rented 4× A100 cluster is ~5–10 wall days. **Plan: rent A100s for P5** |
| Teacher serving | Streaming int4 loader from Exp 5; pre-compute and cache teacher logits in shards of 64 K tokens |
| Gate | MMLU 5-shot ≥ 65 %; GSM8K greedy ≥ 50 %; MATH-500 ≥ 35 % |
| Risks | Ternary MoE quality cliff (no published 1.58-bit MoE at >7 B). Fallback: keep routed at BF16; only shared expert goes ternary (worse memory, but ships) |

## P6 — Stage 3a: SFT

| | |
|---|---|
| Inputs | P5 student |
| Outputs | Instruction-tuned ternary student |
| Loss | `CE` on SFT pairs; halting head kept active; ponder weight 0.02 |
| Token budget | ~1.5 B (755 K examples × ~2 K tokens) |
| Sequence length | 8 192 |
| Hardware | Laptop OK (single epoch over the SFT mix) |
| Wall time | ~5 wall-days laptop |
| Mix | SYNTHETIC-1 (450 K) + Tulu-3 (150 K) + OpenMathInstruct-2 sample (100 K) + Hermes-FC + BFCL (50 K) + Bespoke-Stratos (17 K) + ProLong + LongAlpaca (30 K) + Opus 4.7 distill (8 K) |
| Gate | IFEval strict ≥ 55 % (interim); MMLU 5-shot non-regression vs P5 |
| Risks | Catastrophic forgetting of pre-training. Mitigation: LoRA-only on routed experts during SFT; full FT only on router + shared expert |

## P7 — Stage 3b: DPO alignment

| | |
|---|---|
| Inputs | P6 student |
| Outputs | DPO-aligned student |
| Loss | DPO (β = 0.1) on preference pairs |
| Pairs | UltraFeedback (40 K) + Skywork-Reward (20 K) + SYNTHETIC-1-DPO (11 K) = ~71 K pairs |
| Hardware | Laptop |
| Wall time | ~2 wall-days |
| Gate | IFEval strict ≥ 65 %; TruthfulQA MC1 ≥ 55 %; non-regression on math benches |

## P8 — Stage 3c: Tool-discrimination SFT

| | |
|---|---|
| Inputs | P7 student |
| Outputs | Tool-aware student that knows when NOT to call |
| Examples | 300 hand-curated "looks like a tool call but isn't" cases |
| Token budget | ~600 K |
| Hardware | Laptop |
| Wall time | ~hour |
| Gate | Agentic GSM8K call rate ∈ [20 %, 40 %] (was 0.6 % in v2); BFCL v3 acc ≥ 65 % |

## P9 — Stage 4: PrismML packing (export)

| | |
|---|---|
| Inputs | P8 BF16 student (ternary-aware but still float storage) |
| Outputs | Two GGUF builds from one checkpoint: `TQ1_0` (smallest) and `TQ2_0` (fastest) expert tensors, Q8_0 embeddings — stock llama.cpp formats |
| Token budget | 0 (post-train) |
| Hardware | Laptop CPU pack; no training |
| Gate | Round-trip dequant matches BF16 within rounding; MMLU non-regression ≤ 1.5 pp vs BF16; resident GPU mem ≤ 3.6 GB |
| Script | `experiments/exp6_v3/scripts/export_gguf.py` |
| Risks | None on kernel availability (TQ1_0/TQ2_0 upstream since 2024 — R8 closed June 2026). Residual risk: TQ1_0 kernel rounding vs QAT fake-quant; gate covers it |

## P10 — MCP wiring

| | |
|---|---|
| Inputs | Packed GGUF + FastMCP 3.0 server + tool catalog |
| Outputs | `llama-server --mcp-config ...` end-to-end loop functional |
| Tools | calculator, python_exec, code_exec, web_fetch, file_read, file_write, memory_get, memory_set, shell_exec |
| Constrained decoding | XGrammar-style JSON-schema mask switched at `<tool_call>` |
| Gate | All 9 tools fire correctly on smoke tests; sandbox audit log clean |

## P11 — Full evaluation pass

| | |
|---|---|
| Eval suite | Exp 4 v2 17-bench runner extended with: BFCL v3 (Prompt), MCP-Bench, RULER (8 K to 256 K), LongBench v2 |
| Gate | **Every** v2 benchmark non-regressed; targets in [`PERF_TARGETS.md`](PERF_TARGETS.md) hit |
| Wall time | ~3 wall-days laptop (Q-packed inference is fast) |

## P12 — Public release

| | |
|---|---|
| Artifacts | HF model card, GGUF, Modelfile, demo Space, blog post, updated README |
| Channels | `NAME0x0/AVA-v3` (LoRA + BF16 if applicable), `NAME0x0/AVA-v3-GGUF` |
| Cite | Update `CITATION.cff` to v3 |

---

## Token budget summary

| Phase | Tokens | Cumulative |
|---|---:|---:|
| P3 BF16 warmup | 3 B | 3 B |
| P4 HRM halting curriculum | 1.5 B | 4.5 B |
| P5 QAT + distillation | 8 B | 12.5 B |
| P6 SFT | 1.5 B | 14 B |
| P7 DPO | 0.15 B | 14.15 B |
| P8 tool discrimination | 0.001 B | 14.15 B |
| **Total** | | **~14 B** |

For comparison: AVA v2 consumed ~10 M training tokens (20 741 examples × ~500 tokens). v3 is ~1 400× more compute — most of it in P5 distillation. The 4 GB inference target is unchanged.

---

## Hardware footprint summary

| Phase | Laptop (RTX A2000 4 GB) | A100 4× cluster | Decision |
|---|---|---|---|
| P0–P2 | yes | n/a | Laptop |
| P3 warmup | 30 days | 1 day | **Laptop** (free) |
| P4 halting | 10 days | <1 day | **Laptop** (free) |
| P5 QAT distill | 50–90 days | 5–10 days | **A100 cluster** (rented, ~$2 000–3 000) |
| P6–P8 SFT / DPO / tool | 7 days | <1 day | **Laptop** |
| P9 packing | hour | n/a | Laptop |
| P10–P12 | days | n/a | Laptop |

The recipe ships under "trained on a single laptop" — except for P5, which is the only phase the laptop physically cannot do at reasonable speed. We document P5 rental cost openly in the release.

---

## Reproducibility invariants

All phases log:

- Exact dataset shard hashes.
- All hyperparameters from a single YAML config (one per phase).
- Random seed (fixed per phase, documented per checkpoint).
- Loss curves saved as JSONL.
- Eval results on a fixed eval subset every 1000 steps.

So that a third party can reproduce v3 by running the same scripts against the same dataset hashes.

---

## Risks table (P3..P9)

See [RISKS.md](RISKS.md) for the full version.

| Phase | Top risk | Mitigation |
|---|---|---|
| P3 | Mamba-3 instability at BF16 | Use Mamba-3 paper §5.1 init + low LR for first 5 % of phase |
| P4 | Ponder collapse | Clip ponder loss; clamp budget; freeze halting head if MATH-500 regresses |
| P5 | Ternary MoE quality cliff | Fallback: keep routed at BF16, only shared at ternary |
| P9 | TQ1_0 kernel rounding vs QAT fake-quant | Round-trip test in export script; ship TQ2_0-only if MMLU drops > 1.5 pp (kernel availability risk closed June 2026 — R8) |
