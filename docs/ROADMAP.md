# Roadmap

AVA's roadmap follows the constraint: every milestone must train and run on 4 GB VRAM (or degrade gracefully onto the same machine).

## Done

- [x] Scratch 14M model, retrieval, memory, tooling — Exp 1-3
- [x] QLoRA pipeline on Qwen 3.5 2B — Exp 4 / AVA v1
- [x] **AVA v2 released** — 82% ARC-C / 59% MMLU / 35% (44% k=5) GSM8K, full 17-benchmark eval
- [x] Sparse retrieval ensemble (91/299 ARC with support banks)
- [x] Gemma 4 26B local feasibility — Exp 5
- [x] Two-tier E2B/E4B local runtime — Exp 5
- [x] Open weights, open data, open code, open eval — all of v2

## Active — Exp 6 / AVA v3

Distill Qwen 3.6 35B-A3B → ~3 B ternary MoE student, layered on three 2026 research pillars:

1. **HRM-Text** — Sapient's two-stack latent-recurrent reasoner (H-module + L-module, halting head). Closes v2's MATH/GSM8K gap structurally without emitting CoT tokens.
2. **Mamba-3 (MIMO, complex SSM)** — ICLR 2026 generation-3 subquadratic attention; +1.8 pp over Gated DeltaNet at 1.5 B, half the state size of Mamba-2. Deployed in a 3 : 1 hybrid with gated softmax attention.
3. **PrismML Bonsai / Ternary Bonsai** — Caltech's 1-bit (1.125 bpw) and 1.58-bit packed formats, proving sub-2-bpw quality at 8 B. v3 ships the idea through **stock upstream llama.cpp `TQ1_0`/`TQ2_0`** (group-256 ternary QAT) — no fork dependency (June 2026 revision).

Plus the June 2026 **edge portfolio** ([`docs/v3/EDGES.md`](v3/EDGES.md)): RAM-tier product-key memory (factual capacity with zero VRAM cost — implemented), latent-superposition reasoning curriculum, halting-coupled speculative decoding, NSA long-context option, the precision-scaling tailwind, μP hyperparameter transfer, and an Absolute-Zero self-play flywheel.

> **Pivot (10 June 2026):** v3 is now a **coding specialist** — one domain, all
> languages, ≤4 GB VRAM, trained for **$0** on free compute (donor transplant from
> Qwen3-4B + trace distillation from Qwen3.6-27B + execution-verified self-play),
> free forever. Authoritative plan: [`docs/v3/CODE_PIVOT.md`](v3/CODE_PIVOT.md)
> (phases C0–C8 there supersede P0–P12 below for training).

Full v3 doc set: [`docs/v3/INDEX.md`](v3/INDEX.md). Design: [`docs/v3/ARCHITECTURE_V3.md`](v3/ARCHITECTURE_V3.md). Recipe: [`docs/v3/RECIPE.md`](v3/RECIPE.md). Targets: [`docs/v3/PERF_TARGETS.md`](v3/PERF_TARGETS.md).

Phases:

- **P0** Scaffolding (configs, engine stubs, MCP catalog, scripts) — *done*
- **P1** Download Qwen 3.6 35B-A3B teacher weights
- **P2** Student implementation + init — *in progress (June 2026)*: engine modules implemented (ternary QAT linear, MoTE FFN, Mamba-3 reference mixer, HRM refinement recurrence, full 3.24 B-param assembly + CPU smoke tests); sparse-upcycle init from teacher next
- **P3** BF16 warmup pretraining (2–3 B tokens)
- **P4** HRM halting curriculum — train L-loop and halting head with ACT ponder loss (1–1.5 B tokens)
- **P5** Ternary QAT + teacher distillation (5–8 B tokens, rented A100s)
- **P6** SFT on math + science + reasoning + tool-routing mix
- **P7** DPO alignment
- **P8** Tool-discrimination SFT (300 "when NOT to call" examples)
- **P9** Ternary packing — export to stock-llama.cpp GGUF (TQ1_0 + TQ2_0 builds, Q8_0 embeddings)
- **P10** MCP wiring (FastMCP 3.0 + XGrammar constrained decoding)
- **P11** Full eval — 17-bench v2 suite + BFCL v3 + MCP-Bench + RULER long context
- **P12** Public release

Target: **strict dominance** over v2 on every published benchmark, plus 32 K native context, MCP tools, and ~13× capacity per VRAM. See [`docs/v3/PERF_TARGETS.md`](v3/PERF_TARGETS.md) for the row-by-row gate.

## Planned (post-v3)

- [ ] Extended sequence length training (384 → 1024+ tokens)
- [ ] Verifiable RL post-training (math + science verifiers)
- [ ] Multimodal extension via compact vision encoder (Penguin-VL approach)
- [ ] Structured external memory for continual learning
- [ ] Multilingual fine-tuning starting with Urdu and Arabic
- [ ] On-device deployment via further quantization (target: phone-class hardware)

## Long-term

- General-purpose assistant entirely on consumer hardware
- Community-driven corpus contributions and benchmark extensions

## What's not on the roadmap

- Frontier-scale pretraining (no cluster, no budget — out of scope by design)
- Closed-weight releases (anything shipped will stay open)
- Cloud-only paths (local-first is the entire point)
