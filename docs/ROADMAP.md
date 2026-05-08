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

Distill Qwen 3.6 35B-A3B → 6-8B ternary MoE student with native Gated DeltaNet 3:1 hybrid + MCP tools.

Phases:

- **P0** Scaffolding (configs, engine stubs, MCP catalog, scripts) — *in progress*
- **P1** Download Qwen 3.6 35B-A3B teacher weights
- **P2** Implement ternary linear, MoTE-FFN, SubLN, FLA Gated DeltaNet swap
- **P3** BF16 warmup pretraining
- **P4** Mixed-precision intermediate stage
- **P5** Ternary QAT distillation from teacher logits (BitDistiller stage 3)
- **P6** SFT on a math + science + reasoning + tool-routing mix
- **P7** DPO alignment
- **P8** MCP server (FastMCP 3.0 + XGrammar constrained decoding) + tool catalog
- **P9** GGUF export, llama.cpp serving
- **P10** Full 17-benchmark eval + new tool-routing benchmarks
- **P11** Public release

Target: capacity-per-VRAM ≈ 5× v2 with materially better math + tool routing.

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
