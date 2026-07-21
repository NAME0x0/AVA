# Experiment 6 — AVA v3

Sub-30 B ternary MoE assistant on 4 GB VRAM. Distilled from Qwen 3.6 35B-A3B teacher. Native sub-quadratic attention (Gated DeltaNet 3:1 hybrid). MCP-based tool use.

See [`DESIGN.md`](DESIGN.md) for the full architecture and training plan.

## Status

- **P0** scaffolding — in progress
- **P1+** training, distillation, tools, eval — pending

## Why this exists

AVA v2 (Exp4) is a Qwen 3.5 2B QLoRA fine-tune. Full eval (17 benchmarks, 52,027 evaluation instances) gave 82 % ARC-C / 59 % MMLU / 35 % GSM8K. Math, code, instruction-following are weak. Exp5 (Gemma 4 26B / E4B / E2B) is paused — files retained in `experiments/exp5_gemma4/`. AVA v3 pivots to a Qwen 3.6 base + 1-bit-class quant + MCP tools to push capacity per VRAM by ~5×.

## Layout

```
experiments/exp6_v3/
├── DESIGN.md              — full architecture and training plan
├── README.md              — this file
├── configs/               — YAML configs for each phase (warmup, qat, sft, dpo)
├── corpora/               — curated SFT + DPO mix manifests (to be built)
├── docs/                  — supplementary writeups, ablation logs
├── engine/                — student architecture: MoTE-FFN, ternary linear,
│                            SubLN, FLA Gated DeltaNet swap
├── mcp/                   — MCP server, client, tools, schemas, eval, sandbox
└── scripts/               — training / distillation / export scripts
```

## How exp6 relates to other experiments

| Exp | Track | State |
|---|---|---|
| Exp1-3 | scratch AVA systems | done, retained |
| Exp4 | Qwen 3.5 2B QLoRA (AVA v2) | released; full eval in `experiments/exp4_finetune/eval_v2/` |
| Exp5 | Gemma 4 26B / E4B / E2B inference research | paused; files retained |
| Exp6 (this) | AVA v3 — Qwen 3.6 ternary MoE student + MCP | active |

Reuse from prior experiments:
- Exp5 streaming int4 loader → teacher serving during distillation
- Exp5 TurboQuant tq_cache → KV compression at inference
- Exp5 llama.cpp head build → already supports Qwen 3.6 architecture and MCP client
- Exp4 eval-v2 runner → new benchmarks layer on top
