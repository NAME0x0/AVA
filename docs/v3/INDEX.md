# AVA v3 — Documentation Index

> **Status:** P0 scaffolding (May 2026). Design-stage docs only — no v3 weights yet. See [docs/ROADMAP.md](../ROADMAP.md) for phase progress.

AVA v3 is the next major version of AVA. It is not a fine-tune. It is a new student model built from three converging 2026 research lines:

1. **HRM-Text** — Sapient Intelligence's brain-inspired two-stack latent-recurrent reasoner (1B params, ~40 B training tokens, ARC-C 81.9%, MMLU 60.7%, MATH 56.2%, DROP 82.2% — published April–May 2026).
2. **Mamba-3 / Gated DeltaNet** — generation-3 subquadratic attention with complex-valued state updates and MIMO formulation (ICLR 2026, +1.8 pp over Gated DeltaNet at 1.5 B).
3. **PrismML Bonsai** — Caltech's commercially-viable 1-bit / 1.58-bit family with packed 1.125 bpw storage, FP16 group scales (128), and merged llama.cpp / MLX kernels (Mar–Apr 2026).

Stacked on top of the AVA v2 distillation backbone (Qwen 3.6 35B-A3B teacher → ternary MoE student) and the MCP tool-use stack already designed in [`experiments/exp6_v3/DESIGN.md`](../../experiments/exp6_v3/DESIGN.md).

The hardware constraint is unchanged: **4 GB VRAM laptop, single card, no cloud, no cluster**. The deliverable target is a model that strictly dominates AVA v2 on every published benchmark, runs faster, holds longer context, and uses tools natively.

---

## Read this set in order

| # | Doc | What you'll learn |
|---|---|---|
| 1 | [WHY_V3.md](WHY_V3.md) | One-page motivation: what v2 cannot do and why v3 needs a new architecture |
| 2 | [ARCHITECTURE_V3.md](ARCHITECTURE_V3.md) | Block-level diagram, dimensions, layer ratios, parameter budget, memory math |
| 3 | [HRM_TEXT.md](HRM_TEXT.md) | How the H-module / L-module dual recurrence integrates with the MoE student |
| 4 | [SUBQUADRATIC.md](SUBQUADRATIC.md) | Mamba-3 vs Gated DeltaNet evaluation, hybrid ratio, FLA kernel plan, RoPE placement |
| 5 | [PRISMML.md](PRISMML.md) | 1.125 bpw packed storage, group-scale layout, llama.cpp / MLX export path |
| 6 | [V2_GAP_ANALYSIS.md](V2_GAP_ANALYSIS.md) | Every v2 weakness mapped to the v3 mechanism that closes it |
| 7 | [PERF_TARGETS.md](PERF_TARGETS.md) | Beat-v2 matrix: every benchmark + speed/memory/context targets |
| 8 | [RECIPE.md](RECIPE.md) | End-to-end training pipeline P0..P12 with token budgets and stage losses |
| 9 | [RISKS.md](RISKS.md) | Research risks, ablation plan, and explicit fallback paths |
| 10 | [REFERENCES.md](REFERENCES.md) | Full citation list for every claim in this doc set |

If you only have five minutes: read [WHY_V3.md](WHY_V3.md) and [PERF_TARGETS.md](PERF_TARGETS.md).

---

## Where this doc set fits

- **High-level repo orientation:** [../INDEX.md](../INDEX.md)
- **v2 results** (the bar v3 must beat): [../RESULTS.md](../RESULTS.md)
- **v2 cross-model comparison:** [../COMPARE.md](../COMPARE.md)
- **Experiment progression v1→v2→v3:** [../EXPERIMENTS.md](../EXPERIMENTS.md)
- **Active scaffolding for v3:** [`../../experiments/exp6_v3/`](../../experiments/exp6_v3/)
- **Original v3 design doc** (architecture + training, will be patched to reflect HRM/Mamba-3/PrismML additions): [`../../experiments/exp6_v3/DESIGN.md`](../../experiments/exp6_v3/DESIGN.md)

---

## What is in scope right now

This is a documentation-only milestone. No code, no weights, no GGUF, no eval runs yet. Every claim in this set must be derivable from cited published work, the v2 evaluation already in the repo, or explicit research hypotheses marked as such.

## What is out of scope

- Multimodal (vision/audio) — deferred to a post-v3 branch.
- Verifiable RL post-training (DeepSeek-R1-style GRPO/DAPO) — deferred to v3.1.
- Mobile / phone-class deployment — deferred until v3 stable on laptop class.
