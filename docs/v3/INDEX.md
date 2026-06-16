# AVA v3 — Documentation Index

> **Status (10 June 2026): v3 pivoted to a CODING SPECIALIST — read [CODE_PIVOT.md](CODE_PIVOT.md) first.** One domain, all languages, ≤4 GB VRAM, trained for $0 on free compute (donor transplant + execution-verified self-play), free forever. Engine modules + smoke tests live in [`experiments/exp6_v3/engine/`](../../experiments/exp6_v3/engine/); the architecture below survives the pivot, the training plan and targets are superseded where CODE_PIVOT says so.

AVA v3 is the next major version of AVA. It is not a fine-tune. It is a new student model built from three converging 2026 research lines:

1. **HRM-Text** — Sapient Intelligence's brain-inspired two-stack latent-recurrent reasoner (1B params, ~40 B training tokens, ARC-C 81.9%, MMLU 60.7%, MATH 56.2%, DROP 82.2% — published April–May 2026).
2. **Mamba-3 / Gated DeltaNet** — generation-3 subquadratic attention with complex-valued state updates and MIMO formulation (ICLR 2026, +1.8 pp over Gated DeltaNet at 1.5 B).
3. **PrismML Bonsai** — Caltech's commercially-viable 1-bit / 1.58-bit family proving sub-2-bpw quality at 8 B (Mar–Apr 2026). v3 ships the same idea through **stock upstream llama.cpp TQ1_0/TQ2_0** (group-256 QAT) — no fork dependency.

Stacked on top of the AVA v2 distillation backbone (Qwen 3.6 35B-A3B teacher → ternary MoE student) and the MCP tool-use stack already designed in [`experiments/exp6_v3/DESIGN.md`](../../experiments/exp6_v3/DESIGN.md).

The hardware constraint is unchanged: **4 GB VRAM laptop, single card, no cloud, no cluster**. The deliverable target is a model that strictly dominates AVA v2 on every published benchmark, runs faster, holds longer context, and uses tools natively.

---

## Read this set in order

| # | Doc | What you'll learn |
|---|---|---|
| 0 | [CODE_PIVOT.md](CODE_PIVOT.md) | **THE current plan** — coding specialist, donor transplant, $0 training, self-improvement loop, eval tiers |
| 0b | [INVENTION.md](INVENTION.md) | The demands beyond benchmarks (D1–D9: CPU-only, calibrated honesty, air-gap, teach mode, energy ledger, glass-box decoding…) + novelty claims 1–8 |
| 0c | [RESEARCH_ROUND_2.md](RESEARCH_ROUND_2.md) | June-2026 round-2 sweep: multi-teacher panel, TurboQuant RAM tier, SIA-pattern self-play, Nemotron data, tokenizer transfer (E8), realism revision |
| 0d | [RESEARCH_ROUND_3.md](RESEARCH_ROUND_3.md) | The de-risk package: induction-aware T1 layer selection (implemented), repo-topo packing, decontamination gate (R21), verifier hardening (R20), EMA/GKD lanes — then C1, not more design |
| 0e | [RESEARCH_ROUND_4.md](RESEARCH_ROUND_4.md) | OpenMythos review: loop-index embedding (A16) + LTI-stable injection (A17) added to the HRM loop (E9, implemented, default-off); second link (system-prompt gist) off-topic |
| 0f | [RESEARCH_ROUND_5.md](RESEARCH_ROUND_5.md) | SubQ-1.1-Small review: external validation of the donor-transplant thesis at frontier scale; sample-level loss (implemented); recovery-phase discipline + multi-hop gate (R22); does NOT reopen the long-context cut |
| 0g | [RESEARCH_ROUND_6.md](RESEARCH_ROUND_6.md) | A v3-pattern hobby model in the wild (Gemma-4-12B coder): validates the $0 execution-verified multi-teacher recipe; tightens fallback to fresh-derivation (round-2 §3); base model not usable (12B, Gemma license) |
| 1 | [WHY_V3.md](WHY_V3.md) | One-page motivation: what v2 cannot do and why v3 needs a new architecture |
| 2 | [ARCHITECTURE_V3.md](ARCHITECTURE_V3.md) | Block-level diagram, dimensions, layer ratios, parameter budget, memory math |
| 3 | [HRM_TEXT.md](HRM_TEXT.md) | How the H-module / L-module dual recurrence integrates with the MoE student |
| 4 | [SUBQUADRATIC.md](SUBQUADRATIC.md) | Mamba-3 vs Gated DeltaNet evaluation, hybrid ratio, FLA kernel plan, RoPE placement |
| 5 | [PRISMML.md](PRISMML.md) | Sub-2-bpw packed storage: group-256 ternary QAT → stock TQ1_0/TQ2_0 GGUF export |
| 6 | [EDGES.md](EDGES.md) | **The compounding moat portfolio** — RAM-tier memory, latent superposition reasoning, halting-coupled speculation, NSA, precision tailwind, μP, self-play flywheel |
| 7 | [V2_GAP_ANALYSIS.md](V2_GAP_ANALYSIS.md) | Every v2 weakness mapped to the v3 mechanism that closes it |
| 8 | [PERF_TARGETS.md](PERF_TARGETS.md) | Beat-v2 matrix: every benchmark + speed/memory/context targets |
| 9 | [RECIPE.md](RECIPE.md) | End-to-end training pipeline P0..P12 with token budgets and stage losses |
| 10 | [RISKS.md](RISKS.md) | Research risks, ablation plan, and explicit fallback paths |
| 11 | [REFERENCES.md](REFERENCES.md) | Full citation list for every claim in this doc set |

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

Docs (this set) plus the P2 engine implementation: ternary QAT linear, MoTE FFN, Mamba-3 reference mixer, HRM refinement recurrence with convergence-aware halting, and the full `AVAv3ForCausalLM` assembly with CPU smoke tests (`experiments/exp6_v3/tests/`). No weights, no GGUF, no eval runs yet. Every design claim must be derivable from cited published work, the v2 evaluation already in the repo, or explicit research hypotheses marked as such.

## What is out of scope

- Multimodal (vision/audio) — deferred to a post-v3 branch.
- Verifiable RL post-training (DeepSeek-R1-style GRPO/DAPO) — deferred to v3.1.
- Mobile / phone-class deployment — deferred until v3 stable on laptop class.
