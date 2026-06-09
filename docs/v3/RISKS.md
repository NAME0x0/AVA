# Risks and Fallbacks

Every claim in the v3 plan rests on at least one piece of research that is either fresh in 2026 or extrapolated beyond its published scale. This document lists those bets, the failure modes, and the explicit fallback path for each. No v3 phase starts without a documented exit plan.

The rows are ordered by **stage at which the risk first manifests** — earliest first. The mitigation columns assume the laptop budget; if rented A100s are available, several fallbacks shorten.

---

## 1. Architecture-level risks (manifest by P3)

### R1 — HRM-Text recipe does not transfer to text generation with a Qwen base

| | |
|---|---|
| Risk | HRM-Text (Sapient, May 2026) was reported at 1 B with custom training and ~40 B tokens. Published verification is independent but limited to MMLU/ARC-C/MATH/DROP. The recipe could fail to generalize when grafted onto a Qwen-derived MoE with BitDistiller QAT |
| First detected at | P4 (halting curriculum) — if mean L-steps collapses to 1 or to 6, or if loss diverges |
| Severity if untreated | High — kills the math story; v3 wouldn't beat v2 by enough |
| Probability | Medium |
| Mitigation | Ablation A1 in [HRM_TEXT.md](HRM_TEXT.md) §6: drop the L-loop entirely (set `reasoning_budget=1`). v3 still ships, just without the recurrence-driven MATH gain. Targets in [PERF_TARGETS.md](PERF_TARGETS.md) are recomputed against a 28 %-MATH baseline instead of 50 %, but every v2 row still passes via distillation alone |

### R2 — Mamba-3 (MIMO, complex SSM) is unstable at BF16

| | |
|---|---|
| Risk | Complex-valued state update is new in Mamba-3 (ICLR 2026). BF16 has ~7 mantissa bits, which is at the edge for stable complex rotations |
| First detected at | P3 (warmup) — sudden loss spikes or NaN in the Mamba layer logs |
| Severity | High — Mamba-3 is required for the speed and long-context targets |
| Probability | Medium-low |
| Mitigation | Mamba-3 paper §5.1 init recipe + per-layer norm clip. **Fallback:** swap all L-block Mamba-3 sublayers for Gated DeltaNet (lose ~1.8 pp on average bench but architecture still works). One-line config flip in `experiments/exp6_v3/configs/v3_student_arch.yaml` |

### R3 — Halting head collapse (ponder loss not well-tuned)

| | |
|---|---|
| Risk | Adaptive computation time literature is well-known for unstable budgets — every L-step or always one L-step |
| First detected at | P4 — mean L-steps logs |
| Severity | Medium — model still trains, but reasoning_budget knob loses meaning |
| Probability | Medium |
| Mitigation | Three independent guards: (a) ponder weight clamped to [0.01, 0.10] tuned by sweep; (b) explicit "halt budget" cap of 6; (c) ablation A4 over weights 0.01 / 0.05 / 0.10. If still collapsed: freeze halting at fixed N=2 and ship with constant budget |

---

## 2. Training-stage risks (manifest by P5)

### R4 — Ternary + MoE quality cliff

| | |
|---|---|
| Risk | No published 1.58-bit MoE above 7 B. v3 is at 3.26 B total / 32 experts. Quality could collapse 5+ pp on MMLU during stage 2 QAT |
| First detected at | P5 — mid-stage eval at 50 % of token budget |
| Severity | Critical — kills the headline "4 GB capacity per VRAM" win |
| Probability | Medium |
| Mitigation | **Fallback A**: keep routed experts at BF16; quantize only the shared expert. Cost: routed expert storage goes from 0.30 GB to ~4.2 GB on disk; resident GPU memory rises by ~0.5 GB but still fits. **Fallback B**: drop to top-2 routing instead of top-4 — halves active routed footprint. **Fallback C** (worst case): drop MoE entirely, ship as a dense 1.8 B ternary student (still uses HRM + Mamba-3, still beats v2 on math/code) |

### R5 — Teacher logit caching dominates wall time

| | |
|---|---|
| Risk | Streaming int4 Qwen 3.6 35B-A3B inference at the laptop is ~0.3 tok/s. Pre-computing 8 B teacher logit shards takes >2 wall-years on laptop alone |
| First detected at | P5 setup |
| Severity | High — makes P5 impossible without rented compute |
| Probability | High — already known |
| Mitigation | This is the planned A100 rental phase in [RECIPE.md](RECIPE.md) §P5. Estimated cost: $2 000–3 000 to compute and cache the full 8 B teacher logit corpus once. Cache reused across all subsequent ablations |

### R6 — Distillation token budget exceeds rental compute

| | |
|---|---|
| Risk | 8 B tokens at 4× A100 80GB ≈ 5–10 days. Cost ~$2 500 at on-demand rates. Budget slip |
| First detected at | P5 mid-phase |
| Severity | Medium — affects schedule, not feasibility |
| Mitigation | Subsample SYNTHETIC-1 to 250 K examples, OpenMathInstruct-2 to 50 K. Total budget compresses to 4 B. Math/code targets drop by 3–5 pp; still beats v2 |

### R7 — Catastrophic forgetting during P6 SFT

| | |
|---|---|
| Risk | LoRA on routed experts + full FT on shared expert can overwrite teacher-distilled knowledge |
| First detected at | P6 — MMLU regresses below P5 |
| Severity | High |
| Mitigation | LoRA-only on shared expert during SFT (RECIPE P6 baseline). EWC penalty against P5 weights on routed router. Fall back to LoRA-everywhere if shared-expert FT regresses |

---

## 3. Deployment risks (manifest by P9)

### R8 — PrismML kernels not upstream (CONFIRMED 9 June 2026 → design pivoted, risk closed)

| | |
|---|---|
| Risk | Original plan assumed BB1_0 was merged into llama.cpp head. June 2026 verification: Bonsai kernels live only in the `PrismML-Eng/llama.cpp` fork (`prism` branch); upstream PRs pending. Worse, Bonsai uses group-128 scales while upstream `TQ1_0`/`TQ2_0` use group-256 — formats not interchangeable (llama.cpp discussion #22019) |
| First detected at | Pre-P2 research check (this revision) — caught before any code depended on it |
| Severity | Was low-medium; now none |
| Probability | Materialized |
| Resolution | **Pivot, not fallback**: QAT now trains with group-256 scales natively, so the checkpoint exports to stock upstream `TQ1_0` (1.6875 bpw) / `TQ2_0` (2.0625 bpw) with zero fork dependency. Routed expert storage 0.45 GB; total packed ~1.70 GB with Q8_0 embeddings; resident ~3.3 GB. BB1_0 demoted to opportunistic stretch (−0.30 GB) if PrismML's upstream PR lands before P9. See [PRISMML.md](PRISMML.md) §2 |

### R9 — Mamba-3 kernel not in FLA toolkit

| | |
|---|---|
| Risk | Mamba-3 PR in `fla-org/flash-linear-attention` may not land before we need P3 |
| Status | **Closed (June 2026):** `fla.layers.mamba3` merged into FLA in April 2026 and verified present at head. Gated DeltaNet remains the config-flip fallback only for stability issues (R2), not availability |

### R10 — Ternary Bonsai license restriction on shared expert

| | |
|---|---|
| Risk | If PrismML changes their license terms on Ternary Bonsai weights or kernels, we cannot redistribute v3 freely |
| Severity | Low — we are using the *format*, not the *weights* |
| Mitigation | We do not redistribute any PrismML weights. Our shared expert is BitDistiller-trained from Qwen 3.6 teacher; only the storage format is PrismML's. No license obligation to PrismML |

---

## 4. MCP / tool-use risks (manifest by P10)

### R11 — Tool discrimination overfits to "always call"

| | |
|---|---|
| Risk | The 300-example "when not to call" set may bias the model toward over-refusal of tools |
| First detected at | P8 — agentic GSM8K call rate < 10 % or > 50 % |
| Mitigation | Balanced positive/negative ratio (1:1) in the hand-curated set; real-tool eval before P10 |

### R12 — Sandboxed code-exec breakout

| | |
|---|---|
| Risk | A hostile prompt + a model that fires `code_exec` could execute arbitrary code on the user's laptop |
| Severity | Critical (security) |
| Mitigation | Job Object sandbox on Windows; Docker sandbox on Linux/macOS; default deny network; audit log mandatory; never run as admin. See [`experiments/exp6_v3/mcp/sandbox/`](../../experiments/exp6_v3/mcp/sandbox/) |

### R13 — XGrammar / llguidance constrained-decoding overhead exceeds budget

| | |
|---|---|
| Risk | Constrained decoding adds 40–50 µs per token. At 30 tok/s decode that's ~1.5 ms/token added overhead — small in absolute terms but visible in throughput dashboards |
| Severity | Low |
| Mitigation | Only activate constrained mask inside `<tool_call>...</tool_call>` tags — typical tool call is < 100 tokens, so total overhead per tool use is < 5 ms |

---

## 5. Evaluation risks (manifest by P11)

### R14 — RULER long-context recall drops below 90 % at 128 K

| | |
|---|---|
| Risk | The 3 : 1 hybrid was validated for softmax/Gated DeltaNet (Qwen 3.6 numbers). Mamba-3 substitution may shift the recall curve |
| Severity | Medium — long-context is a release claim |
| Mitigation | Reduce advertised context to 64 K; YaRN to 128 K only. Release notes explicitly mark 256 K / 1 M as "experimental" if recall is borderline |

### R15 — Benchmarks shift between v2 eval and v3 eval (drift)

| | |
|---|---|
| Risk | If we update lm-evaluation-harness or any benchmark loader between v2 and v3, "the same number" may not be the same number |
| Mitigation | Re-run v2 GGUF on v3's eval pipeline at P11 to establish a fresh baseline. If v2 number changes by > 1 pp on any benchmark, document the delta in [`docs/RESULTS.md`](../RESULTS.md) and update the gap-analysis table |

---

## 6. Schedule risks (manifest at any phase)

### R16 — Single-person bandwidth

| | |
|---|---|
| Risk | The pipeline assumes one engineer (the author). Any extended absence stalls everything |
| Mitigation | Each phase has a self-contained YAML config + script. A second person could resume from any checkpoint. Sponsorship via GitHub Sponsors is the funded path |

### R17 — Hardware loss (laptop failure)

| | |
|---|---|
| Risk | All work to date is on a single laptop. SSD or GPU failure destroys local-only state |
| Mitigation | Every checkpoint and dataset shard is pushed to Hugging Face private repos at end of each phase; teacher logits cached to S3-compatible bucket on the rental side |

---

## 7. Aggregate fallback ladder

If multiple risks trigger, the v3 "minimum viable" path is:

1. **First fallback** — drop HRM L-loop (R1 fails). v3 still has Mamba-3 + ternary MoE + MCP tools. Loses MATH headline but every v2 row still passes.
2. **Second fallback** — drop Mamba-3 (R2 fails), use Gated DeltaNet. Loses long-context-decode-throughput edge. Still has HRM + MoE + MCP.
3. **Third fallback** — drop MoE (R4 critical fail), ship as dense 1.8 B ternary student. v3 still beats v2 on math/code via HRM + distillation. Loses the "13× capacity per VRAM" headline.
4. ~~**Fourth fallback** — drop PrismML 1-bit (R8), use TQ1_0 everywhere.~~ **Already taken as the primary path** (June 2026): group-256 TQ1_0/TQ2_0 is now the design, not the fallback. Rung retired.
5. **Worst case** — release as **v3-research-preview**: BF16 student weights only, no GGUF, no MCP. Document the work, publish the ablations, plan v3.1.

Any one of these is acceptable. A combination of (1) + (4) is the *typical* expected outcome — a model that beats v2 on every row but doesn't hit the MATH-500 50 % stretch target.

---

## 8. Risk register summary

| ID | Risk | Stage | P × S | Status |
|---|---|---|---|---|
| R1 | HRM transfer | P4 | M × H | open |
| R2 | Mamba-3 instability | P3 | ML × H | open |
| R3 | Halting collapse | P4 | M × M | open |
| R4 | Ternary MoE cliff | P5 | M × C | open |
| R5 | Teacher caching | P5 | H × H | rental planned |
| R6 | Budget slip | P5 | M × M | open |
| R7 | SFT forgetting | P6 | M × H | open |
| R8 | BB1_0 kernel | P9 | — | **closed** (pivoted to upstream TQ formats, June 2026) |
| R9 | Mamba-3 kernel | P3 | — | **closed** (`fla.layers.mamba3` merged April 2026, verified June 2026) |
| R10 | License | P12 | L × L | closed (no inheritance) |
| R11 | Tool overfit | P8 | M × M | open |
| R12 | Sandbox breakout | P10 | M × C | mitigated by sandbox spec |
| R13 | Constrained decode overhead | P10 | L × L | acceptable |
| R14 | RULER drop | P11 | M × M | open |
| R15 | Bench drift | P11 | L × M | mitigated by re-baselining |
| R16 | Bandwidth | any | M × M | open |
| R17 | Hardware loss | any | L × C | mitigated by HF mirrors |

`P × S` = probability × severity. L=low, M=medium, H=high, C=critical.
