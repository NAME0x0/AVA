# AVA v3 — The Coding Pivot

> **Decision date: 10 June 2026. This document supersedes the general-assistant scope
> everywhere it conflicts.** AVA v3 is now a **coding specialist**: one domain, all
> languages, under 4 GB VRAM, trained for **$0** on free compute, free forever.
>
> Mission: prove that a world-class coding model does not require a datacenter.
> Every doubling of frontier training compute is a bet that intelligence must be
> bought with energy. v3 is the counter-example bet: **inherit, transplant, verify,
> self-improve** — at the cost of one laptop, free Colab/Kaggle quota, and patience.

---

## 1. What changes, what survives

| Aspect | Before (general v3) | Now (v3-Code) |
|---|---|---|
| Domain | strict dominance over v2 on 19 general benchmarks | **coding only** — general benchmarks demoted to sanity floors |
| Training budget | ~14 B tokens, $2–3 K A100 rental for P5 | **~1.5 B tokens, $0** — donor transplant replaces pretraining |
| Starting point | clean-slate init, sparse-upcycle from 35B teacher | **warm-start from a donor model** (inherit its pretraining) |
| Teacher | Qwen 3.6 35B-A3B logit KL (rented) | **Qwen 3.6-27B trace distillation** (Apache 2.0, SWE-bench Verified 77.2) — text traces, generated on free compute |
| Reasoning (HRM refinement loop) | survives | survives — debugging/repair is the ideal refinement workload |
| Mamba-3 hybrid | survives | survives via **LoLCATs-style conversion** of donor attention (cheap) |
| Ternary MoE | survives | survives via **sparse upcycling** of donor FFN + BitDistiller QAT |
| PKM memory tier (E1) | factual world knowledge | **API/library/idiom memory** — stdlib signatures, framework patterns |
| MCP tools + sandbox | one pillar among many | **the product**: model + execution harness ship as one unit |
| Self-play flywheel (E7) | v3.1 lane | **promoted to core training phase** — code has a free verifier |

The architecture thesis is unchanged. What changed is *how it comes into existence*:
not trained from near-scratch, but **transplanted onto a donor that already paid the
pretraining bill**.

---

## 2. Donor and teacher (both free, both Apache 2.0)

### Donor: Qwen3-4B

| Criterion | Why Qwen3-4B |
|---|---|
| License | Apache 2.0 (note: Qwen2.5-Coder-3B is research-license — **disqualified** for a free-for-the-world release) |
| Quality | strongest ≤4 B open generalist with solid code ability; same family as the teacher (idiom match) |
| Shape | hidden 2560, 36 layers, GQA 32/8, head_dim 128, FFN inter 9728, vocab 151 936 (tied embeddings) |
| Tokenizer | Qwen3 151 K — replaces the 248 K assumption from the clean-slate design; saves ~0.17 B embedding params |

The student inherits the donor skeleton. The 24×1792 clean-slate shape in
[ARCHITECTURE_V3.md](ARCHITECTURE_V3.md) remains the reference for the *block
design*; dimensions now come from the donor. Re-evaluate donor choice at P1-Code
(if a stronger Apache ≤4 B coder exists by then, swap — the transplant recipe is
donor-agnostic).

### Teacher: Qwen3.6-27B — now a teacher *panel* (round 2)

SWE-bench Verified 77.2 — flagship-level coding, Apache 2.0, 262 K context. Runs
int4 (~14.5 GB) on Kaggle's free 2×T4 (32 GB combined) for trace generation, and on
the laptop via Exp 5's streaming int4 loader (slow, but overnight is free). We
take **text traces** (problem → reasoning → code → test results), not logits:
no vocab mismatch with the donor, no terabyte logit cache, no rented cluster.

**Revision 2026-06-11 ([RESEARCH_ROUND_2.md](RESEARCH_ROUND_2.md) §3):** v3 is
taught by a **multi-teacher panel**, not one model — Qwen3.6-27B (agentic/SWE),
**Devstral Small 2 24B-2512** (Apache 2.0, SWE-bench V 68.0, tool-driven
multi-file editing; different lineage = different failure modes), the donor
itself as style anchor, and NVIDIA's open Nemotron agentic-trace datasets
(pre-generated, zero serving cost). Conflict protocol is binding: execution
filter → domain routing → local-naturalness trace selection → one format
contract; **text traces only, never logit fusion**. Evidence: TinyLLM
(+5–15.7 pp, students beating teachers at 1–26% size), peer-review KD
(+5.48 pp), Local Naturalness (+9.4 pp over global selection).

---

## 3. The transplant: four published recipes, stacked

Each step starts from a working model and ends in a working model — the failure
ladder never strands us.

**T1 — Linearize: donor attention → Mamba-3 hybrid (LoLCATs recipe).**
LoLCATs (arXiv:[2410.10254](https://arxiv.org/abs/2410.10254)) swaps softmax
attention for linear attention by (a) training the linear layer to match the
attention output (MSE "attention transfer"), then (b) LoRA fine-tune to recover
quality — with **~0.2 % of typical pretraining tokens (≈ 40 M)**. We apply it to 27
of 36 donor layers (keep 9 softmax = 3:1 hybrid), targeting the Mamba-3 mixer; the
attention-transfer loss treats the donor's own attention as the teacher.
Alternatives if quality drops: MOHAWK / Mamba-in-Llama / RADLADS (full-update
variants, costlier but more exact).

**T2 — MoE-ify: donor FFN → 32 ternary experts (sparse upcycling + BitDistiller).**
Sparse upcycling (arXiv:[2212.05055](https://arxiv.org/abs/2212.05055)): replicate
the donor FFN into experts (split + noise), add the BF16 sigmoid router with
bias-based balancing, then run ternary QAT (group-256, `engine/ternary_linear.py`)
with trace distillation as the recovery signal. The shared expert keeps a full
donor-FFN copy so the worst case is "donor quality, smaller storage".

**T3 — Mount the v3 organs (all zero-init = no-op at step 0).**
HRM refinement loop (`cond_proj` zero-init, halting starts at always-halt-k=1),
PKM memory tier (zero-init out_proj), convergence-aware halting — exactly the
mounting design already implemented and tested in `engine/`. The donor's behavior
is bit-identical at mount time; capability grows from there.

**T4 — Specialize: the coding curriculum (§4) + self-play (§5).**

Honest cost of the transplant chain: T1 ≈ 40–80 M tokens, T2 ≈ 300–600 M, T3+T4 ≈
600 M–1 B. Total **≈ 1–1.7 B tokens** — an order of magnitude under the clean-slate
plan, which is what makes $0 arithmetic close (§6).

---

## 4. Coding curriculum ($0 data stack, all redistributable)

| Source | Size / license | Role |
|---|---|---|
| `nvidia/OpenCodeReasoning` + `-2` (arXiv:[2504.01943](https://arxiv.org/abs/2504.01943)) | 735 K + 2.5 M samples, Apache 2.0 | reasoning-trace SFT backbone (Python + C++) |
| Teacher traces (Qwen3.6-27B, self-generated on free compute) | targeted; grows weekly | repair, refactoring, repo-level tasks, weak languages |
| The Stack v2 (filtered subsets) | permissive-only filter | FIM + continued pretraining mix; **all-language coverage** |
| CommitPackFT | 2 GB, permissive | commit-style edit/diff training (the "edit" skill agents need) |
| self-oss-instruct / StarCoder2 pipeline | Apache | instruction diversity |
| Our own execution logs (sandbox) | generated | verified self-play data (§5) |

Curriculum order: FIM + multi-language continued pretrain (during T1/T2 recovery) →
reasoning SFT (OpenCodeReasoning; note their finding that *instruction diversity
beats aggressive execution-filtering* for SFT) → edit/diff SFT (CommitPackFT) →
agentic traces (tool calls inside MCP harness) → execution-feedback RL (§5).

Multi-language is a first-class requirement: MultiPL-E coverage drives the The
Stack v2 sampling weights, and the MoE router gives a measurable hypothesis —
**experts specialize by language family** (we log router entropy per language;
ablation A12).

FIM (fill-in-middle) objective joins next-token from T2 onward — non-negotiable
for editor/agent use.

---

## 5. Self-improvement: the part that makes it a breakthrough

Code is the one domain with a **free, perfect, infinitely patient verifier**: the
computer itself. v3-Code's training closes the loop:

1. **Propose** — the model generates coding tasks + tests (Absolute Zero recipe,
   arXiv:[2505.03335](https://arxiv.org/abs/2505.03335): proposer rewarded for
   tasks of learnable difficulty).
2. **Solve** — at `reasoning_budget=6` + latent restarts (expensive mode).
3. **Verify** — run in the MCP sandbox; tests pass or fail. Stepwise execution
   rewards where applicable (ExecVerify, arXiv:[2603.11226](https://arxiv.org/abs/2603.11226)).
4. **Distill** — verified solutions become SFT data consumed at
   `reasoning_budget=2` (cheap mode). Test-time compute converted into weights.
5. **Repeat** — overnight, on the laptop, forever. Capability grows with idle
   hours, not datacenter dollars.

"Improve itself" gets a concrete ladder, each rung verifiable:

| Rung | Self-improvement behavior | Verified by |
|---|---|---|
| S1 | fixes failing code it wrote, given test output | sandbox (P8-Code gate) |
| S2 | generates its own training tasks + tests (AZR loop) | sandbox + learnability reward |
| S3 | writes/repairs scripts in *this repo's training pipeline* | repo test suite |
| S4 | proposes data-filter and curriculum changes that measurably improve the next cycle | A/B eval on held-out probe set |

S1–S2 are published mechanisms. S3–S4 are hypotheses with falsifiable gates —
flagged as such, and they are the research headline if they hold: **a sub-4 GB
model that participates in its own development loop.**

---

## 6. Zero-budget compute plan

Inventory (all free):

| Resource | Quota | Role |
|---|---|---|
| Laptop RTX A2000 4 GB + 32 GB RAM | 24/7 | self-play loop, teacher trace gen (overnight), small ablations, evals |
| Google Colab free (T4 16 GB) | session-capped, variable | T1/T2 training shards (3–4 h resumable chunks) |
| Kaggle (2×T4, 30 h/week) | 30 GPU-h/wk | the workhorse: weekly long shards + teacher trace generation |
| Hugging Face Hub | free storage | **every 30 min: checkpoint + optimizer state push** — sessions are preemptible by design. Implemented: [`scripts/checkpoint_sync.py`](../../experiments/exp6_v3/scripts/checkpoint_sync.py) — atomic LATEST-pointer protocol, RNG + data-position capture, resume-from-anywhere (Claim 6 in [INVENTION.md](INVENTION.md)). Token via `HF_TOKEN` env only — never in code or configs |
| HF Spaces / local | free | eval dashboards |

Throughput honesty: QLoRA-style training of the ~4 B student on a T4 runs
~1–2 K tok/s. The ~1.5 B-token plan ≈ **250–400 T4-hours ≈ 8–12 weeks of free
quota** (Kaggle 30 h + Colab ~10–20 h + laptop continuous). Every phase is built
as **resumable 3-hour shards**: deterministic data order, seed + step in the
checkpoint name, `push_to_hub` on a timer, auto-resume script. Preemption costs
≤ 30 min of work by construction.

> **Realism revision (2026-06-11, supersedes the paragraph above where they
> conflict — [RESEARCH_ROUND_2.md](RESEARCH_ROUND_2.md) §6):** HRM deep
> supervision multiplies training FLOPs ≈ 4.5× on supervised iterates, and the
> teacher panel consumes ~25% of weekly Kaggle quota for trace generation.
> Honest budget: **400–700 T4-hours ≈ 3–6 calendar months**, with
> single-maintainer bandwidth — not GPU quota — as the binding constraint
> (budget 1–2 redo cycles per transplant phase). Primary eval gate revised to
> *donor + δ on the full matrix; beat all ≤4B opens on ≥ 70% of the matrix*,
> with all/all kept as a stretch goal. Each phase still ends with a published
> HF checkpoint — any phase can ship.

What the $0 constraint kills, explicitly:

- ✂ 8 B-token logit-KL distillation (was P5) → replaced by trace distillation + QAT recovery.
- ✂ 32 K-native training context → **8 K native, YaRN to 32–64 K**, long-context
  curriculum moves to "when compute allows" (repo-level work leans on the agentic
  harness + retrieval over raw context length).
- ✂ 1 M-context ambitions → out of v3-Code scope.
- ✂ General-domain strict dominance → replaced by §7 gates.

---

## 7. What "beat every open model at coding" means (falsifiable)

Weight class is defined by **deployment reality**: ≤ 4 GB VRAM resident at 8 K
context. A 30 B-A3B MoE (e.g. Nemotron-Cascade 2) does not fit and is *above*
class — we report against it anyway, for honesty.

| Tier | Models | Commitment |
|---|---|---|
| Our class (≤ 4 GB resident) | Qwen3-4B (donor!), Yi-Coder-1.5B, Phi-4-mini-class, every ≤4 B open coder | **Beat on every coding eval below. Non-negotiable — includes beating our own donor decisively** |
| One class up (7–9 B) | Qwen2.5-Coder-7B, Yi-Coder-9B, OpenCoder-8B | Beat on ≥ 70 % of the eval matrix |
| Two classes up (14 B+) | Qwen2.5-Coder-14B-class | Beat on agentic evals (SWE-bench Lite/Verified with our MCP harness); competitive elsewhere |

Eval matrix (all public, all reproducible on the laptop): HumanEval+ / MBPP+
(EvalPlus), **LiveCodeBench** (contamination-resistant, rolling), BigCodeBench,
**MultiPL-E** (the all-languages gate: ≥ 12 languages, no language below donor),
CRUXEval (execution reasoning), CanItEdit + Aider polyglot (editing), SWE-bench
Lite + Verified (agentic, with harness). Sanity floors: ARC-E ≥ 75 %, MMLU ≥ 45 %
(catastrophic-forgetting alarms, not goals).

Numeric targets are set **after** the donor baseline pass at P1-Code (measure
donor on the full matrix first; targets = donor + published-class deltas). What is
committed now: the tier table above plus — HumanEval+ ≥ 85 % (v2: 19.5 %),
LiveCodeBench above every ≤4 B open model on the same rolling window, MultiPL-E
average above Qwen2.5-Coder-7B.

---

## 8. Revised phases (v3-Code)

```
C0  Pivot docs + donor baseline eval harness            (this commit)
C1  Donor + teacher acquisition; full eval matrix on donor = the bar
C2  T1 LoLCATs linearization -> 3:1 Mamba-3 hybrid       (~40-80M tok, Kaggle)
C3  T2 sparse upcycle -> ternary MoE + QAT recovery      (~300-600M tok)
C4  T3 mount HRM loop + PKM memory; halting + latent curriculum on code traces
C5  Coding SFT: OpenCodeReasoning + FIM + CommitPackFT + teacher traces
C6  Execution-feedback RL + self-play (S1-S2); agentic harness training
C7  Pack (TQ1_0/TQ2_0 + Q8) -> GGUF; EAGLE-3 draft head; halting-coupled speculation
C8  Full eval matrix vs all tiers; release; S3-S4 self-improvement experiments
```

Gates per phase: each Tn must not regress the donor's coding baseline by > 2 pp
(measured on a fixed 500-problem probe set) before the next phase starts. The
transplant ladder means **any phase can ship**: worst case is "donor + better
storage + tools", best case is the breakthrough.

---

## 9. Why this can matter beyond benchmarks

The frontier narrative says capability scales with compute, and compute scales
with energy. v3-Code tests the counter-claim at the extreme: donor inheritance
(pretraining reused, not repeated) + architecture transplants (published, cheap)
+ a free verifier (execution) + idle-hardware self-play. If a 4 GB laptop model
trained for $0 genuinely outcodes everything in its class and harasses models 3×
its size — that is evidence the field's cost curve is a choice, not a law. Total
training energy for v3-Code is on the order of **tens of kWh** — less than a
single A100-node *day* — and it is reproducible by any student with a laptop and
a free Kaggle account. That reproducibility *is* the global impact.

## References (pivot-specific)

- LoLCATs, arXiv:[2410.10254](https://arxiv.org/abs/2410.10254) — low-cost attention→linear conversion.
- MOHAWK / Mamba-in-Llama, arXiv:[2408.10189](https://arxiv.org/abs/2408.10189) / arXiv:[2408.15237](https://arxiv.org/abs/2408.15237) — full-update conversion fallbacks.
- Sparse upcycling, arXiv:[2212.05055](https://arxiv.org/abs/2212.05055) — dense FFN → MoE init.
- OpenCodeReasoning 1/2, arXiv:[2504.01943](https://arxiv.org/abs/2504.01943) — Apache reasoning-trace data.
- Absolute Zero, arXiv:[2505.03335](https://arxiv.org/abs/2505.03335) — self-play with execution verifier.
- ExecVerify, arXiv:[2603.11226](https://arxiv.org/abs/2603.11226) — stepwise verifiable execution rewards.
- Qwen3.6-27B model card ([HF](https://huggingface.co/Qwen/Qwen3.6-27B)) — teacher; SWE-bench Verified 77.2, Apache 2.0.
- Qwen3 report, arXiv:[2505.09388](https://arxiv.org/abs/2505.09388) — donor family.
- BitDistiller, sparse QAT, EAGLE-3, PKM memory: see [REFERENCES.md](REFERENCES.md) + [EDGES.md](EDGES.md).
