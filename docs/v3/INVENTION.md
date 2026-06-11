# The Invention — what AVA v3 must be, beyond benchmarks

> **Added 10 June 2026.** The owner asked: *"if you were me, what else would you
> demand from AVA v3 for it to be a breakthrough and an invention that genuinely
> helps the world?"* This document is the answer — first the demands (D1–D8, each
> with a falsifiable gate), then the novelty claims written like an invention
> disclosure, so "novel" is a checkable statement instead of marketing.
>
> Companion: [CODE_PIVOT.md](CODE_PIVOT.md) (what we build) · [EDGES.md](EDGES.md) (why it compounds).

---

## Part 1 — The demands I would add

Benchmarks measure capability. These demand **trustworthiness, reach, and
permanence** — the properties that decide whether something helps the world or
just tops a leaderboard.

### D1 — Run with no GPU at all

The 4 GB VRAM target still excludes most of the world's computers. Ternary
weights are the rare format where **CPUs are first-class**: TQ kernels are
integer-add heavy and AVX2-friendly (the entire PrismML 8× speedup story is
about cheap hardware). Demand: **usable on a 4-core CPU laptop with 8 GB RAM —
no GPU, period.**

*Gate:* ≥ 5 tok/s decode at `reasoning_budget=1` on a 4-core AVX2 CPU (llama.cpp
TQ2_0 build). That single number is the difference between "model for people
with gaming laptops" and "model for every student on Earth".

### D2 — Calibrated honesty: it must know when it doesn't know

The single biggest failure of small coding models is **confident hallucination**
— APIs that don't exist, code that looks right. A junior developer who admits
uncertainty is useful; one who lies confidently is dangerous. Demand: every
solution ships with a **self-estimated pass probability**, and that estimate
must be *calibrated*; API/signature claims get checked against the PKM memory
tier before emission.

*Gate:* expected calibration error ≤ 0.15 between predicted and actual test-pass
rates on LiveCodeBench; hallucinated-import rate < 2 % (measured by static
resolution of every import the model emits). The halting head's convergence
signal is the natural seed for this estimator — uncertainty is already a
first-class architectural signal in v3.

### D3 — Absolute privacy: air-gap grade

Code is the most sensitive text people own. Demand: **zero network requirement,
zero telemetry, forever.** The model, memory tier, tools, and evals all run
air-gapped; MCP tools default to local-only with network tools off unless the
user flips them.

*Gate:* full offline smoke suite passes with the network interface disabled.
This is also a *capability* differentiator: every cloud coding assistant
structurally cannot offer it.

### D4 — Teach, don't just solve

The largest "help the world" surface is not professional developers — it is the
hundreds of millions of people *learning* to code with no access to a tutor.
Demand: a first-class **explain mode** — given failing code, produce a
diagnosis a beginner can follow (what broke, why, the minimal fix, the concept
behind it), not just a patch.

*Gate:* a public explain-eval (rubric-scored diagnosis quality on student-bug
corpora) where v3 beats every ≤ 7 B open model. A free Socratic debugger on a
$200 used laptop is arguably more world-changing than any SWE-bench point.

### D5 — Repair, don't rewrite

Real codebases die by a thousand rewrites. Demand: **surgical minimal-diff
edits** that preserve style, comments, and structure — measured, not vibed.

*Gate:* CanItEdit + Aider pass rates in [CODE_PIVOT.md](CODE_PIVOT.md) §7 *plus*
a diff-minimality score (edited-lines / required-lines ≤ 1.5 median) so the
model cannot win by rewriting files wholesale.

### D6 — Publish the energy ledger

If v3's thesis is "intelligence does not require a datacenter", prove it in
units. Demand: **every training phase logs kWh** (GPU wattage × hours, per
platform), and the release publishes the total alongside the benchmarks —
*capability per joule* as a headline metric next to HumanEval+.

*Gate:* release card contains the full ledger; total training energy for
v3-Code lands in the **tens of kWh** (reference: a single 8×A100 node burns
~65 kWh *per day*). Nobody publishes this honestly today; being first matters
— it makes efficiency a competition.

### D7 — The recipe is the product

A free model that only we can build is a curiosity. Demand: **anyone can
re-derive v3 from the donor with one script and free quota** — data hashes,
seeds, configs, checkpoint lineage all public. The invention must be
*independently reconstructible* or it is not an invention, it is a artifact.

*Gate:* a third party (or a clean machine) reproduces phase C2 end-to-end from
README instructions alone, resuming purely from the public HF checkpoints.

### D8 — It maintains itself

Already on the roadmap as S1–S4 ([CODE_PIVOT.md](CODE_PIVOT.md) §5); the demand
makes it operational: a **nightly self-maintenance loop** — run the regression
probe set, diagnose any failure, propose (and at S3, apply behind a human gate)
the fix, and append verified self-play data — every night, on idle hardware.

*Gate:* 30 consecutive nights unattended without a regression escaping the
loop's own detection.

### D9 — Glass-box decoding: transparency as a product surface

Added 2026-06-11 ([RESEARCH_ROUND_2.md](RESEARCH_ROUND_2.md) §5). The model
must show its work *mechanically*, not just rhetorically:

- **Per-token confidence** exposed in the API and surfaced as annotations on
  generated code (calibrated per D2 — the two demands share one machinery).
- **Parametric memory attribution**: product-key memory lookups are discrete
  indices; log which slots fired per token and map slots to training
  provenance. Attribution from *inside the weights*, nearly free because the
  index structure already exists.
- **Structured trace** (analysis → plan → code → self-check) as the only
  output mode in explain contexts (D4 synergy).
- **Per-response compute meter**: HRM steps taken, draft tokens
  accepted/rejected, memory hits — D6's energy ledger, per answer.

*Gate:* attribution logging costs ≤ 5% decode speed, and logged memory indices
are demonstrably non-uniform (informative) on the probe set.

---

## Part 2 — Novelty claims (invention-disclosure style)

"Unique" is checkable: each claim below names a combination we have not found
in published or shipped systems as of 10 June 2026, with the implementation
locus and its supporting evidence base. Prior art is cited for *components*; the
claims are about the **combinations**.

**Claim 1 — Self-improving pocket engineer (the system claim).** An integrated
system comprising (a) a sub-4 GB-resident code model with adaptive latent
recurrence, (b) a sandboxed execution verifier, (c) a persistent RAM-tier
memory, and (d) an idle-time self-play trainer, such that the *deployed
artifact improves measurably on the user's own hardware without external data
or services.* Components: HRM/ACT, Absolute Zero, PKM — all published. The
closed loop on consumer hardware: not shipped anywhere we can find.

**Claim 2 — Convergence-aware halting with stochastic restart.** A halting head
conditioned on the fixed-point residual `‖z_k − z_{k−1}‖`, combined with
perturbed-restart escape under low confidence. (HRM_TEXT.md §1b D2+D3;
implemented in `engine/hrm_core.py`.) Prior art: ACT (blind sigmoid), HRM
(fixed-point recurrence), 2601.10679 (restarts, analysis-only). The fusion as
a trained inference policy: novel.

**Claim 3 — Halting-coupled speculative decoding.** One difficulty scalar
jointly controls speculative draft depth and latent compute per token — easy
tokens drafted deep and pondered shallow, hard tokens the reverse. (EDGES.md
E3.) Speculation and adaptive computation each published; their coupling
through a shared learned signal: novel.

**Claim 4 — VRAM-free knowledge tier on consumer deployment.** Product-key
memory with the value table resident in system RAM/SSD on an edge device, GPU
receiving only top-k rows per token, scaling factual capacity independently of
GPU memory. (EDGES.md E1; `engine/pkm_memory.py`.) PKM is published at
datacenter scale; the consumer placement and its capacity-pack distribution
model: novel.

**Claim 5 — Zero-budget architecture transplant pipeline.** The chained
conversion — LoLCATs linearization → sparse ternary-MoE upcycling → zero-init
organ mounting (recurrence + memory) — taking a permissively-licensed dense
donor to the full v3 architecture for ≈ 1.5 B tokens on free preemptible
compute, with the property that every intermediate state is a working model.
(CODE_PIVOT.md §3.) Each transform published alone; the chain, its
all-states-shippable invariant, and its $0 instantiation: novel.

**Claim 6 — Checkpoint-anywhere training fabric.** Training treated as a swarm
of anonymous preemptible workers (laptop/Colab/Kaggle) synchronized solely
through an atomic latest-pointer protocol on a public model hub, with RNG and
data-position capture making resumption bit-deterministic.
(`scripts/checkpoint_sync.py` — implemented and tested.) Trivial components,
no shipped open training run is structured this way end-to-end.

**Claim 7 — Latent curriculum distillation.** Teacher chain-of-thought
progressively internalized into refinement-loop iterations (token thoughts →
latent steps) during distillation, yielding a knob-controlled latent reasoner.
(HRM_TEXT.md §1b D1 + EDGES.md E2.) Coconut and HRM published separately; the
distillation-time fusion: novel — flagged hypothesis, gated by ablation A9.

**Claim 8 — Parametric memory attribution (candidate, added 2026-06-11).**
Per-token provenance reporting from product-key memory indices in a
consumer-grade model: each generated token carries the identity of the memory
slots that contributed, mapped to their training-data provenance. (D9;
[RESEARCH_ROUND_2.md](RESEARCH_ROUND_2.md) §5.) Interpretability work probes
memories post-hoc; no shipped open model exposes parametric-memory attribution
as an inference-time API. Struck if logged indices prove uninformative
(near-uniform) or cost > 5% decode speed.

Claims 2–8 are each independently publishable if validated; Claim 1 is the
invention. If ablations falsify a claim, it is struck here — this document is
a ledger, not a brochure.

---

## What I would *not* demand

For honesty's sake, demands considered and rejected:

- **"Beat GLM-4.7 / frontier giants."** Physically incoherent at 4 GB. The
  invention is the capability-per-resource frontier, not the absolute frontier.
- **General-domain excellence.** Specialization *is* the efficiency thesis.
  Sanity floors only.
- **Natural-language breadth in explanations (v3.0).** Explain mode ships
  English-first; multilingual explanation is v3.1 — one thing done excellently
  before ten things done adequately.
- **Weight-level self-modification.** S3/S4 operate on the *pipeline and data*,
  behind human gates and the sandbox. A self-improving system must be
  verifiable at every rung — that constraint is load-bearing for both safety
  and science.
