# The Edge Portfolio — v3's compounding moats

> **Added 9 June 2026.** This document answers one question: *what does v3 do that no
> shipped model does, where the advantage grows as we scale a resource?* Every edge
> below has (a) a published mechanism with numbers, (b) a reason it is still
> unexploited in deployed open models, (c) a scaling axis along which its benefit
> *increases*, and (d) a falsifiable gate. Hypotheses are marked as such.
>
> Companion docs: [ARCHITECTURE_V3.md](ARCHITECTURE_V3.md) · [HRM_TEXT.md](HRM_TEXT.md) · [RECIPE.md](RECIPE.md) · [RISKS.md](RISKS.md)

## The thesis: multiply orthogonal axes, not one

A dense transformer has one scaling axis — parameters — and it is the axis our 4 GB
ceiling caps hardest. v3's design philosophy is the inverse: **put every capability on
the cheapest resource that can carry it, and make each one independently elastic.**

| Capability | Where v3 puts it | Resource it scales with | VRAM cost of scaling |
|---|---|---|---|
| Procedural skill | Ternary MoE weights | VRAM (fixed budget) | — (the only fixed axis) |
| Factual knowledge | **Product-key memory tier (E1)** | System RAM / SSD | **~0** |
| Long-context state | Mamba-3 + NSA sparse attention (E4) | Context length | sub-linear |
| Reasoning depth | Refinement loop + latent superposition (E2) | Test-time compute | 0 (weight reuse) |
| Wall-clock → accuracy | Self-drafted speculation (E3) | Decode speed | ~2 % params |
| Training quality | Precision scaling law + μP (E5, E6) | Training compute | 0 |
| Post-release growth | Self-play flywheel (E7) | Idle laptop hours | 0 |

These axes are *orthogonal resources*: disk, RAM, context window, inference seconds,
training tokens, idle time. Scaling any one improves v3 while VRAM stays at 4 GB.
Scaling several multiplies. That is the honest version of "exponential quality from
scaling": not one curve bending upward, but a **product of six independent curves**
where every shipped competitor is riding only one or two.

---

## E1 — The RAM-tier memory: factual capacity without VRAM

**Mechanism.** Product-key memory (PKM) layers: a vast learned key-value table where
each token retrieves only the top-k (k≈32) of N values via a product-key factorization
— two top-k searches over √N keys each instead of one over N. Compute per token is
negligible; the layer is pure capacity.

**Published proof.** Meta, *Memory Layers at Scale* (arXiv:[2412.09764](https://arxiv.org/abs/2412.09764)):
models with memory layers beat dense models given **2× more compute**, and beat
**MoE models matched for both compute and parameters**, with gains *most pronounced on
factual tasks* — exactly v2's weak surface (HellaSwag, factual recall). Gains grew
monotonically with memory size in their sweeps (tested to 128 B memory parameters).

**The unexploited angle (our zero-day).** Memory-layer lookups are
*bandwidth-trivial*: per token, the layer touches 2√N key rows + k value rows — a few
hundred KB. It is the one parameter class that **does not need to live in VRAM at
all**. No shipped open model exploits this. v3 places the entire value table in
**system RAM (32 GB on the reference laptop), with an SSD-backed growth path**:

- v3 base: 1 M slots × 1792 dims ≈ **1.8 B extra parameters at ~1.8 GB RAM (Q8), 0 VRAM**.
- Growth path: 16 M slots ≈ 28 B memory parameters on SSD — a "factual capacity
  pack" users download separately, like a game texture pack. Same weights, bigger brain.

**Scaling behavior.** Capacity/VRAM for v3 becomes unbounded by GPU: the
quality-per-VRAM curve *steepens* as memory grows, while every dense competitor's
curve is flat at their parameter count. Combined with the MoE experts, v3's
"effective parameter" story is: 3.24 B core + 1.8 B memory at launch, growable to
~30 B-class factual capacity with zero VRAM change.

**Status & gate.** Implemented at P2 (`engine/pkm_memory.py`, shared pool, H-block
branch). Ablation A8: +memory vs −memory at P3 scale on a factual probe set; ship
only if ≥ +2 pp on factual cluster (NaturalQuestions-style probes + HellaSwag) at
equal step count. Risk R18 covers the llama.cpp export op (fallback: PyTorch serving
path for the memory build, GGUF build ships without memory).

**Round-2 upgrade (2026-06-11, [RESEARCH_ROUND_2.md](RESEARCH_ROUND_2.md) §1.2).**
TurboQuant-style 2-bit data-oblivious quantization (RyanCodrai/turbovec, MIT —
random rotation → Lloyd-Max buckets, SIMD nibble-LUT kernels) applied to the
value table: ~3.8 GB fp16 → **~0.5 GB + scales** at recall loss measured under
1 pp in vector-retrieval benchmarks. Directly attacks R19 (RAM-bandwidth
latency): ~7× fewer bytes per EmbeddingBag gather. Same library doubles as a
zero-VRAM repo-retrieval index for the agentic harness (air-gap compatible).
Discrete lookup indices also feed D9 memory attribution (INVENTION.md Claim 8).

---

## E2 — Latent superposition reasoning: the refinement loop becomes a search

**Mechanism.** Coconut (*Training Large Language Models to Reason in a Continuous
Latent Space*, arXiv:[2412.06769](https://arxiv.org/abs/2412.06769)) feeds the last
hidden state back as the next input embedding — reasoning in latent space, no tokens.
Theory follow-up (arXiv:[2509.23365](https://arxiv.org/abs/2509.23365)) proves the
training dynamics produce **superposition**: a continuous thought encodes *multiple
candidate reasoning paths simultaneously*, so latent reasoning performs implicit
breadth-first search where token CoT must commit to one path. June 2026 follow-up
(arXiv:[2606.07720](https://arxiv.org/abs/2606.07720)) adds persistent memory slots to
relieve the "concept bottleneck".

**The unexploited combination.** Nobody has published Coconut-style latent reasoning
**inside an HRM refinement loop, trained by distillation**. The pieces interlock:

1. Our refinement loop (HRM_TEXT.md §1b) already iterates a latent state — it *is*
   Coconut's recurrence, but with dedicated weights-reuse blocks and halting.
2. Coconut's curriculum (progressively replace teacher CoT tokens with latent steps)
   gives the L-loop **explicit reasoning content to internalize** — solving the
   biggest open question of P4 (what supervises the loop beyond next-token loss).
3. The superposition theory explains *why* perturbed restarts (D3) work: a restart
   re-samples which superposed path collapses out. Restart + superposition =
   stochastic latent BFS — with the teacher's CoT as the search curriculum.

**Curriculum spec (P4b).** Teacher emits k-step CoT for each training problem. Stage
schedule: replace the first j CoT sentences with j L-iterations (j = 0 → k over the
curriculum), supervising the final answer + remaining CoT. End state: full latent
reasoning, zero emitted thought tokens, `reasoning_budget` slider controls search
depth.

**Scaling behavior.** Test-time scaling laws (Snell et al.,
arXiv:[2408.03314](https://arxiv.org/abs/2408.03314)) show accuracy grows
log-linearly with test-time compute *when the model can use it*. Latent iterations
are ~50× cheaper than CoT tokens (no vocab projection, no KV append per thought, no
detokenize/retokenize), so v3 buys more search per joule than any token-CoT model.
Benefit grows with problem difficulty — the curve v2 was worst on.

**Status & gate.** Hypothesis (combination unpublished — flagged as such). Ablation
A9 at P4: latent-curriculum vs plain halting curriculum on GSM8K/MATH-500 probes;
adopt if ≥ +3 pp MATH-500 at equal token budget. Falsified if latent steps fail to
absorb CoT content (MATH-500 delta < 1 pp).

---

## E3 — Speed is accuracy: halting-coupled self-speculation

**Mechanism (revised 2026-07-09).** The donor **ships trained MTP heads**: the
whole Qwen3.5 family releases with built-in multi-token prediction, supported in
llama.cpp via `--spec-type draft-mtp` (PR #22673; bleeding-edge builds), with
**1.7× measured decode speedup** on Qwen3.6-27B and unsloth MTP GGUFs already
published for Qwen3.5-4B. E3 therefore no longer requires training a draft head —
the EAGLE-3 lane (arXiv:[2503.01840](https://arxiv.org/abs/2503.01840), llama.cpp
PR [#18039](https://github.com/ggml-org/llama.cpp/pull/18039)) is demoted to
fallback if surgery destroys the MTP heads. **New invariant: every phase gate
checks MTP-head preservation** — losing them costs a free 1.7×.

**The unexploited coupling (our second zero-day).** v3 has a signal no other model
has: the **convergence-aware halting head already computes per-token difficulty**
(it reads the fixed-point residual). Speculation and pondering are the same decision
viewed from opposite ends — "how predictable is what comes next":

- halting prob high (easy region) → draft deep (`--spec-draft-n-max` up), L-steps = 1
- halting prob low (hard region) → draft shallow, spend the saved budget on L-steps

One scalar drives both knobs. No published system couples adaptive computation with
speculative depth. With donor MTP heads the coupling target is the MTP draft length,
not a trained EAGLE head — cost drops to a lookup table on the halting logit driving
an existing llama.cpp flag. Expected effect: the *joint* throughput-accuracy frontier
dominates either knob alone.

**Why it compounds.** Wall-clock saved by speculation converts directly into accuracy
through E2: at fixed latency, 2–3× decode speed funds 2–3× more latent restarts or
self-consistency samples — and test-time scaling laws price that at +several pp on
math. Speed stops being a comfort feature and becomes an accuracy multiplier.

**Status & gate.** P10 feature (post-training; draft head trained on P6 SFT data —
SpecForge, arXiv:[2603.18567](https://arxiv.org/abs/2603.18567), is the training
framework). Ablation A11: coupled vs independent knobs on the latency-accuracy
frontier. Low risk — worst case is stock EAGLE-3 speedup with no coupling gain.

---

## E4 — Long context that gets cheaper, not just longer

**Mechanism.** NSA — *Native Sparse Attention*
(arXiv:[2502.11089](https://arxiv.org/abs/2502.11089), ACL 2025): trainable,
hardware-aligned sparsity (compressed + selected + sliding branches) that **matches or
exceeds full attention** on general and long-context benchmarks while delivering
large speedups at 64 K+ across prefill, decode, and backward. FSA
(OpenReview [c5mdo1hWrs](https://openreview.net/forum?id=c5mdo1hWrs)) provides
kernels for consumer GPUs.

**v3 placement.** The 6 H-blocks are our only quadratic component and dominate cost
beyond 32 K (SUBQUADRATIC.md §3). Converting them to NSA at the long-context
fine-tune stage (P-late 64 K → 256 K curriculum) removes the last quadratic term:
**every layer of v3 becomes sub-quadratic** with published evidence that quality does
not degrade — the "exceeds full attention" result makes this a free upgrade, not a
trade.

**Scaling behavior.** v3's cost-per-token *flattens* with context while quality
holds; every full-attention competitor's cost curve is quadratic. The gap between v3
and the field therefore **widens automatically as contexts get longer** — and the
industry trend is monotonically longer contexts.

**Status & gate.** Config-gated option (`h_block.attention: full | nsa`), exercised
only at the long-context phase; native-32K training stays full-attention (RoPE/recall
baseline first). Gate: RULER 128 K ≥ 90 % with NSA on, else ship full-attention
H-blocks and YaRN-128K only.

---

## E5 — The precision tailwind: ternary improves with scale

**Mechanism.** *Scaling Laws for Precision*
(arXiv:[2411.04330](https://arxiv.org/abs/2411.04330), 465-model sweep): low-precision
training reduces "effective parameter count" by a *fixed factor* — meaning the loss
penalty of low-bit weights **shrinks relative to model quality as parameter count
grows**, and "training larger models in lower precision may be compute optimal."
Corroborated by the QAT scaling law study (arXiv:[2505.14302](https://arxiv.org/abs/2505.14302))
and the 100 T-token quantization study (arXiv:[2411.17691](https://arxiv.org/abs/2411.17691))
— which also warns the *opposite* holds for post-training quantization on over-trained
models (PTQ degradation grows with training tokens; QAT avoids this).

**Why this is an edge and not just a fact.** It selects a *direction* every other
small-model builder is walking away from: the field ships PTQ'd dense models — the
configuration the scaling laws say ages worst. v3's QAT-ternary-MoE sits in the
configuration the laws say ages best: more total parameters (MoE + memory tier),
fewer bits, quantization baked into training. Every future v3.x scale-up (more
experts, bigger memory) gets a *better* ternary trade than v3.0 did — the
architecture is positioned on the compute-optimal frontier as it grows.

**Status.** Not a feature — a thesis governing all sizing decisions. Recorded here so
future scale-ups cite it instead of re-deriving.

---

## E6 — μP: tune at 50 M, deploy at 3.24 B

**Mechanism.** Maximal-update parametrization (Tensor Programs V,
arXiv:[2203.03466](https://arxiv.org/abs/2203.03466)): under μP, optimal learning
rate, init scale, and related hyperparameters become **width-invariant** — tune on a
tiny proxy, transfer zero-shot to the full model. Cerebras-GPT
(arXiv:[2304.03208](https://arxiv.org/abs/2304.03208)) validated end-to-end at GPT
scale.

**Why it matters disproportionately for us.** The single biggest risk class in
RISKS.md is training instability in novel components (R2 Mamba-3 complex state, R3
halting collapse, ternary QAT LR sensitivity). A 50 M-parameter μP proxy of the full
v3 topology (the `V3Config.tiny()` lineage) lets us sweep LR / ponder weight / memory
LR multiplier overnight on the laptop and transfer the optimum. One A100 rental, no
hyperparameter roulette. For a single-engineer project this converts the scarcest
resource — training attempts — into a solved problem.

**Status & gate.** P3 engineering practice: μP-parametrize the stack, verify LR
transfer across 50 M → 200 M → 800 M proxies (loss curves must superpose); transfer
to 3.24 B. Gate: best proxy LR within 2× of best full-size LR on a 1-day probe run.

---

## E7 — The flywheel: capability that grows after release

**Mechanism.** Absolute Zero (arXiv:[2505.03335](https://arxiv.org/abs/2505.03335),
NeurIPS 2025 spotlight): a model *proposes its own tasks*, solves them, and learns
from verifiable rewards via a code executor — **zero external data**, SOTA among
zero-data reasoners, works across model scales and classes.

**The v3 fit nobody else has.** Absolute Zero needs a sandboxed executor and a
training loop. v3 *ships* both: the MCP sandbox (`experiments/exp6_v3/mcp/sandbox/`)
is a hardened code executor, and the QAT/LoRA training scripts run on the laptop.
Overnight idle compute becomes a self-play curriculum: propose → solve with
`reasoning_budget=6` → verify in sandbox → distill verified traces back at budget 2.
This is also the **honest mechanism behind "self-distillation of test-time compute"**:
expensive-mode solutions become cheap-mode training data (the same trick frontier
labs run at datacenter scale), on one laptop.

**Scaling behavior.** Quality scales with *cumulative idle hours* — a resource
every deployed laptop has in abundance and no static checkpoint can use.

**Status & gate.** **Promoted to core training phase by the coding pivot (10 June
2026)** — code's execution verifier makes this the primary $0 data engine; see
[CODE_PIVOT.md](CODE_PIVOT.md) §5 (phase C6, self-improvement ladder S1–S4). Gate:
verified pass-rate gain on a held-out probe set after 100 self-play hours, no
regression elsewhere.

---

## E8 — Code-native tokenizer transfer (gated, round 2)

**Mechanism.** ZeTT-style zero-shot tokenizer transfer (hypernetwork embedding
init) from the donor's 151,936-token general vocabulary to a ~80 K
code-weighted vocabulary, followed by a short recovery run.

**Published proof.** ZeTT outperforms FVT/FOCUS for code-generation tokenizer
transfer; TokenAdapt halves perplexity ratios vs ReTok/TransTokenizer; tokenizer
quality can dominate scale (a 350 M model with the right tokenizer beating a
2.7 B one, ACL). Byte-level/tokenizer-free remains unproven at this scale for
code — rejected.

**Why it compounds.** The donor's tied embedding is 389 M params (~12% of core
budget) sized for general multilingual text. A code-weighted vocabulary returns
~180 M params (real Q8 GGUF megabytes), shrinks the per-token softmax, and
compresses code into fewer tokens — fewer tokens per line multiplies with
every speed edge (D1 CPU decode, E3 speculation).

**Status & gate.** Off the critical path; run only if T1–T4 land early.
Ablation A15: transfer + recovery must hold the C1 500-problem probe within
1 pp of pre-transfer — else revert. (A12 belongs to router-entropy-per-language
in CODE_PIVOT §4.) See [RESEARCH_ROUND_2.md](RESEARCH_ROUND_2.md) §4.

---

## The 1000× ledger (kept honest)

"1000× better than v2" is meaningless on a single benchmark — MMLU does not go to
59,200 %. It is meaningful as a **product of orthogonal capability ratios**, each
individually defensible:

| Axis | v2 | v3 target | Ratio |
|---|---|---|---:|
| Usable context (trained) | 384 tok | 1 M (YaRN, gated) | ~2 600× |
| Parameters per GB VRAM (incl. RAM tier at launch) | ~1.4 B / 1.4 GB | 5.0 B / ~1.9 GB VRAM | ~2.6× |
| Factual capacity growth path (E1, no VRAM change) | none | → ~30 B-class | open-ended |
| Reasoning compute elasticity per token | 1× fixed | 1–6 L-steps × restarts × budget | ~10× |
| Tool-call usefulness (agentic GSM8K call-correctness) | 0.6 % | ≥ 65 % | ~100× |
| MATH-500 | 18.8 % | ≥ 50 % | 2.7× |
| Decode tok/s at fixed quality (TQ2_0 + EAGLE-3) | ~45 | ~100+ | ~2.3× |
| Post-release improvement rate (E7) | 0 | > 0 | ∞ (strictly) |

Product of the bounded rows alone exceeds 10⁶ — quote it as "1000×" with a straight
face *only* in the composite sense above, never about a single score. The strict
per-benchmark commitments stay where they always were: [PERF_TARGETS.md](PERF_TARGETS.md),
every row ≥ v2.

## E9 — Loop stability + depth-index (OpenMythos-derived, implemented)

**Mechanism.** Two opt-in refinements to the HRM refinement loop, both
implemented in `engine/hrm_core.py`, both default-off and no-op at init
(donor warm-start preserved): a **loop-index embedding** (sinusoidal
recurrence-depth signal, RoPE analog over depth) so shared loop weights know
which iteration they are on; and an **LTI-stable injection** that decays the
inter-iteration carry with a diagonal `A = exp(-exp(·)) ∈ [0,1)`, giving
provable spectral radius < 1 — the structural fix for the fixed-point blow-up
that D3's perturbed restart currently band-aids.

**Why it compounds.** Near-zero cost (one gate scalar + one diagonal SSM per
unit), and it makes an *existing* edge cheaper: if A17 cuts D3 restart
frequency, adaptive inference spends fewer wasted forward passes — speed that
multiplies with E3 speculation and D1 CPU decode.

**Status & gate.** Implemented + tested (5 tests). Off until measured.
A16 (loop-index): enable if it improves the loss-vs-L-step curve at equal
steps (bar = non-negative, cost ≈ 0). A17 (LTI): enable if it reduces
grad-norm spikes / D3 restart rate without hurting refinement gain. Full
rationale: [RESEARCH_ROUND_4.md](RESEARCH_ROUND_4.md).

## E10 — Fuzzy-glue co-processor + skill packs (PAW pattern, gated)

**Mechanism.** Program-as-Weights (arXiv:[2607.02512](https://arxiv.org/abs/2607.02512),
July 2026, CC BY 4.0, code released): compile a natural-language function spec into a
~23 MB LoRA executed by a **frozen Qwen3-0.6B interpreter** — 0.6B+adapter beats
Qwen3-32B *prompting* on their FuzzyBench (73.8 vs 68.7) at 1/50 memory, 30 tok/s on
a MacBook. Covered task families = exactly the glue an agentic coding loop burns
tokens on: JSON repair, log triage, format conversion, parsing, tool calling (93 %
on ToolCall-15).

**v3 fit (two-tier local stack).**

    AVA 4B (VRAM)                  reasoning, code synthesis, planning
    0.6B + hot-swap LoRAs (RAM)    fuzzy glue: parse tool output, repair JSON,
                                   triage logs, rerank retrieval hits

The co-processor lives in system RAM beside the E1 memory tier — zero VRAM, same
philosophy (capability on the cheapest resource that carries it). Distribution
pattern generalizes E1's "texture pack": 23 MB hot-swappable **skill packs** on a
frozen base — procedures, not just facts; a v3.x can compile packs from its own
verified self-play data. Independent validation of the compute→weights thesis
(CODE_PIVOT §5 step 4) at per-function granularity.

**Honest caveats.** Single-step functions only (no multi-turn reasoning, no code
synthesis — authors' own limitation list); benchmark is self-defined and
synthetic (gpt-5.2-generated, LLM-verified); compiler↔interpreter coupled (we
consume their released pair, never train our own compiler); repo one week old.

**Status & gate.** Hypothesis, off critical path, v3.1+. Ablation A18: profile the
C6 agent loop — adopt only if the 4B provably wastes ≥ 10 % of agentic tokens on
glue subtasks that the 0.6B+adapter handles at ≥ parity accuracy. Else drop with
zero cost.

## What would kill each edge

| Edge | Kill condition | Survives as |
|---|---|---|
| E1 memory tier | A8 < +2 pp factual; or RAM lookup latency > 15 % decode budget | core v3 without memory; revisit at v3.1 |
| E2 latent curriculum | A9 < +1 pp MATH-500 | plain halting curriculum (already gated) |
| E3 coupled speculation | A11 frontier ≤ stock EAGLE-3 | stock EAGLE-3 (still 2×+) |
| E4 NSA H-blocks | RULER 128 K < 90 % | full-attention H + YaRN-128K |
| E5 precision thesis | ternary cliff at P5 (R4) | R4 fallback ladder |
| E6 μP | proxy LR fails to transfer | conventional sweep at 800 M only |
| E7 flywheel | reward hacking / no verified gain | static v3.0 (unchanged) |
| E8 tokenizer transfer | A15 probe drop > 1 pp after recovery | donor vocabulary (unchanged) |
| E9 loop stability + depth-index | A16/A17 no gain at proxy | current HRM loop (unchanged; both default off) |
| E10 fuzzy-glue co-processor | A18: glue < 10 % of agentic tokens, or 0.6B+adapter below parity | 4B handles its own glue (unchanged) |

Every kill condition leaves v3 ≥ the pre-edge design. The portfolio is strictly
additive in expectation — that is what makes it a moat rather than a bet.

---

## References (edge-specific)

- Berges et al., *Memory Layers at Scale*, arXiv:[2412.09764](https://arxiv.org/abs/2412.09764).
- Lample et al., *Large Memory Layers with Product Keys*, arXiv:[1907.05242](https://arxiv.org/abs/1907.05242).
- Hao et al., *Training LLMs to Reason in a Continuous Latent Space* (Coconut), arXiv:[2412.06769](https://arxiv.org/abs/2412.06769).
- *Emergence of Superposition: Training Dynamics of Chain of Continuous Thought*, arXiv:[2509.23365](https://arxiv.org/abs/2509.23365).
- *Persistent Memory for Continuous Latent Reasoning*, arXiv:[2606.07720](https://arxiv.org/abs/2606.07720).
- Snell et al., *Scaling LLM Test-Time Compute Optimally*, arXiv:[2408.03314](https://arxiv.org/abs/2408.03314).
- Li et al., *EAGLE-3*, arXiv:[2503.01840](https://arxiv.org/abs/2503.01840); llama.cpp PR [#18039](https://github.com/ggml-org/llama.cpp/pull/18039); SpecForge, arXiv:[2603.18567](https://arxiv.org/abs/2603.18567).
- Yuan et al., *Native Sparse Attention*, arXiv:[2502.11089](https://arxiv.org/abs/2502.11089); FSA kernels, OpenReview [c5mdo1hWrs](https://openreview.net/forum?id=c5mdo1hWrs).
- Kumar et al., *Scaling Laws for Precision*, arXiv:[2411.04330](https://arxiv.org/abs/2411.04330); *Scaling Law for QAT*, arXiv:[2505.14302](https://arxiv.org/abs/2505.14302); *Scaling Laws for Quantized LLMs at 100T tokens*, arXiv:[2411.17691](https://arxiv.org/abs/2411.17691).
- Yang et al., *Tensor Programs V (μP)*, arXiv:[2203.03466](https://arxiv.org/abs/2203.03466); Cerebras-GPT, arXiv:[2304.03208](https://arxiv.org/abs/2304.03208).
- Zhao et al., *Absolute Zero*, arXiv:[2505.03335](https://arxiv.org/abs/2505.03335).
- Behrouz et al., *Titans: Learning to Memorize at Test Time*, arXiv:[2501.00663](https://arxiv.org/abs/2501.00663) — evaluated for the memory tier; deferred to v3.1 pending the critical re-analysis in arXiv:[2510.09551](https://arxiv.org/abs/2510.09551) (reproduction concerns). E1's PKM design wins on evidence maturity.
- *Program-as-Weights*, arXiv:[2607.02512](https://arxiv.org/abs/2607.02512) — E10; code at github.com/programasweights.
- Qwen3.5/3.6 MTP: llama.cpp PR #22673 (`--spec-type draft-mtp`); unsloth MTP GGUFs for the full Qwen3.5 family — E3 revision basis.
