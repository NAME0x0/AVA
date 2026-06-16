# Research Round 6 — a v3-pattern model in the wild (2026-06-16)

> Reviewed `yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF`, a hobby
> coding model on Hugging Face. It is **not usable by v3** (wrong base, wrong
> license, wrong size, no benchmarks) but it is a real-world, $0, hobbyist
> instance of **v3's exact distillation recipe**, which both validates the
> design and surfaces one concrete refinement.

---

## What it is

A **distillation fine-tune of `google/gemma-4-12B-it`** on execution-verified
Python coding data, by a hobbyist ("personal/hobby project — shared as-is").
GGUF ladder Q2_K (4.83 GB) / Q4_K_M (7.38 GB, recommended) / Q6_K / Q8_0
(12.7 GB). 256K context, native thinking channel. No benchmarks reported.

The training recipe, quoted from the card:

- Primary teacher **Composer 2.5**: "the teacher solved each problem, its code
  was **run against the task's tests, and only the passing solutions were
  kept**."
- Fallback teacher **Fable 5**: "we took the problems where **Composer 2.5 got
  it wrong** and handed them to Fable 5 to *redo* — **re-deriving a fresh,
  self-consistent chain-of-thought** and a correct solution, again **gated on
  passing the tests**."
- Combined: "real CoT for the bulk … plus synthetic 'second-attempt' CoT to
  patch the failures — **both verified by execution before anything entered
  training**."

## Why it matters: it is v3's design, built by a hobbyist on free tools

This is precisely v3's round-2 multi-teacher distillation protocol:
**execution filter → failure-fallback routing to a second teacher → only
verified traces train the student.** Someone with no budget already runs this
recipe and ships GGUF — i.e. v3's exact method, audience, and deployment path,
confirmed practical at the hobby/$0 level. That is encouraging external
evidence for the most labor-intensive part of the v3 plan (C5 teacher panel).

## The one refinement adopted

The card is explicit that the fallback teacher **re-derives a fresh,
self-consistent CoT from scratch**, rather than repairing the failed trace.
This is a real distinction v3's round-2 protocol had left unspecified: a
stitched correction trains the student on a reasoning chain that does **not**
actually lead to the verified-correct answer, which is exactly the kind of
inconsistent supervision that degrades small students. Round-2 §3 step 2 is
tightened accordingly: **fallback = fresh derivation, execution-gated, never a
repair of the primary's trace.**

## What v3 does NOT take

- **The base model.** Gemma-4-12B is the wrong size (Q2_K alone is 4.83 GB,
  already over v3's 4 GB budget — a useful reminder of why v3 transplants a
  ~4 B donor and goes ternary instead of quantizing a 12 B) and carries the
  Gemma license, not Apache-2.0, so it fails v3's free-for-the-world mission.
  v3 already disqualified non-Apache, >4 GB donors.
- **"Reduced refusals / no safety hedging."** The card advertises a
  refusal-stripped posture. v3's posture is the opposite end: D2 calibrated
  honesty and D3 air-gap, not refusal removal. Explicitly not adopted.
- **The numbers.** No benchmarks are reported; quality is unverified. This is a
  recipe data point, not a results data point.

## Ledger

No code, no new claim, no new risk. One protocol tightening to round-2 §3
(fresh-derivation fallback) and external validation that v3's $0 multi-teacher
recipe is already practiced in the wild. Design otherwise frozen until C1.

## References (round 6)

- `huggingface.co/yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF`
  (model card; Composer 2.5 + Fable 5 execution-verified distillation).
- Cross-ref: [RESEARCH_ROUND_2.md](RESEARCH_ROUND_2.md) §3 (multi-teacher
  conflict protocol), [CODE_PIVOT.md](CODE_PIVOT.md) §5 (self-play /
  execution verification).
