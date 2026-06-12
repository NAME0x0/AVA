# Research Round 3 — the de-risk package (2026-06-12)

> Third sweep, different character: rounds 1–2 *added capability*; this round
> **converts risk into evidence** on the existing plan. The design is
> saturated — each further document is worth less than one measured number.
> Everything here either de-risks T1/T2 (the failure bulge), hardens the
> training signal, or buys publication credibility. After this: C1, not more
> design.

---

## 1. Induction-aware layer selection for T1 — implemented

**Problem.** T1 keeps 9/36 donor layers as softmax attention by uniform 3:1
spacing — blind. Hybrid-architecture ablations (ICLR 2025 hybrid analysis;
*Systematic Analysis and Design Insights*, arXiv 2510.04800; *Functional
Component Ablation*, arXiv 2603.22473) consistently find: associative recall,
copying, and in-context learning live in a **small set of induction/retrieval
heads**, and hybrids keep their quality only when the *kept* attention layers
host them. Code is the worst-case domain — variable-name copying and API echo
are pure induction.

**Fix (implemented).** `experiments/exp6_v3/scripts/induction_probe.py`:
repeated-sequence probe `[A; A]` measures per-head attention mass on the
induction target (one past the previous occurrence), aggregates to layer
scores (mean of top-2 heads), and selects a spacing-constrained keep-set.
Offline tests in `tests/test_induction_probe.py` (4 pass). Run on the real
donor in 4-bit before T1 starts; the probe JSON becomes a training input,
checked into the run config.

**Gate addition (T1).** After linearization, re-run the probe on the hybrid:
kept layers must retain ≥ 80% of their pre-conversion induction score, and a
needle-in-repo retrieval probe must stay within 5 pp of donor. Catches
"linearized but lobotomized" before 80 M tokens are spent.

## 2. Repo-level topological packing (C3/C4 data)

DeepSeek-Coder proved it at scale: order files within a training context by
**dependency topology** (imports first), pack repo-wise rather than file-wise.
Cross-file completion improves materially; cost is a preprocessing script, not
GPU hours. The Stack v2 subsets get parsed import graphs (tree-sitter, all
matrix languages), topo-sorted, packed to 8 K with FIM applied *after* packing
so spans can cross file boundaries within a repo. This was missing from the
C3/C4 pipeline spec — now mandatory.

## 3. Decontamination + dedup gate (publication credibility)

A "publishable research-grade" claim dies at review without it:

- **Dedup**: MinHash-LSH near-dedup over all SFT/pretrain mixes (the standard
  BigCode pipeline, runs on CPU overnight).
- **Decontamination**: 10-gram + normalized-AST overlap against *every* eval
  set in the matrix (HumanEval+, MBPP+, LiveCodeBench, BigCodeBench,
  MultiPL-E, CRUXEval, CanItEdit, Aider, SWE-bench instances + tests).
  Matches are dropped and **counted in a published contamination report**
  per training phase.
- LiveCodeBench's time-windowing is kept as the temporal holdout: only
  problems published after the final data cut count for the headline number.

New risk entry R21; gate: contamination report ships with every released
checkpoint (folds into D6/D7 reporting).

## 4. Agentic task generation: SWE-Playground + SERA-style SVG

Round-2 left agentic training data as "Nemotron traces + self-play". Two 2025–26
additions close the gap between *function-level* self-play and *repo-level*
agentic skill:

- **SWE-Playground** (2025): fully synthetic SWE environments — generated
  repos, issues, tests, scaffolds. Infinite repo-level tasks without GitHub
  scraping; slots into C5/C6 as the bridge tier between AZR function tasks
  and real SWE-bench-style work.
- **SERA-style soft verification** (AllenAI 2026): brittle unit-test rewards
  invite hacking; SERA's soft-verification grading (partial credit from
  execution traces + diff plausibility) is the published antidote. Adopt for
  self-play reward shaping: hard execution filter for *inclusion*, soft
  verification for *ranking*.

## 5. Verifier hardening — protecting E7 from reward hacking

E7's kill condition is reward hacking. Defenses, cheapest first:

1. **Held-out hidden tests**: every self-play task generates 2× tests; half
   are hidden from the solving context. Pass = pass on hidden half.
2. **Mutation smoke-test on generated tests**: a test suite that kills < 30%
   of simple mutants (operator flips, off-by-one) is too weak to act as a
   verifier — task discarded. Catches "assert True" degeneracy mechanically.
3. **Property-based templates** (Hypothesis-style) for algorithmic task
   families — harder to overfit than example-based asserts.
4. **SERA soft verification** (see §4) for ranking among passing solutions —
   diff minimality (D5) enters the reward here.

## 6. Cheap training-recipe upgrades (no new risk)

- **Checkpoint EMA / last-k soup**: average the last k shard checkpoints
  before each gate eval (+0.5–1 pp typical, zero cost — the checkpoint fabric
  already stores them).
- **GKD on-policy lane (gated)**: donor and Qwen3.6-27B share the Qwen3
  tokenizer family — token-level on-policy distillation (student samples,
  teacher grades) is *possible* without a logit cache for that teacher only.
  Kaggle 2×T4: teacher int4 on GPU0, student QLoRA on GPU1. Gated experiment
  at C5 (A14): adopt only if it beats trace-SFT at equal GPU-hours. Text
  traces remain the default; Devstral/Nemotron lanes stay text-only.
- **Canon-layer ablation (A13, optional)**: lightweight causal-conv residual
  branches ("canon layers", Physics of LM 4.1) reported to deepen effective
  reasoning at small scale; trivially compatible with our blocks. Run at the
  50 M μP proxy only; promote only on a clear win.

## 7. What this round explicitly does *not* do

No new headline capability, no new demands, no new claims. Claims 1–8 and
demands D1–D9 are frozen until C1 produces numbers. The next improvement to
AVA v3 is **evidence**: donor baseline (C1) → probe-informed T1 → first gate.

## References (round 3)

- ICLR 2025, *Hybrid Transformer-Mamba Language Models* (layer-placement ablations).
- *Hybrid Architectures: Systematic Analysis and Design Insights*, arXiv 2510.04800.
- *Functional Component Ablation Reveals Specialization in Hybrid LMs*, arXiv 2603.22473.
- Wu et al., *Retrieval Head Mechanistically Explains Long-Context Factuality*, arXiv 2404.15574.
- DeepSeek-Coder, arXiv 2401.14196 (repo-level topological packing).
- SWE-Playground (2025); SERA, AllenAI 2026 (soft verification, open pipeline).
- Agarwal et al., *GKD: On-Policy Distillation*, arXiv 2306.13649.
- BigCode dedup/decontamination pipeline (The Stack); LiveCodeBench time-windowing.
- Allen-Zhu, *Physics of LM Part 4.1* (canon layers, 2025).
