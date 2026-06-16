# Research Round 5 — SubQ-1.1-Small review (2026-06-16)

> Subquadratic AI released the SubQ-1.1-Small model card + technical report.
> Read in full. **Net: it is not usable by v3 directly (wrong scale, closed,
> core mechanism withheld), but it is the strongest external validation yet of
> v3's central bet, and it yields one implemented technique plus several
> methodology lessons.** It does *not* reopen the long-context scope v3 cut.

---

## What SubQ-1.1-Small actually is

A **long-context** language model from Subquadratic AI, built by **converting an
open-weight frontier donor**: they replaced the donor's dense attention with
**Subquadratic Sparse Attention (SSA)** — a content-dependent learned sparse
attention with linear compute *and* memory end-to-end (including the
selection/indexing stage, unlike DeepSeek's quadratic Lightning Indexer) — then
ran staged context extension (262K → 512K → 1M → 2M via YaRN) with ~1T tokens
of long-context continued pretraining, then capability-balancing post-training.

Headline results: RULER 99.12% @128K; single-needle NIAH 100% @1M/2M and 98%
@6M/12M (≈12× the primary training window) while attending to **0.13% of token
pairs**; GPQA Diamond 85.4 pass@1; LiveCodeBench v6 89.7 pass@4; AutomationBench
Finance 13% (between Opus 4.8 16% and Sonnet 4.6 8%).

**"Small" is relative to frontier, not to v3.** Those are frontier-tier scores;
the model runs on B200/H100, is closed/commercial, and the **SSA mechanism is
explicitly out of scope** ("The mechanism by which SSA meets these requirements
is outside the scope of this report"). So: not a donor, not a teacher, not
implementable. We take validation and method, not the architecture.

## Why this matters a lot for v3: it validates the transplant thesis

v3's entire $0 plan rests on **donor transplant** (CODE_PIVOT §3): take a
permissive open-weight donor, surgically replace its sequence-mixing
(LoLCATs attention→Mamba-3), recover with continued pretraining. SubQ did
*exactly this move* — dense→SSA on a frontier donor, recover with CPT, staged
YaRN extension — and reached frontier-competitive quality. This is independent,
at-scale evidence that **attention-replacement-plus-recovery on a donor works**,
which is the single riskiest assumption in v3 (R-class, T1). It also reuses YaRN
staged extension, which v3 already plans for 8K→32–64K.

The difference: v3 replaces attention with a *recurrent/SSM* mixer (cheaper,
the compression-vs-retrieval tradeoff the SubQ report itself dwells on), while
SubQ keeps a sparse *attention* that preserves arbitrary-position retrieval.
SubQ's own §2.4 is a clear-eyed warning about exactly v3's choice — see the
hybrid risk below.

## Implemented: sample-level loss aggregation (SubQ §3.4)

The one cleanly-portable technique. SubQ averages cross-entropy **per example,
then over the batch**, instead of over all tokens, so a few very long sequences
do not dominate the gradient. Directly relevant once v3's YaRN-extended,
variable-length batches enter training.

Implemented as `V3Config.sample_level_loss` (default off — token-level is
unchanged for fixed-length training). 4 tests in
`tests/test_sample_level_loss.py`: runs/finite, differs from token-level on
unequal lengths, handles all-ignored rows, and **matches token-level exactly
when all examples have equal supervised length** (the correctness anchor).
Turn on for the long-context / mixed-length phases.

## Adopted as method (no code, but binding on the plan)

1. **Explicit recovery phases after every capability-damaging stage.** SubQ's
   earlier checkpoints "showed retrieval improved while knowledge-intensive
   evaluations regressed"; they fixed it with staged post-training that
   *optimizes multiple capabilities at once* and interleaves **recovery
   phases**. This is precisely v3's T1/T2 risk (linearize / MoE-ify damages the
   donor). Action: each transplant stage's gate becomes **multi-capability**
   (not just the target probe), and each stage is followed by an explicit
   recovery phase before the next surgery. Folded into CODE_PIVOT §3.

2. **Don't checkpoint-select on a single proxy.** SubQ found **MRCR moved the
   wrong way** relative to real tasks; **RULER tracked deployment behavior
   better**, and they kept *fixed qualitative spot-checks* (repo-scale code
   reasoning, multi-document synthesis, contract analysis) for selection.
   Action: v3 adds a small **fixed deployment-shaped spot-check set** for
   checkpoint selection alongside the eval matrix + decontamination gate
   (round 3) — a checkpoint that wins the proxy but loses the spot-checks does
   not ship.

3. **Coding data trains general routing.** SubQ reports code data *improved
   non-code long-context retrieval* "because code is dense with cross-position
   dependencies that train general routing behavior." Independent support for
   v3's coding-specialist thesis: the specialty is also a strong signal for the
   model's long-range/routing machinery, not a narrowing.

4. **Throughput-as-scaling-variable.** SubQ's real unlock was running 100+
   experiments cheaply, not the inference speedup. v3's analogs already exist —
   the checkpoint-anywhere fabric (Claim 6) + μP 50M proxies (E6). This
   reinforces prioritizing iteration velocity over single big runs.

## Sharpened risk: the hybrid multi-hop warning (MiniMax M1→M2)

SubQ §2.5 documents **MiniMax-M1 (hybrid lightning+full attention) → M2
(returned to full attention on every layer)**: during development hybrids
*matched full attention on standard benchmarks but showed clear deficits on
higher-order multi-hop reasoning at larger scale*, and the efficient-attention
support infra (low-precision state, prefix cache, speculative decoding) was less
mature. This is a direct caution for v3's hybrid Mamba-3 design. It does **not**
say hybrids fail — it says the deficit hides on release-length standard
benchmarks and surfaces on multi-hop reasoning. Mitigations v3 already has or
adds:
- Keep enough softmax layers, placed by evidence — v3's **induction-probe**
  keep-set (round 3) is exactly this, and it should be re-validated on a
  **multi-hop** probe, not just single-needle retrieval.
- Add an explicit **multi-hop reasoning probe** to the T1/T2 gate suite (not
  just single-key recall), since that is where SubQ/MiniMax saw the silent
  regression.
Recorded as R22.

## What v3 explicitly does NOT take from this

- **SSA itself** — undisclosed; not implementable. We do not invent a lookalike.
- **The long-context mission.** SubQ is a million-token model; v3 deliberately
  cut to 8K native + YaRN 32–64K for $0 reasons (CODE_PIVOT §6), leaning on the
  agentic harness for repo-scale work. SubQ is impressive but must not drag v3
  back toward chasing context length — that tradeoff was made on purpose.
- **The donor/weights** — closed commercial model, unknown base, no license to
  build on.

## Ledger

No new novelty claim — these are adopted techniques/lessons. Strengthens the
transplant bet (T1) with external at-scale evidence; adds R22 (hybrid multi-hop
deficit) to the risk register; adds `sample_level_loss` to the engine. Design
otherwise frozen until C1.

## References (round 5)

- Ramirez, Whedon, Vayani, Vo. *SubQ-1.1-Small Technical Report*, Subquadratic
  AI, 2026 (subq.ai). SSA mechanism withheld.
- DeepSeek-AI. *Native Sparse Attention*, arXiv:2502.11089 (2025);
  *DeepSeek-V4-Flash* CSA/HCA + Lightning Indexer (2026).
- MiniMax. *Why did MiniMax-M2 end up a full-attention model?* (2025);
  *MiniMax-M2*, arXiv:2605.26494 (2026) — the hybrid→full reversal.
- Xu et al. *From 128K to 4M (UltraLong)*, arXiv:2504.06214 (2025) — unmasked
  cross-document packing, which SubQ also uses.
- Jelassi et al. *Repeat after me: Transformers beat SSMs at copying*,
  ICML 2024 — the compression-vs-retrieval gap motivating the softmax keep-set.
