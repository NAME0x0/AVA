# Research Round 4 — OpenMythos review (2026-06-16)

> Two external links evaluated for v3. One yielded two adoptable, implemented
> refinements to the HRM loop; the other was off-topic. This round is small and
> concrete: code landed, tests pass, design stays frozen otherwise.

---

## The two links

| Link | What it is | Verdict |
|---|---|---|
| `github.com/kyegomez/OpenMythos` | MIT recurrent-depth transformer (RDT): prelude/recurrent/coda, GQA/MLA, sparse MoE, ACT halting, LTI-stable injection, loop-index embedding. ~5,100 LOC, real working code (not a skeleton — parses, has tests) | **Mine two ideas; skip the model.** It is a less-developed parallel cousin of v3 — no benchmarks, no donor transplant, no ternary, no edge portfolio. Not a donor, not a teacher. But two of its mechanisms fill genuine gaps in v3's HRM loop |
| `gist.github.com/gsans/…` | A "Claude Mythos 5 / Fable 5" system prompt (model behavior policy text) | **Irrelevant to v3.** No architecture, no training method, nothing to build a small coding model from. Named "Mythos" — likely why it was paired with OpenMythos — but unrelated. Noted and set aside |

Skepticism applied: kyegomez repos are frequently hype/skeletons, and OpenMythos cites June-2026 arXiv IDs (e.g. 2604.12946 "Parcae/Prairie") that cannot be verified. **The adopted math does not depend on those citations** — LTI stability via diagonal-SSM discretization is standard Mamba algebra, ACT is Graves 2016, and the loop-index embedding is a trivially-sound positional analog. We took the mechanisms, not the claims.

## What v3 already had (so we only take the deltas)

`engine/hrm_core.py` already implements a refinement loop with shared L-blocks,
**convergence-aware halting** that reads the fixed-point residual (more
sophisticated than OpenMythos's plain ACT), **perturbed-restart escape** (D3 —
OpenMythos has no equivalent), deep supervision with 1-step-gradient detach, and
zero-init conditioning that preserves the donor warm-start. v3 is ahead on
halting and escape. The two things it lacked:

### D5 — Loop-index embedding (implemented, gated A16)

v3's L-blocks received no explicit signal for *which iteration* they were on;
they distinguished depth only implicitly via the changing residual and cond
re-injection. OpenMythos's sinusoidal loop-index embedding (a RoPE analog over
recurrence depth) gives the shared weights an explicit depth signal so they can
implement functionally distinct operations early vs late in the loop.

Implementation: `loop_index_embedding()` + a **zero-init learned gate** in
`HRMRepeatUnit`. Default off; when enabled, the gate starts at 0 so a
warm-started network is unperturbed, then learns the signal in. Zero parameters
beyond the gate scalar.

### D6 — LTI-stable injection (implemented, gated A17)

v3's outer refinement carried state with an implicit unit coefficient
(`z_next = f(z)` where `f` includes the blocks' internal residual) — marginally
stable, able to drift, which is exactly the fixed-point blow-up that the D3
perturbed-restart band-aids. OpenMythos's LTI injection decays the carry with a
diagonal `A = exp(-exp(·)) ∈ [0,1)` — provable spectral radius < 1, the **same
discretization v3 already uses inside Mamba-3**, applied to the depth loop. It
is the *structural* fix to which D3 is the reactive patch.

Adaptation (the port was not a drop-in): v3's blocks already carry `z` via
internal residuals, so the OpenMythos form `A·h + B·e + Transformer` would
double-count the carry and *expand* the state. The corrected update stabilizes
the **increment**:

```
Δ = f(z) − z
z_next = A⊙z + Δ + B⊙cond        # decay carry, keep the refinement step
```

At `A = 1, B = 0` this is exactly the current rule, so the contraction is
opt-in. A zero-init interpolation gate makes it a no-op at enable for the donor
warm-start; the gate phases the contraction in. `LTIStableInjection` in
`engine/hrm_core.py`.

## Verification

- All existing tests unchanged and green (features default off): **18 → 23**
  with the 5 new tests in `tests/test_hrm_openmythos_features.py`.
- New tests prove: no-op-at-enable for both features (donor invariant held),
  loop-index varies per iteration, `A` is never expansive for any parameters
  (`A ∈ [0,1]`) and strictly contractive at init (`A ≈ 0.368`), and a forced-on
  LTI keeps the state finite under an expansive block that otherwise overflows.
- Config flags threaded through `V3Config` → `student_model`; smoke tests green.
- ruff clean.

## Gates (when to actually turn these on)

These are ablations, not commitments — both stay off until measured at the μP
proxy / C2:

- **A16 (loop-index)**: enable only if it improves the refinement curve
  (loss vs L-step) or MATH/code probe at equal steps; cost is ~0, so the bar is
  just "non-negative".
- **A17 (LTI stability)**: enable if it reduces training instability (grad-norm
  spikes, the failures D3 currently rescues) without hurting the refinement
  gain. If D3 restart frequency drops materially with A17 on, that is the win.

## Claims / risk ledger

No new novelty claim — these are adopted published-style mechanisms, not v3
inventions. They *strengthen* existing items: Claim 2 (convergence-aware halting
+ restart) gains a contractive-update companion that makes the restart rarer;
R-class loop-stability concerns in the HRM design get a structural mitigation.
Design otherwise frozen until C1 numbers — this round added safety, not scope.

## References (round 4)

- OpenMythos, kyegomez, MIT (recurrent-depth transformer implementation).
- Graves, *Adaptive Computation Time*, arXiv:1603.08983 (ACT — the halting prior).
- Gu & Dao, *Mamba* / Mamba-3 diagonal-SSM ZOH discretization (the `A = exp(-exp·)` form v3 already uses).
- Saunshi et al., *Reasoning with Latent Thoughts*, arXiv:2502.17416 (looped-transformer reasoning — plausibly real, cited by OpenMythos).
- OpenMythos-cited June-2026 IDs (2604.12946, 2604.07822) — **unverified**; adopted math is independent of them.
