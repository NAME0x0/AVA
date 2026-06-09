# HRM-Text Integration

> **Source paper:** Wang et al., *Hierarchical Reasoning Model* — arXiv:2506.21734 (HRM-base, 27 M params, ARC-AGI / Sudoku / maze).
> **Productization:** Sapient Intelligence, HRM-Text, 1 B params, public launch 18 May 2026. Independent April 2026 numbers: MMLU 60.7, ARC-C 81.9, MATH 56.2, DROP 82.2. Trained on ~40 B effective tokens (≈ 1 000× less than typical pre-training).

The reason HRM-Text matters for AVA v3 is that it shows **latent multi-step reasoning works at the 1 B scale**, without chain-of-thought tokens, with extreme token efficiency. It is the only published 2026 architecture that closes AVA v2's math gap structurally rather than by adding parameters.

This document specifies how the HRM dual-recurrence is grafted onto our ternary-MoE student.

---

## 1. The original HRM-base mechanism, in one paragraph

HRM-base (Wang et al., 2506.21734) is a 27 M-parameter network with two interdependent recurrent modules: an **H-module** (high-level, slow) updates its hidden state once per "macro-step", while an **L-module** (low-level, fast) iterates many times inside each macro-step, conditioned on the current H-state. Reasoning happens entirely in the continuous latent space of the two modules — no chain-of-thought tokens are emitted. The number of iterations is decided per problem by a halting head (Q-learning-style adaptive computation time). On ARC-AGI, Sudoku, and maze tasks, the 27 M model beats much larger transformer LMs because the recurrence acts as a depth multiplier without an actual parameter increase.

HRM-Text (Sapient, May 2026) scales that recipe to a 1 B model trained on ~40 B tokens. The published numbers (MMLU 60.7, MATH 56.2) suggest the recipe generalizes beyond synthetic puzzles to general language modeling.

---

## 1b. What independent analyses found — and how v3 responds (June 2026 update)

Two independent studies of HRM-base landed after our original doc set. Both change *how* we build the recurrence, neither kills it. We integrate them rather than ignore them — this section is the delta.

**ARC Prize, "The Hidden Drivers of HRM's Performance on ARC-AGI"** ([arcprize.org/blog/hrm-analysis](https://arcprize.org/blog/hrm-analysis)):

- On the hidden ARC-AGI-1 test set HRM-base scores 32 % (vs 41 % claimed on public data); 2 % on ARC-AGI-2.
- **The H/L hierarchy itself is a small effect**: a vanilla transformer with identical parameter count lands within ~5 pp without tuning.
- **The outer refinement loop is the real engine**: going from 1 → 2 refinement loops at inference is worth +13 pp, and training with 16 refinement loops *doubles* performance vs training with 1.
- A large share of HRM's ARC score is task memorization (training only on the 400 eval tasks already yields 31 %).

**"Are Your Reasoning Models Reasoning or Guessing?" (arXiv:[2601.10679](https://arxiv.org/abs/2601.10679))** — mechanistic analysis:

- HRM recurrence converges to fixed points, and **gets trapped at the first fixed point it finds**, even when wrong — including failures on puzzles with a single unknown cell.
- Improvement across iterations is not gradual; correctness arrives in abrupt "grokking"-style jumps.
- Their **Augmented HRM** — data augmentation + input perturbation + bootstrapped restarts — lifts Sudoku-Extreme from 54.5 % → 96.9 %. Escaping bad fixed points is worth more than deeper convergence into them.

Scope caveat: both studies analyze the 27 M puzzle HRM, not the 1 B HRM-Text, and HRM-Text's independently verified language numbers stand. But the *mechanism* risk transfers, so v3's design responds in four ways:

| # | Design response | Source finding it answers |
|---|---|---|
| D1 | **Refinement-first training.** Deep supervision on intermediate L-iterates: CE+KL loss evaluated at L-steps 2, 4, 6 (not only the final iterate), with 1-step gradient detachment between segments (memory stays O(1)). We train the loop that is proven to carry the gains. | ARC Prize: refinement-loop count during *training* is the dominant lever |
| D2 | **Convergence-aware halting.** The halting head reads `[z_L ; z_H ; ‖z_L − z_L_prev‖]` — the fixed-point residual is an explicit input, so the head learns to distinguish "converged" from "stuck". | 2601.10679: blind halting cannot see fixed-point traps |
| D3 | **Latent restart escape.** Under `reasoning_budget="adaptive"`, if halting confidence < τ after the budget is spent, restart the L-loop once from `z_H + ε` (small Gaussian perturbation) and keep the higher-confidence iterate. Inference-only; zero training cost. | 2601.10679: perturbation + bootstrapping is the proven escape (54.5 → 96.9) |
| D4 | **Hierarchy held under an ablation gate.** Ablation A3 is now decisive, not exploratory: if the H/L split shows < 1 pp over an equal-parameter non-hierarchical recurrent stack at P3 scale, we drop the split and keep only the refinement loop + halting. Per ARC Prize, that retains most of the win. | ARC Prize: hierarchy ≈ +5 pp at best on puzzles |

Net effect: v3's bet is no longer "HRM hierarchy works" — it is the narrower, better-evidenced claim that **trained refinement loops with convergence-aware halting and escape restarts work**. The H/L split is an implementation detail we keep only while it pays for itself.

---

## 2. What we keep, what we change

| HRM feature | Kept in v3? | Notes |
|---|---|---|
| Two-stack recurrence (H + L) | ✅ | Core of v3 block |
| Single-forward-pass reasoning, no CoT tokens | ✅ | Inference API unchanged from a normal decoder |
| Adaptive halting (sigmoid + budget) | ✅ | Per-token, max 6 L-steps |
| Q-learning halting trainer | ⚠️ replaced | Use ACT-style ponder loss (simpler, differentiable, no replay buffer) |
| Pure-recurrent H and L | ⚠️ extended | H gets one softmax-attention sublayer + MoE FFN; L gets one Mamba-3 sublayer + MoE FFN — recurrence is across the time dimension, not just within hidden state |
| Trained from scratch on synthetic puzzles | ❌ | We warm-start from Qwen 3.6 27B dense and distill from the 35B-A3B MoE |
| Token-supervised | ❌ replaced | Token CE + teacher KL + MiniLM-attn distillation (BitDistiller recipe) |

The non-obvious choice is using **ponder loss** instead of HRM-base's Q-learning halting. We considered both:

| Halting trainer | Pros | Cons | Decision |
|---|---|---|---|
| HRM Q-learning | Used in original paper; converges on puzzle tasks | Requires reward signal + replay; brittle at 4 GB | Reject |
| ACT ponder loss | Differentiable, single forward pass, fits BitDistiller stage 2 | Slight bias toward longer ponders unless penalized | **Accept**, with target ponder penalty 0.01 |
| Fixed N = 4 | Trivial; identical compute every token | Wastes compute on easy tokens, starves hard ones | Reject |

The L-step budget (mean 2.5, max 6) is exposed to inference callers as a `reasoning_budget` parameter — see [`PERF_TARGETS.md`](PERF_TARGETS.md) for the eval-time setting.

---

## 3. Block-level integration

Recap from [ARCHITECTURE_V3.md](ARCHITECTURE_V3.md):

```
For each of 24 blocks, for each input token:

  ── H-step (runs once per token) ───────────────────────────────
  z_H  ←  Block.H( h_prev , token_embed )         // softmax attn + MoE
                                                  // RoPE applied here
  ── L-loop (runs up to 6 times per token) ─────────────────────
  z_L  ← z_H                                       // condition on H
  for k in 1..6:
      z_L  ←  Block.L( z_L , z_H , token_embed )  // Mamba-3 + MoE
      σ    ←  halting_head( z_L , z_H , ‖z_L − z_L_prev‖ )   // convergence-aware sigmoid (D2)
      if σ ≥ 0.5 and k ≥ 1: break
  // adaptive mode only (D3): if σ < τ_restart after k = max_steps,
  // restart once from z_H + ε and keep the higher-σ iterate
  ── output ─────────────────────────────────────────────────────
  h_block_out  ←  z_L
```

Two facts that make this work inside our 4 GB / 32 K-context budget:

1. **Activations across the L-loop are not all retained.** Only the latest `z_L` is propagated forward. Earlier iterates exist only inside the kernel. Memory cost of recurrence = 1 × hidden-state, regardless of L-steps.
2. **The Mamba-3 sublayer carries the "memory of the loop"** via its state vector. The complex-valued state update (Mamba-3 §3) is exactly the right shape for storing partial reasoning across L-steps. This is why we pair HRM with Mamba-3 specifically, not with Gated DeltaNet — see [`SUBQUADRATIC.md`](SUBQUADRATIC.md) §2.

---

## 4. Training the halting head

```
loss = α_ce  · CE(student_logits, target_token)
     + α_kl  · KL(student_logits || teacher_logits)
     + α_at  · MiniLM_attn_distill(student_attn, teacher_attn)
     + α_pl  · ponder_loss(halting_probs)

α_ce = 0.5    α_kl = 0.3    α_at = 0.15    α_pl = 0.05
```

Ponder loss (Graves 2016 ACT) gives gradient pressure to halt early on easy tokens and to keep iterating on hard ones. The 0.05 weight is small but non-zero — strong enough to keep mean L-steps near 2.5 without starving hard examples.

**Deep supervision (D1, June 2026 revision):** the CE and KL terms are evaluated not only at the final L-iterate but also at intermediate iterates 2 and 4 (weights 0.2 / 0.3 / 0.5 for iterates 2 / 4 / final), with the hidden state detached between supervised segments (HRM's 1-step gradient). This trains the refinement loop directly — the component the ARC Prize ablations identified as the actual performance driver — and keeps backprop memory independent of L-step count.

A simple ablation gate during training: if MMLU-Math sub-score regresses by > 1 pp after a halting-head update, freeze the head for one epoch.

---

## 5. Inference-time reasoning slider

The v3 inference API exposes:

```python
model.generate(prompt, reasoning_budget="auto")     # default, mean ~2.5
model.generate(prompt, reasoning_budget=1)          # fast: 1 L-step always
model.generate(prompt, reasoning_budget=6)          # max think: 6 L-steps always
model.generate(prompt, reasoning_budget="adaptive") # halt by sigmoid only
```

This is the v3 equivalent of OpenAI's "reasoning effort" knob, but it costs nothing extra at low budgets and no extra parameters at high budgets.

Throughput estimates on RTX A2000 (target, not yet measured):

| `reasoning_budget` | Mean L-steps/token | Decode tok/s | Use case |
|---|---|---|---|
| 1 | 1.0 | ~50 | Casual chat, code completion |
| `auto` (default) | ~2.5 | ~30 | General reasoning, default eval |
| 6 | 6.0 | ~12 | MATH-500, hard ARC-AGI tasks |

---

## 6. Open questions and ablations

These are the unsettled HRM-related questions for v3. Each gets one ablation during P3–P5 (see [RECIPE.md](RECIPE.md)):

| Ablation | Hypothesis to test |
|---|---|
| A1 | Removing L-loop entirely (budget=1) — how much MATH-500 do we lose? |
| A2 | Replacing Mamba-3 L-sublayer with a second softmax-attention layer — does HRM still work without subquadratic state? |
| A3 | H-step every layer vs every 4 layers — is per-block H necessary? |
| A4 | Ponder loss weight 0.01 / 0.05 / 0.10 — does the L-budget converge cleanly? |
| A5 | Reasoning budget at eval time — is `auto` always better than fixed 4? |
| A6 | Deep-supervision segment count (final-only vs 2/4/final vs every iterate) — does refinement-loop training transfer to text as it did on ARC puzzles? |
| A7 | Latent restart escape on/off at eval — does the 2601.10679 perturbation result transfer to MATH-500 / GSM8K? |

All ablations are small enough to run as 1 B-token reruns of Stage 1, parallelizable across rented GPUs.

---

## 7. References

- Wang et al., *Hierarchical Reasoning Model*, arXiv:2506.21734.
- Sapient Intelligence, *HRM-Text* press release and independent verification, April 2026.
- Graves, *Adaptive Computation Time for Recurrent Neural Networks*, 2016 (ACT ponder loss).
- ARC Prize, *The Hidden Drivers of HRM's Performance on ARC-AGI* — hierarchy ≈ small, refinement loop ≈ large.
- *Are Your Reasoning Models Reasoning or Guessing? A Mechanistic Analysis of Hierarchical Reasoning Models*, arXiv:2601.10679 — fixed-point traps, perturbed-restart escape.

Full citation list: [REFERENCES.md](REFERENCES.md).
