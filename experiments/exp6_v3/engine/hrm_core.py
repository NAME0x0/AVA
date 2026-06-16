"""HRM-style refinement recurrence — June 2026 revision (refinement-first).

Implements the design responses D1–D3 from docs/v3/HRM_TEXT.md section 1b:

- **D1 refinement loop**: the unit's L-blocks are re-applied up to
  ``max_l_steps`` times with shared weights (depth multiplier, zero extra
  parameters). For training, intermediate iterates at the configured
  supervision steps are returned to the trainer, and the hidden state is
  detached between supervised segments (HRM 1-step gradient — backprop memory
  independent of step count).
- **D2 convergence-aware halting**: the halting head reads
  ``[z_L ; cond ; ||z_L - z_prev|| / sqrt(d)]`` so it can distinguish
  "converged" from "stuck at a wrong fixed point" (arXiv:2601.10679).
- **D3 latent restart escape**: in adaptive mode at inference, if halting
  confidence stays below a threshold after the budget is spent, the loop
  restarts once from a perturbed initial state and the higher-confidence
  outcome wins (perturbation+bootstrapping result of arXiv:2601.10679).

Two optional refinements ported from OpenMythos (kyegomez, MIT) after the
2026-06-16 review (docs/v3/RESEARCH_ROUND_4.md), both **default-off** and
**no-op at initialization** so the donor warm-start stays bit-identical and
the existing tested forward path is unchanged:

- **D5 loop-index embedding** (``use_loop_index_embedding``): a sinusoidal
  depth-index signal added before each L-group, scaled by a zero-init learned
  gate, so shared loop weights can tell which iteration they are on (a RoPE
  analog over recurrence depth instead of token position).
- **D6 LTI-stable injection** (``use_lti_injection``): an opt-in contractive
  update ``z_k = (1-g)·f(z) + g·(A⊙z + (f(z)-z) + B⊙cond)`` with a diagonal
  ``A = exp(-exp(·)) ∈ [0,1)`` guaranteeing spectral radius < 1 by
  construction — the same SSM discretization v3 already uses inside Mamba-3,
  applied to the outer refinement loop. The increment ``f(z)-z`` is kept while
  the carry is decayed (v3 blocks already carry ``z`` via internal residuals).
  The interpolation gate ``g`` is zero-init, so at enable time the update
  equals the current rule; ``g`` lets the provable-contraction term phase in.
  This is the structural counterpart to the D3 perturbed-restart band-aid for
  fixed-point blow-up.

Halting is trained with an ACT-style ponder objective (Graves 2016): the unit
returns the differentiable expected step count; the P4 trainer regresses it
toward ``mean_l_steps_target`` (docs/v3/RECIPE.md).

P2 scope notes (deliberate simplifications, revisited at P4/P9):
- Training runs all ``max_l_steps`` iterations; the halting head receives
  gradient through the ponder objective, not through an ACT output mixture.
- Inference halting is decided on the batch-mean halting probability;
  per-token state freezing lands together with the inference state cache.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn


@dataclass
class HRMOutput:
    """Per-unit recurrence diagnostics and training signals."""

    hidden: Tensor
    expected_steps: Tensor          # [b, t] differentiable E[steps] (ponder signal)
    halt_probs: Tensor              # [k, b, t] per-iteration halting probabilities
    steps_used: int                 # actual L-iterations executed
    restarted: bool = False         # D3 escape fired
    supervised_iterates: list[Tensor] = field(default_factory=list)


def loop_index_embedding(
    z: Tensor, loop_t: int, loop_dim: int, theta: float = 10000.0
) -> Tensor:
    """Sinusoidal embedding of the recurrence-depth index (additive bias).

    Analogous to RoPE for token position, but over loop iteration. Returns a
    bias of shape ``[hidden]`` (broadcast over batch/time); only the first
    ``loop_dim`` channels are populated. Ported from OpenMythos (MIT).
    """
    half = torch.arange(0, loop_dim, 2, device=z.device, dtype=torch.float32)
    freqs = 1.0 / (theta ** (half / loop_dim))
    angles = loop_t * freqs
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)[:loop_dim]
    full = torch.zeros(z.shape[-1], device=z.device, dtype=torch.float32)
    full[:loop_dim] = emb
    return full.to(z.dtype)


class LTIStableInjection(nn.Module):
    """Provably contractive injection for the outer refinement loop.

    v3's L-blocks already carry the state through their own internal residuals,
    so ``transformer_out`` already contains a unit-coefficient copy of ``z``.
    Stabilizing the *increment* rather than re-adding the carry, the update is

        Δ = transformer_out − z                  (the loop's refinement step)
        z_next = A⊙z + Δ + B⊙cond                (decay the carry, keep Δ)

    with diagonal ``A = exp(-exp(clamp(log_dt + log_A))) ∈ [0,1)`` guaranteeing
    spectral radius < 1 by construction (ZOH discretization of a negative-real
    diagonal continuous-time SSM — the same form as Mamba-3's per-step decay).
    At ``A = 1, B = 0`` this is exactly the current rule (``z_next =
    transformer_out``), so the contraction is opt-in.

    A zero-init interpolation gate ``g = sigmoid(gate_logit)`` with
    ``gate_logit`` init very negative makes the module's contribution start at
    ~0, so enabling it does not perturb a warm-started (donor) network; the
    gate learns the contraction term in. Ported/adapted from OpenMythos (MIT).
    """

    def __init__(self, hidden_size: int, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))
        self.log_dt = nn.Parameter(torch.zeros(1, dtype=dtype))
        self.B = nn.Parameter(torch.full((hidden_size,), 0.1, dtype=dtype))
        # gate init -> sigmoid(-8) ~= 3e-4: effectively the current rule at enable.
        self.gate_logit = nn.Parameter(torch.full((1,), -8.0, dtype=dtype))

    def get_A(self) -> Tensor:
        return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)))

    def forward(self, z: Tensor, cond: Tensor, transformer_out: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate_logit).to(z.dtype)
        delta = transformer_out - z
        stabilized = self.get_A().to(z.dtype) * z + delta + self.B.to(z.dtype) * cond
        return (1.0 - g) * transformer_out + g * stabilized


class HaltingHead(nn.Module):
    """Convergence-aware halting: sigma = f(z_L, cond, fixed-point residual)."""

    def __init__(self, hidden_size: int, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        bottleneck = max(hidden_size // 4, 8)
        self.net = nn.Sequential(
            nn.Linear(2 * hidden_size + 1, bottleneck, dtype=dtype),
            nn.GELU(),
            nn.Linear(bottleneck, 1, dtype=dtype),
        )
        with torch.no_grad():
            final = self.net[-1]
            assert isinstance(final, nn.Linear)
            if final.bias is not None and final.bias.device.type != "meta":
                # Start biased toward continuing (~12% halt prob) so early
                # training actually exercises the refinement loop.
                final.bias.fill_(-2.0)

    def forward(self, z: Tensor, cond: Tensor, residual_norm: Tensor) -> Tensor:
        feats = torch.cat([z, cond, residual_norm], dim=-1)
        return torch.sigmoid(self.net(feats)).squeeze(-1)               # [b, t]


class HRMRepeatUnit(nn.Module):
    """One repeat unit: refinement loop over the L-blocks, then the H-block.

    The unit does not build its own blocks — the model passes them in — so
    this module stays architecture-agnostic (works for Mamba-3, Gated
    DeltaNet, or attention L-blocks alike).

    Conditioning: ``cond`` is the previous unit's H output (token embeddings
    for the first unit), injected through a zero-initialized projection so the
    warm-started network behaves like a plain feed-forward stack at step 0.
    """

    def __init__(
        self,
        l_blocks: nn.ModuleList,
        h_block: nn.Module,
        hidden_size: int,
        max_l_steps: int = 6,
        supervised_steps: tuple[int, ...] = (2, 4, 6),
        detach_between_segments: bool = True,
        restart_confidence_threshold: float = 0.35,
        restart_perturbation_std: float = 0.02,
        use_loop_index_embedding: bool = False,
        loop_index_dim: int = 64,
        use_lti_injection: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.l_blocks = l_blocks
        self.h_block = h_block
        self.max_l_steps = max_l_steps
        self.supervised_steps = supervised_steps
        self.detach_between_segments = detach_between_segments
        self.restart_confidence_threshold = restart_confidence_threshold
        self.restart_perturbation_std = restart_perturbation_std

        self.cond_proj = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        with torch.no_grad():
            if self.cond_proj.weight.device.type != "meta":
                self.cond_proj.weight.zero_()
        self.halting_head = HaltingHead(hidden_size, dtype=dtype)
        self._sqrt_d = float(hidden_size) ** 0.5

        # D5 — loop-index embedding (default off; zero-init gate => no-op at enable).
        self.use_loop_index_embedding = use_loop_index_embedding
        self.loop_index_dim = min(loop_index_dim, hidden_size)
        if use_loop_index_embedding:
            self.loop_emb_gate = nn.Parameter(torch.zeros(1, dtype=dtype))
        # D6 — LTI-stable injection (default off; contractive outer-loop update).
        self.use_lti_injection = use_lti_injection
        self.lti = (
            LTIStableInjection(hidden_size, dtype=dtype) if use_lti_injection else None
        )

    def _run_l_group(
        self, z: Tensor, cond: Tensor, loop_t: int = 0, **block_kwargs: object
    ) -> Tensor:
        z_in = z + self.cond_proj(cond)
        if self.use_loop_index_embedding:
            bias = loop_index_embedding(z_in, loop_t, self.loop_index_dim)
            z_in = z_in + self.loop_emb_gate.to(z_in.dtype) * bias
        out = z_in
        for blk in self.l_blocks:
            out = blk(out, **block_kwargs)
        if self.lti is not None:
            out = self.lti(z, cond, out)
        return out

    def _refine(
        self,
        x: Tensor,
        cond: Tensor,
        steps: int,
        adaptive: bool,
        collect_supervised: bool,
        **block_kwargs: object,
    ) -> HRMOutput:
        z = x
        z_prev = x
        halt_probs: list[Tensor] = []
        supervised: list[Tensor] = []
        survival = torch.ones(x.shape[:-1], device=x.device, dtype=torch.float32)
        expected = torch.zeros_like(survival)
        steps_used = 0

        for k in range(1, steps + 1):
            z = self._run_l_group(z, cond, loop_t=k - 1, **block_kwargs)
            residual = (z - z_prev).float().norm(dim=-1, keepdim=True) / self._sqrt_d
            p = self.halting_head(z, cond, residual.to(z.dtype))        # [b, t]
            halt_probs.append(p)
            expected = expected + survival                              # E[steps] = sum_k q_k
            survival = survival * (1.0 - p.float())
            steps_used = k
            z_prev = z

            if collect_supervised and k in self.supervised_steps:
                supervised.append(z)
                if self.detach_between_segments and k != steps:
                    z = z.detach()                                       # 1-step gradient
                    z_prev = z_prev.detach()

            if adaptive and not self.training and k >= 1 and p.mean().item() >= 0.5:
                break

        return HRMOutput(
            hidden=z,
            expected_steps=expected,
            halt_probs=torch.stack(halt_probs, dim=0),
            steps_used=steps_used,
            supervised_iterates=supervised,
        )

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        reasoning_budget: int | str = "auto",
        **block_kwargs: object,
    ) -> tuple[Tensor, HRMOutput]:
        """Returns (unit output after H-block, recurrence diagnostics)."""
        if self.training:
            result = self._refine(
                x, cond, self.max_l_steps, adaptive=False,
                collect_supervised=True, **block_kwargs,
            )
        elif isinstance(reasoning_budget, int):
            steps = max(1, min(reasoning_budget, self.max_l_steps))
            result = self._refine(
                x, cond, steps, adaptive=False, collect_supervised=False, **block_kwargs
            )
        else:
            result = self._refine(
                x, cond, self.max_l_steps, adaptive=True,
                collect_supervised=False, **block_kwargs,
            )
            if (
                reasoning_budget == "adaptive"
                and result.halt_probs[-1].mean().item() < self.restart_confidence_threshold
            ):
                # D3: escape a suspected wrong fixed point — one perturbed retry.
                noise = torch.randn_like(x) * self.restart_perturbation_std
                retry = self._refine(
                    x + noise, cond, self.max_l_steps, adaptive=True,
                    collect_supervised=False, **block_kwargs,
                )
                if retry.halt_probs[-1].mean().item() > result.halt_probs[-1].mean().item():
                    result = retry
                    result.restarted = True

        out = self.h_block(result.hidden, **block_kwargs)
        result.hidden = out
        return out, result
