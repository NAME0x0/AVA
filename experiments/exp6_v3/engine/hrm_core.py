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

    def _run_l_group(self, z: Tensor, cond: Tensor, **block_kwargs: object) -> Tensor:
        z = z + self.cond_proj(cond)
        for blk in self.l_blocks:
            z = blk(z, **block_kwargs)
        return z

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
            z = self._run_l_group(z, cond, **block_kwargs)
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
