"""Mixture of Ternary Experts (MoTE) FFN block.

Per layer:
- Router: full-precision linear, sigmoid scoring, top-K token-choice.
  Load balancing is bias-based (DeepSeek-V3 style): a non-trainable per-expert
  bias is added to the scores *for selection only* and nudged outside the
  gradient path; no auxiliary loss term.
- Routed experts: N ternary 2-matrix FFNs (up_proj -> ReLU^2 -> down_proj).
  The 2-matrix ReLU^2 expert follows BitNet b1.58 2B4T and matches the v3
  parameter budget in docs/v3/ARCHITECTURE_V3.md (32 x 2 x 1792 x 768).
- Shared expert: 1 ternary FFN, larger intermediate, always active.
- Combine: out = shared(x) + sum_k gate_k * expert_k(x).

The expert loop below is correctness-first (token gather per expert). The
fused grouped-GEMM path is a P3 optimization; this module is the numerical
reference for it.

References:
- DeepSeekMoE fine-grained experts: arxiv.org/abs/2401.06066
- DeepSeek-V3 bias-based load balance: arxiv.org/abs/2412.19437
- MoTE ternary experts: arxiv.org/abs/2506.14435
- BitNet b1.58 2B4T (ReLU^2 FFN): arxiv.org/abs/2504.12285
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .ternary_linear import TernaryLinear


class TernaryExpert(nn.Module):
    """2-matrix ternary FFN: down(relu(up(x))^2)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        group_size: int = 256,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.up_proj = TernaryLinear(hidden_size, intermediate_size, group_size=group_size, dtype=dtype)
        self.down_proj = TernaryLinear(intermediate_size, hidden_size, group_size=group_size, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        h = F.relu(self.up_proj(x))
        return self.down_proj(h * h)


class MoTEFFN(nn.Module):
    """MoE FFN with ternary routed experts + 1 always-active shared expert."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size_routed: int,
        intermediate_size_shared: int,
        num_routed_experts: int = 32,
        top_k: int = 4,
        group_size: int = 256,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k

        self.router = nn.Linear(hidden_size, num_routed_experts, bias=False, dtype=dtype)
        # Selection-only bias, updated outside autograd (update_balance_bias).
        self.register_buffer("balance_bias", torch.zeros(num_routed_experts))
        # Fraction of routed slots each expert served in the last forward.
        self.register_buffer("last_expert_load", torch.zeros(num_routed_experts))

        self.experts = nn.ModuleList(
            TernaryExpert(hidden_size, intermediate_size_routed, group_size, dtype)
            for _ in range(num_routed_experts)
        )
        self.shared_expert = TernaryExpert(hidden_size, intermediate_size_shared, group_size, dtype)

    def forward(self, x: Tensor) -> Tensor:
        batch_shape = x.shape[:-1]
        flat = x.reshape(-1, self.hidden_size)

        scores = torch.sigmoid(self.router(flat))                      # [T, E]
        biased = scores + self.balance_bias.to(scores.dtype)
        _, sel = biased.topk(self.top_k, dim=-1)                       # [T, K]
        gates = scores.gather(-1, sel)                                 # [T, K]
        gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        out = self.shared_expert(flat)
        for e in range(self.num_routed_experts):
            token_idx, slot_idx = (sel == e).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            expert_out = self.experts[e](flat[token_idx])
            out = out.index_add(
                0, token_idx, expert_out * gates[token_idx, slot_idx].unsqueeze(-1)
            )

        if not torch.jit.is_scripting():
            with torch.no_grad():
                load = torch.bincount(sel.reshape(-1), minlength=self.num_routed_experts)
                self.last_expert_load.copy_(load.float() / max(sel.numel(), 1))

        return out.reshape(*batch_shape, self.hidden_size)

    @torch.no_grad()
    def update_balance_bias(self, lr: float = 1e-3) -> None:
        """DeepSeek-V3 bias nudge: lighten overloaded experts, boost starved ones.

        Call once per optimizer step from the trainer, after forward.
        """
        load = self.last_expert_load
        self.balance_bias += lr * torch.sign(load.mean() - load)
