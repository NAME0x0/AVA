"""Mixture of Ternary Experts (MoTE) FFN block.

Architecture (per layer):
- Router: BF16 linear projection, top-K token-choice, bias-based load balance.
- Routed experts: N ternary FFNs (gate_proj + up_proj + down_proj), SwiGLU.
- Shared expert: 1 BF16 FFN, intermediate dim ~5x routed, always active.
- Combine: shared_out + sum_k(gate_k * routed_out_k).

P0 stub. Real implementation lands at P2.

References:
- DeepSeekMoE fine-grained: arxiv.org/abs/2401.06066
- MoTE: arxiv.org/abs/2506.14435
- Q-Sparse + BitNet + MoE: arxiv.org/html/2402.17764
- DynaExq dynamic expert quantization: (cite TBD)
"""
from __future__ import annotations

from typing import Any


class MoTEFFN:
    """MoE FFN with ternary routed experts + 1 BF16 shared expert."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_routed_experts: int = 32,
        num_shared_experts: int = 1,
        top_k: int = 4,
        shared_intermediate_multiplier: float = 5.0,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError("P2 task — implement after design freeze")
