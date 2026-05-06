"""Gated DeltaNet attention sublayer (FLA toolkit wrapper).

Hybrid stack: 3 × GatedDeltaNet + 1 × GatedAttention per repeat unit (Qwen 3.6
pattern). Linear-time decode, chunkwise prefill, RoPE only inside the
GatedAttention layers.

P0 stub. Real implementation lands at P2 — wraps fla-org/flash-linear-attention
kernels rather than reimplementing.

References:
- Gated DeltaNet: openreview.net/forum?id=r8H7xhYPwz (ICLR 2025)
- FLA toolkit: github.com/fla-org/flash-linear-attention
- Qwen 3.6 layer layout: huggingface.co/Qwen/Qwen3.6-27B (config.json)
"""
from __future__ import annotations

from typing import Any


class GatedDeltaNetBlock:
    """Sub-quadratic recurrent attention block via FLA kernels."""

    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int = 48,
        num_kv_heads: int = 16,
        head_dim: int = 128,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError("P2 task — wire FLA kernels")
