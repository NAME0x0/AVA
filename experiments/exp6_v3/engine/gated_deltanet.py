"""Gated DeltaNet fallback mixer (FLA toolkit wrapper).

Fallback L-block sublayer for risk R2 (Mamba-3 BF16 instability) — see
docs/v3/RISKS.md. One-line config flip: ``l_block.sublayer: gated_deltanet``
in configs/v3_student_arch.yaml.

This is a thin wrapper around the released FLA kernel; there is no pure-PyTorch
reference path here because the fallback only matters on CUDA training runs.

References:
- Gated DeltaNet: openreview.net/forum?id=r8H7xhYPwz (ICLR 2025)
- FLA toolkit: github.com/fla-org/flash-linear-attention
"""
from __future__ import annotations

from torch import Tensor, nn


class GatedDeltaNetBlock(nn.Module):
    """Sub-quadratic recurrent mixer via the released FLA GatedDeltaNet kernel."""

    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int = 14,
        num_kv_heads: int | None = None,
        head_dim: int = 128,
    ) -> None:
        super().__init__()
        try:
            from fla.layers.gated_deltanet import (  # type: ignore[import-not-found]
                GatedDeltaNet,
            )
        except ImportError:
            raise ImportError(
                "GatedDeltaNetBlock requires the flash-linear-attention package: "
                "pip install -U flash-linear-attention"
            ) from None
        self.inner = GatedDeltaNet(
            hidden_size=hidden_size,
            num_heads=num_q_heads,
            head_dim=head_dim,
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.inner(x)
        # FLA layers may return (output, aux, cache) tuples.
        return out[0] if isinstance(out, tuple) else out
