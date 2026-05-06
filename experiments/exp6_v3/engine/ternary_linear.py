"""Ternary {-1, 0, +1} quantization-aware Linear (BitNet b1.58 / Bonsai style).

P0 stub. Real implementation lands at P2.

Reference papers:
- BitNet b1.58: arxiv.org/abs/2402.17764
- Bonsai 1-bit: prismml.com/news/bonsai-8b
- ParetoQ unified QAT: arxiv.org/abs/2502.02631
- BitNet Distillation (BitDistiller-3): arxiv.org/abs/2510.13998
"""
from __future__ import annotations

from typing import Any


class TernaryLinear:
    """Drop-in nn.Linear replacement with ternary weight QAT.

    Forward (training):
        y = x @ ((sign(W) * abs(W).round_to_unit_step()) / scale)
    where scale = mean(abs(W)) and rounding gives {-1, 0, +1}.

    The full-precision master weights are kept as `.weight_master`. Gradient is
    applied to the master, then quantized weights are recomputed each forward.

    Forward (inference):
        Pre-pack the ternary weights into 1.58-bit GGUF blocks. No master kept.
    """

    def __init__(self, in_features: int, out_features: int,
                  bias: bool = False, **kwargs: Any) -> None:
        raise NotImplementedError("P2 task — implement after design freeze")
