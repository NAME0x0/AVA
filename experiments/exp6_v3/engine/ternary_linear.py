"""Ternary {-1, 0, +1} quantization-aware Linear (BitNet b1.58 family).

Implements the BitNet b1.58 absmean quantization rule with group-wise scales:

    for each group g of `group_size` consecutive input weights in a row:
        gamma_g = mean(|W_g|) + eps
        W_q_g   = clip(round(W_g / gamma_g), -1, +1) * gamma_g

Training uses the straight-through estimator (STE): the forward pass sees the
quantized weights, the backward pass updates the full-precision master weights.

Group size 256 matches upstream llama.cpp TQ1_0/TQ2_0 blocks, so the exported
GGUF reproduces the QAT forward pass bit-for-bit (docs/v3/PRISMML.md section 2).

Activations stay in the compute dtype: llama.cpp ternary kernels quantize
activations internally (Q8_K), and training-time activation quantization is
deliberately deferred to a P5 ablation (docs/v3/RECIPE.md).

References:
- BitNet b1.58: arxiv.org/abs/2402.17764
- BitNet b1.58 2B4T technical report: arxiv.org/abs/2504.12285
- ParetoQ unified QAT: arxiv.org/abs/2502.02631
- BitDistiller: arxiv.org/abs/2402.10631
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

_EPS = 1e-5


class TernaryLinear(nn.Module):
    """Drop-in nn.Linear replacement with group-wise ternary weight QAT.

    The master weights (``self.weight``) stay full precision and receive the
    gradients; every forward pass re-derives the ternary fake-quant view.
    ``export_ternary()`` emits the integer trits + FP16 scales for GGUF
    packing, using the *same* scale rule as the forward pass.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        group_size: int = 256,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Group along the input dimension. If the row is shorter than the
        # group (tiny test configs), fall back to one scale per row.
        if in_features % group_size != 0:
            group_size = in_features
        self.group_size = group_size

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias: nn.Parameter | None = nn.Parameter(
                torch.zeros(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def _group_scales(self, w32: Tensor) -> Tensor:
        """Absmean scale per (row, group). Shape [out, in // group_size, 1]."""
        grouped = w32.view(self.out_features, -1, self.group_size)
        return grouped.abs().mean(dim=-1, keepdim=True).clamp_min(_EPS)

    def fake_quant_weight(self) -> Tensor:
        """Dequantized ternary view of the master weight (no STE applied)."""
        w32 = self.weight.float()
        scales = self._group_scales(w32)
        grouped = w32.view(self.out_features, -1, self.group_size)
        quant = (grouped / scales).round().clamp_(-1.0, 1.0) * scales
        return quant.view_as(self.weight).to(self.weight.dtype)

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        w_q = self.fake_quant_weight()
        # Straight-through estimator: forward sees w_q, backward updates w.
        w_ste = w + (w_q - w).detach()
        return F.linear(x, w_ste, self.bias)

    @torch.no_grad()
    def export_ternary(self) -> tuple[Tensor, Tensor]:
        """Return (trits int8 [out, in], scales fp16 [out, in // group_size]).

        Bit-exact with ``fake_quant_weight``; consumed by
        ``scripts/export_gguf.py`` for TQ1_0/TQ2_0 packing.
        """
        w32 = self.weight.float()
        scales = self._group_scales(w32)
        grouped = w32.view(self.out_features, -1, self.group_size)
        trits = (grouped / scales).round().clamp_(-1.0, 1.0).to(torch.int8)
        return trits.view(self.out_features, self.in_features), scales.squeeze(-1).half()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, group_size={self.group_size}"
        )
