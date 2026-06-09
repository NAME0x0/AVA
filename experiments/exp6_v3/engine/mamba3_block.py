"""L-block sequence mixer: Mamba-3-style complex MIMO SSM.

Two paths:

1. **Production** — wraps ``fla.layers.mamba3.Mamba3`` (Triton kernels, merged
   into flash-linear-attention April 2026). Selected with ``kernel="fla"``.
2. **Reference** — ``ReferenceMamba3Mixer`` below: a pure-PyTorch sequential
   recurrence implementing the Mamba-3 state update

       h_t = a_t * h_t-1 + B_t x_t^T        (a_t complex, per head/state)
       y_t = Re(C_t^H h_t) + D * x_t

   with data-dependent discretization ``a_t = exp(dt_t * (-exp(A_log) + i*theta))``.
   O(T) in Python — used for CPU smoke tests and as the numerical oracle for
   the Triton path. Selected with ``kernel="reference"``.

The complex-valued state is the Mamba-3 signature: phase lets the state carry
"what I already counted" information across HRM L-iterations
(docs/v3/SUBQUADRATIC.md section 2b).

References:
- Mamba-3 (ICLR 2026): arxiv.org/abs/2603.15569
- Gated DeltaNet fallback: openreview.net/forum?id=r8H7xhYPwz
- FLA toolkit: github.com/fla-org/flash-linear-attention
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ReferenceMamba3Mixer(nn.Module):
    """Pure-PyTorch Mamba-3-style mixer. Numerically faithful, not fast."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        state_size: int = 64,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.state_size = state_size
        inner = num_heads * head_dim
        self.inner = inner

        # x, z (gate), B, C, dt — single fused input projection.
        self.in_proj = nn.Linear(
            hidden_size,
            2 * inner + 2 * num_heads * state_size + num_heads,
            bias=False,
            dtype=dtype,
        )
        self.out_proj = nn.Linear(inner, hidden_size, bias=False, dtype=dtype)
        self.out_norm = nn.RMSNorm(inner, eps=1e-6)

        # Complex decay: a = exp(dt * (-exp(A_log) + i * theta)).
        self.A_log = nn.Parameter(torch.empty(num_heads, state_size))
        self.theta = nn.Parameter(torch.empty(num_heads, state_size))
        self.dt_bias = nn.Parameter(torch.empty(num_heads))
        self.D = nn.Parameter(torch.ones(num_heads))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            # Decay magnitudes spread over [0.5, 0.99]-ish (Mamba-2/3 init style).
            self.A_log.uniform_(math.log(0.5), math.log(8.0))
            # Phases spread over [0, pi/8] — small rotations at init for BF16
            # stability (Mamba-3 section 5.1 guidance).
            self.theta.uniform_(0.0, math.pi / 8.0)
            # dt softplus-inverse init around ~[1e-3, 1e-1].
            self.dt_bias.uniform_(math.log(math.expm1(1e-3)), math.log(math.expm1(1e-1)))

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        h_heads, s, d = self.num_heads, self.state_size, self.head_dim

        proj = self.in_proj(x).float()
        xs, z, bmat, cmat, dt = proj.split(
            [self.inner, self.inner, h_heads * s, h_heads * s, h_heads], dim=-1
        )
        xs = xs.view(b, t, h_heads, d)
        bmat = bmat.view(b, t, h_heads, s)
        cmat = cmat.view(b, t, h_heads, s)
        dt = F.softplus(dt + self.dt_bias.float())                     # [b, t, h]

        # Complex per-step decay a_t: [b, t, h, s].
        decay_re = -torch.exp(self.A_log.float())                       # [h, s]
        a = torch.exp(
            torch.complex(
                dt.unsqueeze(-1) * decay_re,
                dt.unsqueeze(-1) * self.theta.float(),
            )
        )

        state = torch.zeros(b, h_heads, s, d, dtype=torch.complex64, device=x.device)
        ys = []
        for step in range(t):
            update = bmat[:, step].unsqueeze(-1).to(torch.complex64) * xs[
                :, step
            ].unsqueeze(-2).to(torch.complex64)                         # [b, h, s, d]
            state = a[:, step].unsqueeze(-1) * state + update
            y = (cmat[:, step].unsqueeze(-1).to(torch.complex64).conj() * state).sum(
                dim=-2
            ).real                                                      # [b, h, d]
            ys.append(y)
        y = torch.stack(ys, dim=1)                                      # [b, t, h, d]
        y = y + self.D.float().view(1, 1, -1, 1) * xs                   # skip path

        y = y.reshape(b, t, self.inner)
        y = self.out_norm(y) * F.silu(z)                                # gated output
        return self.out_proj(y.to(x.dtype))


def build_l_mixer(
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    state_size: int = 64,
    kernel: str = "auto",
    dtype: torch.dtype | None = None,
) -> nn.Module:
    """Factory for the L-block sequence mixer.

    kernel:
      - "fla"            — fla.layers.mamba3 Triton path (production, P3+)
      - "gated_deltanet" — fla.layers.gated_deltanet fallback (risk R2)
      - "reference"      — pure-PyTorch oracle (CPU tests)
      - "auto"           — fla if importable, else reference
    """
    if kernel in ("fla", "auto"):
        try:
            from fla.layers.mamba3 import Mamba3  # type: ignore[import-not-found]

            return Mamba3(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                state_size=state_size,
            )
        except ImportError:
            if kernel == "fla":
                raise ImportError(
                    "kernel='fla' requires flash-linear-attention >= the April 2026 "
                    "release with fla.layers.mamba3. Install with: "
                    "pip install -U flash-linear-attention"
                ) from None
    if kernel == "gated_deltanet":
        from .gated_deltanet import GatedDeltaNetBlock

        return GatedDeltaNetBlock(
            hidden_size=hidden_size, num_q_heads=num_heads, head_dim=head_dim
        )
    return ReferenceMamba3Mixer(hidden_size, num_heads, head_dim, state_size, dtype)
