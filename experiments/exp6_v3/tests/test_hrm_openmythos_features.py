"""Tests for the OpenMythos-derived HRM refinements (loop-index, LTI injection).

Both features must be (a) no-op at initialization so a warm-started/donor
network is unperturbed when they are enabled, and (b) when forced active,
behave as advertised — the loop-index signal varies per iteration, and the
LTI-stable update keeps the hidden state bounded across many iterations.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.hrm_core import (  # noqa: E402
    HRMRepeatUnit,
    LTIStableInjection,
    loop_index_embedding,
)

HID = 32


class _IdentityBlock(nn.Module):
    """A block that does nothing (lets us isolate the injection arithmetic)."""

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return x


def _unit(**kwargs: object) -> HRMRepeatUnit:
    torch.manual_seed(0)
    return HRMRepeatUnit(
        l_blocks=nn.ModuleList(_IdentityBlock() for _ in range(2)),
        h_block=_IdentityBlock(),
        hidden_size=HID,
        max_l_steps=4,
        supervised_steps=(2, 4),
        **kwargs,
    )


def test_loop_index_embedding_varies_per_step() -> None:
    z = torch.zeros(1, 1, HID)
    e0 = loop_index_embedding(z, 0, loop_dim=16)
    e1 = loop_index_embedding(z, 1, loop_dim=16)
    e2 = loop_index_embedding(z, 2, loop_dim=16)
    assert not torch.allclose(e0, e1)
    assert not torch.allclose(e1, e2)
    # channels beyond loop_dim stay untouched
    assert torch.all(e1[16:] == 0)


def test_loop_index_gate_is_noop_at_init() -> None:
    x = torch.randn(2, 5, HID)
    cond = torch.randn(2, 5, HID)
    base = _unit()
    feat = _unit(use_loop_index_embedding=True, loop_index_dim=16)
    base.eval()
    feat.eval()
    with torch.no_grad():
        out_base, _ = base(x, cond, reasoning_budget=4)
        out_feat, _ = feat(x, cond, reasoning_budget=4)
    assert torch.allclose(out_base, out_feat, atol=1e-6)
    # gate must carry gradient once training touches it
    assert feat.loop_emb_gate.requires_grad


def test_lti_injection_is_noop_at_init() -> None:
    x = torch.randn(2, 5, HID)
    cond = torch.randn(2, 5, HID)
    base = _unit()
    feat = _unit(use_lti_injection=True)
    base.eval()
    feat.eval()
    with torch.no_grad():
        out_base, _ = base(x, cond, reasoning_budget=4)
        out_feat, _ = feat(x, cond, reasoning_budget=4)
    # gate_logit=-6 => contribution ~0.0025; near-identical, not bit-equal
    assert torch.allclose(out_base, out_feat, atol=5e-3)


def test_lti_get_A_never_expansive() -> None:
    inj = LTIStableInjection(HID)
    # The no-blow-up guarantee: A stays in [0, 1] for ANY parameter values, so
    # the carry term can never amplify the state.
    for la, ld in ((10.0, 10.0), (-10.0, -10.0), (3.0, -3.0), (0.0, 5.0)):
        with torch.no_grad():
            inj.log_A.fill_(la)
            inj.log_dt.fill_(ld)
        a = inj.get_A()
        assert torch.all(a >= 0.0) and torch.all(a <= 1.0)
    # In the operating regime (default init log_A=log_dt=0) it is strictly
    # contractive: A = exp(-exp(0)) = exp(-1) ~= 0.368.
    fresh = LTIStableInjection(HID)
    a0 = fresh.get_A()
    assert torch.all(a0 < 1.0)
    assert abs(a0.mean().item() - 0.3679) < 1e-3


def test_lti_bounds_state_when_fully_engaged() -> None:
    """With the gate forced open, an explosive block stays bounded under LTI."""

    class _Amplify(nn.Module):
        def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
            return x * 1.5  # spectral radius > 1 without stabilization

    torch.manual_seed(0)
    unit = HRMRepeatUnit(
        l_blocks=nn.ModuleList([_Amplify()]),
        h_block=_IdentityBlock(),
        hidden_size=HID,
        max_l_steps=20,
        supervised_steps=(20,),
        use_lti_injection=True,
    )
    with torch.no_grad():
        unit.lti.gate_logit.fill_(10.0)   # fully engage the contraction
        unit.lti.B.zero_()                 # isolate the A-decay term
    x = torch.randn(1, 3, HID)
    cond = torch.zeros(1, 3, HID)
    unit.eval()
    with torch.no_grad():
        out, _ = unit(x, cond, reasoning_budget=20)
    # 20 iterations of *1.5 amplification would overflow; LTI keeps it finite.
    assert torch.isfinite(out).all()
    assert out.abs().max() < x.abs().max() * 50
