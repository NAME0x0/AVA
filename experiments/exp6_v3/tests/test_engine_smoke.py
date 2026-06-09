"""Smoke tests for the AVA v3 engine (P2).

Run from the repo root:
    pytest experiments/exp6_v3/tests/test_engine_smoke.py -q

All tests are CPU-only and use V3Config.tiny(); the full-size test
instantiates on the meta device (no memory allocated).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import (  # noqa: E402
    AVAv3ForCausalLM,
    MoTEFFN,
    ProductKeyMemory,
    ReferenceMamba3Mixer,
    TernaryLinear,
    V3Config,
    count_full_size_params,
)

torch.manual_seed(0)


def test_ternary_linear_quantizes_to_three_levels() -> None:
    layer = TernaryLinear(128, 32, group_size=64)
    trits, scales = layer.export_ternary()
    assert trits.shape == (32, 128)
    assert scales.shape == (32, 2)
    assert set(trits.unique().tolist()) <= {-1, 0, 1}
    # Fake-quant view must equal trits * scales exactly (bit-exact export).
    wq = layer.fake_quant_weight()
    rebuilt = (
        trits.view(32, 2, 64).float() * scales.view(32, 2, 1).float()
    ).view(32, 128)
    assert torch.allclose(wq, rebuilt.to(wq.dtype), atol=1e-3)


def test_ternary_linear_gradients_reach_master_weights() -> None:
    layer = TernaryLinear(64, 16, group_size=64)
    out = layer(torch.randn(4, 64))
    out.sum().backward()
    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()


def test_mote_ffn_routes_and_balances() -> None:
    ffn = MoTEFFN(
        hidden_size=64,
        intermediate_size_routed=32,
        intermediate_size_shared=64,
        num_routed_experts=4,
        top_k=2,
        group_size=64,
    )
    x = torch.randn(2, 8, 64)
    out = ffn(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    # Load tracking populated and sums to top_k fraction (2 slots / 4 experts).
    assert ffn.last_expert_load.sum().item() == pytest.approx(1.0, abs=1e-5)
    before = ffn.balance_bias.clone()
    ffn.update_balance_bias(lr=0.01)
    assert not torch.equal(before, ffn.balance_bias)


def test_reference_mamba3_forward_and_grad() -> None:
    mixer = ReferenceMamba3Mixer(hidden_size=64, num_heads=2, head_dim=32, state_size=8)
    x = torch.randn(2, 12, 64, requires_grad=True)
    y = mixer(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_full_model_forward_training_mode() -> None:
    cfg = V3Config.tiny()
    model = AVAv3ForCausalLM(cfg).train()
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(ids, labels=ids)
    assert out.logits.shape == (2, 16, cfg.vocab_size)
    assert out.loss is not None and torch.isfinite(out.loss)
    assert out.ponder_loss is not None and torch.isfinite(out.ponder_loss)
    out.loss.backward()
    # Training mode runs the full refinement budget in every unit.
    assert all(u.steps_used == cfg.max_l_steps for u in out.unit_outputs)
    # Deep supervision iterates collected at the configured steps.
    assert all(
        len(u.supervised_iterates) == len(cfg.supervised_steps) for u in out.unit_outputs
    )


def test_reasoning_budget_controls_compute() -> None:
    cfg = V3Config.tiny()
    model = AVAv3ForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        fast = model(ids, reasoning_budget=1)
        slow = model(ids, reasoning_budget=cfg.max_l_steps)
    assert all(u.steps_used == 1 for u in fast.unit_outputs)
    assert all(u.steps_used == cfg.max_l_steps for u in slow.unit_outputs)
    # Same token stream, different latent compute, same output shape.
    assert fast.logits.shape == slow.logits.shape


def test_adaptive_mode_runs_and_reports_steps() -> None:
    cfg = V3Config.tiny()
    model = AVAv3ForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        out = model(ids, reasoning_budget="adaptive")
    assert all(1 <= u.steps_used <= cfg.max_l_steps for u in out.unit_outputs)


def test_pkm_memory_forward_and_sparse_grad() -> None:
    mem = ProductKeyMemory(hidden_size=64, num_keys=16, key_dim=16, topk=4, num_heads=1)
    x = torch.randn(2, 8, 64, requires_grad=True)
    out = mem(x)
    assert out.shape == x.shape
    # Zero-init out_proj: memory branch is a no-op at step 0.
    assert out.abs().max().item() == 0.0
    out.sum().backward()
    assert x.grad is not None
    # Sparse touch: only retrieved value rows receive gradient.
    grad_rows = (mem.values.weight.grad.abs().sum(dim=-1) > 0).sum().item()
    assert grad_rows <= 2 * 8 * 4  # tokens x topk upper bound
    assert mem.keys.grad is not None and torch.isfinite(mem.keys.grad).all()


def test_memory_build_forward_tiny() -> None:
    cfg = V3Config.tiny()
    cfg.use_memory = True
    cfg.memory_num_keys = 16
    cfg.memory_key_dim = 16
    cfg.memory_topk = 4
    cfg.memory_heads = 1
    cfg.memory_units = (0, 1)
    model = AVAv3ForCausalLM(cfg).train()
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(ids, labels=ids)
    assert out.loss is not None and torch.isfinite(out.loss)
    out.loss.backward()
    assert model.memory is not None
    assert model.memory.values.weight.grad is not None


def test_memory_build_param_budget() -> None:
    cfg = V3Config()
    cfg.use_memory = True
    sizes = count_full_size_params(cfg)
    # EDGES.md E1: ~1.9B RAM-tier parameters on top of the ~3.24B core.
    assert sizes["memory_ram_tier"] / 1e9 == pytest.approx(1.89, abs=0.1)
    assert (sizes["total"] - sizes["memory_ram_tier"]) / 1e9 == pytest.approx(3.24, abs=0.1)


def test_full_size_parameter_budget() -> None:
    sizes = count_full_size_params()
    total_b = sizes["total"] / 1e9
    # ARCHITECTURE_V3.md target: ~3.26 B total (tolerance covers the
    # mixer-projection bookkeeping differences noted in the doc).
    assert 2.9 <= total_b <= 3.6, f"total params {total_b:.3f} B outside budget"
    assert sizes["routed_experts"] / 1e9 == pytest.approx(2.11, abs=0.1)
    assert sizes["embedding_tied"] == 248320 * 1792
