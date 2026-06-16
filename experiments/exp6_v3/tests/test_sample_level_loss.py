"""Tests for SubQ-style sample-level loss aggregation (V3Config.sample_level_loss)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.student_model import AVAv3ForCausalLM, V3Config  # noqa: E402


def _tiny(sample_level: bool) -> AVAv3ForCausalLM:
    cfg = V3Config.tiny()
    cfg.sample_level_loss = sample_level
    torch.manual_seed(0)
    return AVAv3ForCausalLM(cfg)


def test_sample_level_runs_and_is_finite() -> None:
    model = _tiny(sample_level=True)
    ids = torch.randint(0, model.cfg.vocab_size, (3, 16))
    out = model(ids, labels=ids, reasoning_budget=2)
    assert out.loss is not None and torch.isfinite(out.loss)
    assert torch.isfinite(out.lm_loss)


def test_token_vs_sample_differ_on_unequal_lengths() -> None:
    """With one long and one short example, the two aggregations must differ."""
    tok = _tiny(sample_level=False)
    smp = _tiny(sample_level=True)
    smp.load_state_dict(tok.state_dict())  # identical weights, only agg differs

    ids = torch.randint(0, tok.cfg.vocab_size, (2, 20))
    labels = ids.clone()
    # Example 0 is "short": mask all but the first 4 target positions.
    labels[0, 5:] = -100

    with torch.no_grad():
        lt = tok(ids, labels=labels, reasoning_budget=2).lm_loss
        ls = smp(ids, labels=labels, reasoning_budget=2).lm_loss
    # Token-level weights by token count (long example dominates); sample-level
    # gives the two examples equal weight. They must not coincide here.
    assert not torch.isclose(lt, ls, atol=1e-4)


def test_sample_level_handles_all_ignored_row() -> None:
    model = _tiny(sample_level=True)
    ids = torch.randint(0, model.cfg.vocab_size, (2, 12))
    labels = ids.clone()
    labels[0, :] = -100  # entire example has no supervised tokens
    out = model(ids, labels=labels, reasoning_budget=2)
    assert out.loss is not None and torch.isfinite(out.loss)


def test_equal_lengths_token_and_sample_match() -> None:
    """When every example has the same number of supervised tokens, the two
    aggregations are mathematically identical."""
    tok = _tiny(sample_level=False)
    smp = _tiny(sample_level=True)
    smp.load_state_dict(tok.state_dict())
    ids = torch.randint(0, tok.cfg.vocab_size, (4, 16))  # full-length, equal
    with torch.no_grad():
        lt = tok(ids, labels=ids, reasoning_budget=2).lm_loss
        ls = smp(ids, labels=ids, reasoning_budget=2).lm_loss
    assert torch.isclose(lt, ls, atol=1e-5)
