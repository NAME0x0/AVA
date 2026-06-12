"""Offline tests for the T1 induction-head probe (tiny random model, CPU)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.induction_probe import (  # noqa: E402
    build_repeated_batch,
    induction_scores,
    run_probe,
    select_keep_set,
)


def _tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    return Qwen2ForCausalLM(config).eval()


def test_repeated_batch_shape_and_repetition() -> None:
    ids = build_repeated_batch(256, batch=3, half_len=16, seed=0, device=torch.device("cpu"))
    assert ids.shape == (3, 32)
    assert torch.equal(ids[:, :16], ids[:, 16:])


def test_induction_scores_shape_and_range() -> None:
    model = _tiny_model()
    ids = build_repeated_batch(256, batch=2, half_len=16, seed=1, device=torch.device("cpu"))
    scores = induction_scores(model, ids, half_len=16)
    assert scores.shape == (6, 4)
    assert torch.all(scores >= 0) and torch.all(scores <= 1)


def test_select_keep_set_spacing_and_fallback() -> None:
    scores = torch.tensor([0.9, 0.8, 0.7, 0.1, 0.6, 0.05])
    chosen = select_keep_set(scores, keep=3, min_spacing=2)
    assert chosen == sorted(chosen) and len(chosen) == 3
    assert all(b - a >= 2 for a, b in zip(chosen, chosen[1:], strict=False))
    # quota larger than spacing allows -> fallback fills the rest
    assert len(select_keep_set(scores, keep=5, min_spacing=3)) == 5


def test_run_probe_end_to_end() -> None:
    model = _tiny_model()
    result = run_probe(model, vocab_size=256, keep=2, batch=2, half_len=16, seeds=(0,))
    assert len(result["keep_softmax_layers"]) == 2
    assert len(result["convert_to_mamba3"]) == 4
    assert set(result["keep_softmax_layers"]).isdisjoint(result["convert_to_mamba3"])
    assert len(result["layer_scores"]) == 6
