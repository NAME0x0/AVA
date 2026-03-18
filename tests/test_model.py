from ava.config import ModelConfig
from ava.model import build_model, torch


def test_recurrent_depth_model_runs_forward() -> None:
    config = ModelConfig(
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=False,
        architecture="recurrent_depth",
        loop_repeats=3,
        recurrent_prelude_layers=1,
        recurrent_coda_layers=1,
    )
    model = build_model(config, 260)
    idx = torch.randint(0, 260, (2, 16), dtype=torch.long)
    logits, loss = model(idx, idx)
    assert logits.shape == (2, 16, 260)
    assert loss is not None
    assert config.effective_layers() == 8
    assert model.loop_step_embeddings is not None
    assert torch.count_nonzero(model.loop_step_embeddings.weight).item() == 0


def test_rmsnorm_swiglu_forward() -> None:
    config = ModelConfig(
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=False,
        architecture="transformer",
        norm_type="rmsnorm",
        activation="swiglu",
    )
    model = build_model(config, 260)
    idx = torch.randint(0, 260, (2, 16), dtype=torch.long)
    logits, loss = model(idx, idx)
    assert logits.shape == (2, 16, 260)
    assert loss is not None


def test_rope_forward() -> None:
    config = ModelConfig(
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=False,
        architecture="transformer",
        position_encoding="rope",
    )
    model = build_model(config, 260)
    assert model.wpe is None
    assert model.rope_cos is not None
    assert model.rope_sin is not None
    idx = torch.randint(0, 260, (2, 32), dtype=torch.long)
    logits, loss = model(idx, idx)
    assert logits.shape == (2, 32, 260)
    assert loss is not None


def test_gated_recurrence_forward() -> None:
    config = ModelConfig(
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=False,
        architecture="recurrent_depth",
        loop_repeats=3,
        recurrent_prelude_layers=1,
        recurrent_coda_layers=1,
        recurrent_gate=True,
    )
    model = build_model(config, 260)
    assert model.recurrent_gate_proj is not None
    idx = torch.randint(0, 260, (2, 16), dtype=torch.long)
    logits, loss = model(idx, idx)
    assert logits.shape == (2, 16, 260)
    assert loss is not None


def test_full_modern_architecture() -> None:
    """Test the complete modern stack: RMSNorm + SwiGLU + RoPE + gated recurrence."""
    config = ModelConfig(
        block_size=64,
        n_layer=4,
        n_head=4,
        n_embd=32,
        dropout=0.0,
        bias=False,
        architecture="recurrent_depth",
        loop_repeats=4,
        recurrent_prelude_layers=1,
        recurrent_coda_layers=1,
        norm_type="rmsnorm",
        activation="swiglu",
        position_encoding="rope",
        recurrent_gate=True,
    )
    model = build_model(config, 260)
    assert model.wpe is None
    assert model.recurrent_gate_proj is not None
    idx = torch.randint(0, 260, (2, 32), dtype=torch.long)
    logits, loss = model(idx, idx)
    assert logits.shape == (2, 32, 260)
    assert loss is not None
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"


def test_modern_architecture_generates() -> None:
    config = ModelConfig(
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=False,
        architecture="recurrent_depth",
        loop_repeats=2,
        recurrent_prelude_layers=1,
        recurrent_coda_layers=1,
        norm_type="rmsnorm",
        activation="swiglu",
        position_encoding="rope",
        recurrent_gate=True,
    )
    model = build_model(config, 260)
    model.eval()
    idx = torch.tensor([[1]], dtype=torch.long)
    output = model.generate(idx, max_new_tokens=10, temperature=0.8)
    assert output.shape == (1, 11)


def test_ignore_index_minus_100() -> None:
    """Verify that -100 tokens are properly handled in loss."""
    config = ModelConfig(
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=16,
    )
    model = build_model(config, 260)
    torch.manual_seed(42)
    idx = torch.randint(0, 260, (1, 16), dtype=torch.long)
    targets = torch.randint(0, 260, (1, 16), dtype=torch.long)
    targets_partial = targets.clone()
    targets_partial[:, :8] = -100
    _, loss_full = model(idx, targets)
    _, loss_partial = model(idx, targets_partial)
    assert loss_full is not None and torch.isfinite(loss_full)
    assert loss_partial is not None and torch.isfinite(loss_partial)


def test_cosine_lr_schedule() -> None:
    from ava.config import ExperimentConfig, TrainingConfig, TokenizerConfig, ModelConfig as MC, MemoryConfig, ToolConfig
    config = ExperimentConfig(
        name="test-cosine",
        tokenizer=TokenizerConfig(),
        model=MC(block_size=32, n_layer=2, n_head=2, n_embd=16),
        training=TrainingConfig(
            learning_rate=1e-3,
            warmup_steps=10,
            max_steps=110,
            lr_schedule="cosine",
            min_lr_ratio=0.1,
        ),
        memory=MemoryConfig(),
        tools=ToolConfig(),
    )
    from ava.train import _lr_for_step
    lr_0 = _lr_for_step(config, 0)
    assert lr_0 < 1e-3
    lr_10 = _lr_for_step(config, 10)
    assert abs(lr_10 - 1e-3) < 1e-6
    lr_60 = _lr_for_step(config, 60)
    assert lr_60 < 1e-3
    lr_109 = _lr_for_step(config, 109)
    assert lr_109 < lr_60
    assert lr_109 >= 1e-3 * 0.1 - 1e-8
