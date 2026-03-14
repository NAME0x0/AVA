import shutil
from pathlib import Path

from ava.config import ExperimentConfig
from ava.inspect import _forward_with_trace, trace_checkpoint
from ava.model import build_model, torch
from ava.tokenizer import ByteTokenizer


def _tiny_config() -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "name": "ava-inspect-tiny",
            "tokenizer": {"kind": "byte", "vocab_size": 260},
            "model": {
                "block_size": 64,
                "n_layer": 2,
                "n_head": 2,
                "n_embd": 64,
                "dropout": 0.0,
                "bias": False,
            },
            "training": {"device": "cpu", "dtype": "float32"},
            "memory": {},
            "tools": {},
        }
    )


def test_forward_trace_matches_model_logits() -> None:
    config = _tiny_config()
    tokenizer = ByteTokenizer()
    torch.manual_seed(0)
    model = build_model(config.model, tokenizer.vocab_size)
    prompt = "Question: What is 2 + 2?\nAnswer: "
    idx = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long)
    logits, _ = model(idx)
    traced_logits, trace = _forward_with_trace(
        model,
        idx,
        tokenizer,
        top_k_neurons=4,
        top_k_logits_count=4,
        top_k_attention=2,
    )
    assert torch.allclose(logits, traced_logits, atol=1e-5)
    assert len(trace["layers"]) == config.model.n_layer
    assert trace["top_next_token_logits"]


def test_trace_checkpoint_returns_step_and_layer_data() -> None:
    workspace = Path("sessions") / "test-inspect-workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    config = _tiny_config()
    tokenizer = ByteTokenizer()
    torch.manual_seed(0)
    model = build_model(config.model, tokenizer.vocab_size)
    checkpoint_path = workspace / "tiny.pt"
    torch.save({"model": model.state_dict(), "config": config.to_dict()}, checkpoint_path)

    trace = trace_checkpoint(
        checkpoint_path,
        "Question: What is 2 + 2?\nAnswer: ",
        requested_device="cpu",
        max_new_tokens=2,
        top_k_neurons=4,
        top_k_logits=4,
        top_k_attention=2,
    )
    assert trace["config_name"] == "ava-inspect-tiny"
    assert trace["steps"]
    assert len(trace["steps"][0]["layers"]) == config.model.n_layer
    assert trace["steps"][0]["layers"][0]["top_mlp_neurons"]
    assert trace["steps"][0]["layers"][0]["attention"]

    shutil.rmtree(workspace)

