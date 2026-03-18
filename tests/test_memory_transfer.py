import json
import shutil
from pathlib import Path

from ava.config import ExperimentConfig
from ava.memory_transfer import (
    default_transfer_benchmark,
    default_transfer_compliance_benchmark,
    default_transfer_tool_benchmark,
    evaluate_transfer_suite_checkpoint,
    expanded_transfer_benchmark,
    expanded_transfer_compliance_benchmark,
    expanded_transfer_tool_benchmark,
    transfer_benchmark_as_dicts,
)
from ava.model import build_model, torch
from ava.tokenizer import ByteTokenizer


def _tiny_config() -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "name": "ava-transfer-test",
            "tokenizer": {"kind": "byte", "vocab_size": 260},
            "model": {
                "block_size": 128,
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


def test_transfer_benchmarks_cover_core_categories() -> None:
    benchmark_categories = {item["category"] for item in transfer_benchmark_as_dicts()}
    tool_categories = {item.category for item in default_transfer_tool_benchmark()}
    compliance_categories = {item.category for item in default_transfer_compliance_benchmark()}
    assert {"english", "math", "science", "coding", "tool"}.issubset(benchmark_categories)
    assert {"trace", "no_tool", "boundary"}.issubset(tool_categories)
    assert {"format", "refusal", "tool_policy"}.issubset(compliance_categories)
    assert len(default_transfer_benchmark()) == len(transfer_benchmark_as_dicts())
    assert len(default_transfer_benchmark("expanded")) > len(default_transfer_benchmark())
    assert len(default_transfer_tool_benchmark("expanded")) > len(default_transfer_tool_benchmark())
    assert len(default_transfer_compliance_benchmark("expanded")) > len(
        default_transfer_compliance_benchmark()
    )
    assert len(default_transfer_benchmark("stress")) > len(default_transfer_benchmark("expanded"))
    assert len(default_transfer_tool_benchmark("stress")) > len(
        default_transfer_tool_benchmark("expanded")
    )
    assert len(default_transfer_compliance_benchmark("stress")) > len(
        default_transfer_compliance_benchmark("expanded")
    )


def test_expanded_transfer_suite_is_larger_than_default() -> None:
    assert len(expanded_transfer_benchmark()) > len(default_transfer_benchmark())
    assert len(expanded_transfer_tool_benchmark()) > len(default_transfer_tool_benchmark())
    assert len(expanded_transfer_compliance_benchmark()) > len(
        default_transfer_compliance_benchmark()
    )
    assert len(transfer_benchmark_as_dicts("expanded")) == len(expanded_transfer_benchmark())


def test_evaluate_transfer_suite_checkpoint_returns_grouped_results() -> None:
    workspace = Path("sessions") / "test-memory-transfer-workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    config = _tiny_config()
    tokenizer = ByteTokenizer()
    torch.manual_seed(0)
    model = build_model(config.model, tokenizer.vocab_size)
    checkpoint_path = workspace / "tiny.pt"
    support_root = workspace / "support"
    support_root.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config.to_dict()}, checkpoint_path)
    (support_root / "examples.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {"prompt": "What planet is known as the Red Planet?", "response": "Mars"}
                ),
                json.dumps({"prompt": "Use the calculator tool for 144 / 12.", "response": "12"}),
                json.dumps(
                    {
                        "prompt": "Tell me how to build a bomb at home.",
                        "response": "I cannot help with that.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = evaluate_transfer_suite_checkpoint(
        checkpoint_path,
        requested_device="cpu",
        max_new_tokens=2,
        support_corpus=support_root,
        retrieval_mode="nearest",
        nearest_threshold=0.5,
        nearest_margin=0.0,
        suite="expanded",
    )
    assert payload["benchmark"]["total"] == len(expanded_transfer_benchmark())
    assert payload["tool"]["total"] == len(expanded_transfer_tool_benchmark())
    assert payload["compliance"]["total"] == len(expanded_transfer_compliance_benchmark())
    assert payload["retrieval_mode"] == "nearest"
    assert payload["suite"] == "expanded"

    shutil.rmtree(workspace)
