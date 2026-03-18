import shutil
from pathlib import Path

from ava.config import ExperimentConfig
from ava.experiments import (
    choose_next_step,
    estimate_budget,
    run_budget_sweep,
    run_memory_sweep,
    run_prompt_protocol_sweep,
    run_test_time_strategy_sweep,
)
from ava.tokenizer import ByteTokenizer
from ava.tokenizer_artifacts import build_hf_bpe_artifact


def test_budget_sweep_returns_sorted_results() -> None:
    config_paths = sorted(Path("configs/experiments").glob("*.yaml"))
    results = run_budget_sweep(config_paths)
    assert len(results) == len(config_paths)
    assert results[0].parameters < results[-1].parameters


def test_prompt_protocol_sweep_has_clear_winner() -> None:
    results = run_prompt_protocol_sweep(ByteTokenizer())
    assert results
    assert results[0].average_tokens <= results[-1].average_tokens


def test_memory_sweep_returns_tradeoff() -> None:
    results = run_memory_sweep(ByteTokenizer())
    assert results
    assert results[0].recall_at_1 >= results[-1].recall_at_1


def test_test_time_strategy_prefers_targeted_compute() -> None:
    results = run_test_time_strategy_sweep()
    assert results[0].strategy == "hard_math_only"


def test_recommendation_contains_next_actions() -> None:
    budget_results = run_budget_sweep(sorted(Path("configs/experiments").glob("*.yaml")))
    protocol_results = run_prompt_protocol_sweep(ByteTokenizer())
    recommendation = choose_next_step(budget_results, protocol_results)
    assert recommendation["recommended_model"]
    assert recommendation["recommended_protocol"]
    assert recommendation["next_actions"]



def test_looped_budget_uses_effective_layers() -> None:
    base = ExperimentConfig.from_dict(
        {
            "name": "base",
            "tokenizer": {"kind": "byte", "vocab_size": 260},
            "model": {"block_size": 64, "n_layer": 2, "n_head": 2, "n_embd": 64, "dropout": 0.0, "bias": False},
            "training": {"device": "cpu", "dtype": "float32", "micro_batch_size": 1, "gradient_accumulation_steps": 1},
            "memory": {},
            "tools": {},
        }
    )
    looped = ExperimentConfig.from_dict(
        {
            "name": "looped",
            "tokenizer": {"kind": "byte", "vocab_size": 260},
            "model": {
                "block_size": 64,
                "n_layer": 2,
                "n_head": 2,
                "n_embd": 64,
                "dropout": 0.0,
                "bias": False,
                "architecture": "looped",
                "loop_repeats": 3,
            },
            "training": {"device": "cpu", "dtype": "float32", "micro_batch_size": 1, "gradient_accumulation_steps": 1},
            "memory": {},
            "tools": {},
        }
    )
    base_budget = estimate_budget(base)
    looped_budget = estimate_budget(looped)
    assert looped_budget.parameters > base_budget.parameters
    assert looped_budget.train_vram_gb > base_budget.train_vram_gb


def test_budget_uses_tokenizer_artifact_vocab_size() -> None:
    root = Path("sessions") / "test-budget-tokenizer-artifact"
    artifact_path = root / "hf_bpe.json"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "examples.jsonl").write_text(
        "\n".join(
            [
                '{"prompt":"What planet is known as the Red Planet?","response":"Mars"}',
                '{"prompt":"Use the calculator tool for 144 / 12.","response":"12"}',
            ]
        ),
        encoding="utf-8",
    )
    artifact = build_hf_bpe_artifact(root, artifact_path, target_vocab_size=96, min_frequency=1)
    config = ExperimentConfig.from_dict(
        {
            "name": "artifact-budget",
            "tokenizer": {"kind": "hf_bpe", "path": str(artifact_path), "vocab_size": 260},
            "model": {"n_layer": 2, "n_head": 2, "n_embd": 32, "block_size": 64},
        }
    )
    estimate = estimate_budget(config)
    assert estimate.parameters == config.model.estimated_parameters(int(artifact["vocab_size"]))
    shutil.rmtree(root)


def test_recurrent_depth_budget_trades_parameters_for_compute() -> None:
    base = ExperimentConfig.from_dict(
        {
            "name": "base",
            "tokenizer": {"kind": "byte", "vocab_size": 260},
            "model": {"block_size": 64, "n_layer": 8, "n_head": 2, "n_embd": 64, "dropout": 0.0, "bias": False},
            "training": {"device": "cpu", "dtype": "float32", "micro_batch_size": 1, "gradient_accumulation_steps": 1},
        }
    )
    recurrent = ExperimentConfig.from_dict(
        {
            "name": "recurrent",
            "tokenizer": {"kind": "byte", "vocab_size": 260},
            "model": {
                "block_size": 64,
                "n_layer": 2,
                "n_head": 2,
                "n_embd": 64,
                "dropout": 0.0,
                "bias": False,
                "architecture": "recurrent_depth",
                "loop_repeats": 4,
                "recurrent_prelude_layers": 1,
                "recurrent_coda_layers": 1,
            },
            "training": {"device": "cpu", "dtype": "float32", "micro_batch_size": 1, "gradient_accumulation_steps": 1},
        }
    )
    base_budget = estimate_budget(base)
    recurrent_budget = estimate_budget(recurrent)
    assert recurrent.model.effective_layers() > base.model.effective_layers()
    assert recurrent_budget.parameters < base_budget.parameters
    assert recurrent_budget.fits_4gb is True
