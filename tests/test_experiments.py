from pathlib import Path

from ava.experiments import (
    choose_next_step,
    run_budget_sweep,
    run_memory_sweep,
    run_prompt_protocol_sweep,
    run_test_time_strategy_sweep,
)
from ava.tokenizer import ByteTokenizer


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

