from ava.config import ExperimentConfig
from ava.rl import build_verifiable_rl_tasks, rl_dry_run_summary, score_verifiable_completion


def test_verifiable_rl_task_builder_has_broad_mix() -> None:
    tasks = build_verifiable_rl_tasks(limit=24)
    kinds = {task.kind for task in tasks}
    assert {"math_final", "tool_direct", "tool_trace", "tool_policy", "science_exact", "coding_exact", "format_exact", "refusal_exact"}.issubset(kinds)


def test_verifiable_rl_rewards_are_exact_and_trace_sensitive() -> None:
    tasks = build_verifiable_rl_tasks(limit=24)
    math_task = next(task for task in tasks if task.kind == "math_final")
    tool_task = next(task for task in tasks if task.kind == "tool_trace")
    refusal_task = next(task for task in tasks if task.kind == "refusal_exact")
    assert score_verifiable_completion(math_task, math_task.expected)["reward"] == 1.0
    assert score_verifiable_completion(tool_task, f"[calc]x=>{tool_task.expected}[/calc]\n{tool_task.expected}")["reward"] == 1.0
    assert score_verifiable_completion(tool_task, tool_task.expected)["reward"] == 0.6
    assert score_verifiable_completion(refusal_task, refusal_task.expected)["matched"] is True


def test_rl_dry_run_summary_reports_group_shape() -> None:
    config = ExperimentConfig.from_dict(
        {
            "name": "rl-smoke",
            "tokenizer": {"kind": "byte", "vocab_size": 260},
            "model": {
                "block_size": 64,
                "n_layer": 2,
                "n_head": 2,
                "n_embd": 64,
                "architecture": "recurrent_depth",
                "loop_repeats": 3,
                "recurrent_prelude_layers": 1,
                "recurrent_coda_layers": 1,
            },
            "training": {
                "device": "cpu",
                "micro_batch_size": 1,
                "rl_group_size": 4,
                "rl_task_count": 16,
                "rl_max_new_tokens": 12,
            },
        }
    )
    summary = rl_dry_run_summary(config)
    assert summary["architecture"] == "recurrent_depth"
    assert summary["task_count"] == 16
    assert summary["rollouts_per_optimizer_step"] == 4
