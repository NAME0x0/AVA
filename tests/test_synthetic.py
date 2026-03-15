from ava.synthetic import (
    generate_tool_repair_examples,
    generate_tool_sft_examples,
    repair_expression_pool,
)


def test_generate_tool_sft_examples_is_deterministic_and_balanced() -> None:
    first = generate_tool_sft_examples()
    second = generate_tool_sft_examples()
    assert first == second
    assert len(first) >= 100
    kinds = {item["kind"] for item in first}
    assert {"trace", "tool_direct", "no_tool", "boundary"}.issubset(kinds)
    trace_rows = [item for item in first if item["kind"] == "trace"]
    assert any(item["response"].startswith("[calc]") and "\n" in item["response"] for item in trace_rows)


def test_generate_tool_sft_examples_large_scale_expands_trace_curriculum() -> None:
    base = generate_tool_sft_examples(scale="base")
    large = generate_tool_sft_examples(scale="large")
    assert len(large) > len(base)
    large_trace_rows = [item for item in large if item["kind"] == "trace"]
    assert len(large_trace_rows) > 500


def test_generate_tool_repair_examples_targets_direct_trace_and_refusal_modes() -> None:
    examples = generate_tool_repair_examples()
    kinds = {item["kind"] for item in examples}
    assert {"trace", "tool_direct", "no_tool", "boundary", "refusal"}.issubset(kinds)
    prompts = {item["prompt"] for item in examples}
    assert "Use the calculator tool for 144 / 12." in prompts
    assert "Use the calculator tool for 144 / 12. Reply with only the answer." in prompts
    assert "Use the calculator tool for sqrt(81). Return a compact calculator trace followed by the final answer." in prompts
    assert any(item["response"] == "I cannot help with that." for item in examples if item["kind"] == "refusal")


def test_repair_expression_pool_is_deterministic_and_sqrt_heavy() -> None:
    expressions = repair_expression_pool()
    assert expressions == repair_expression_pool()
    assert "sqrt(81)" in expressions
    assert len([item for item in expressions if item.startswith("sqrt(")]) >= 8


def test_generate_tool_canonical_patch_examples_is_canonical_and_targeted() -> None:
    from ava.synthetic import generate_tool_canonical_patch_examples

    examples = generate_tool_canonical_patch_examples()
    prompts = {item["prompt"] for item in examples}
    assert "Use the calculator tool for 144 / 12." in prompts
    assert "Use the calculator tool for sqrt(81). Return a compact calculator trace followed by the final answer." in prompts
    assert "Calculate 144 / 12 with the calculator tool. Reply with only the answer." not in prompts
