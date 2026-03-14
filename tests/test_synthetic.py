from ava.synthetic import generate_tool_sft_examples


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
