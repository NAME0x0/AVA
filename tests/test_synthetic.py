from ava.synthetic import (
    generate_math_reasoning_support_examples,
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


def test_generate_public_reasoning_patch_examples_balances_labels_and_preserves_tool_anchors() -> None:
    from ava.synthetic import generate_public_reasoning_patch_examples

    examples = generate_public_reasoning_patch_examples()
    choices = [item["response"] for item in examples if item["kind"] in {"science_mc", "commonsense_mc"}]
    assert set(choices) == {"A", "B", "C", "D"}
    math_rows = [item for item in examples if item["kind"] == "math_word"]
    assert len(math_rows) >= 20
    assert all(item["prompt"].endswith("Reply with only the final answer.") for item in math_rows)
    prompts = {item["prompt"] for item in examples}
    assert "Use the calculator tool for 144 / 12." in prompts


def test_generate_teacher_distill_examples_adds_sop_metadata() -> None:
    from ava.synthetic import generate_teacher_distill_examples

    examples = generate_teacher_distill_examples(teacher_model="codex")
    sample = examples[0]
    assert sample["teacher_model"] == "codex"
    assert sample["source_type"] == "synthetic_teacher"
    assert sample["format_contract"]
    assert sample["verifier_status"] == "pass"
    assert isinstance(sample["tags"], list)


def test_generate_public_benchmark_distill_examples_uses_train_splits_and_anchors() -> None:
    from ava.synthetic import generate_public_benchmark_distill_examples

    examples = generate_public_benchmark_distill_examples(arc_limit=3, gsm8k_limit=4, arc_rotations=1, anchor_repeats=2)
    arc_rows = [item for item in examples if item["kind"] in {"arc_mc", "arc_mc_aug"}]
    gsm_rows = [item for item in examples if item["kind"] == "gsm8k_train"]
    anchor_rows = [item for item in examples if item["source_type"] == "anchor"]
    assert len(arc_rows) == 6
    assert len(gsm_rows) == 4
    assert len(anchor_rows) > 10
    assert {item["response"] for item in arc_rows}.issubset({"A", "B", "C", "D"})
    assert len({item["prompt"] for item in arc_rows}) == len(arc_rows)
    assert all(item["prompt"].endswith("Reply with only the final answer.") for item in gsm_rows)



def test_generate_math_reasoning_support_examples_adds_short_rationales() -> None:
    examples = generate_math_reasoning_support_examples()
    assert examples
    assert all(item["category"] == "math" for item in examples)
    assert all(item["teacher_rationale_short"] for item in examples)
    assert all(item["format_contract"] == "final_answer_only" for item in examples)


def test_public_science_support_examples_from_rows_preserve_labels_and_rotations() -> None:
    from ava.synthetic import _openbookqa_train_examples_from_rows, _sciq_train_examples_from_rows

    sciq_examples = _sciq_train_examples_from_rows([
        {
            "question": "What gas do plants take in from the air for photosynthesis?",
            "correct_answer": "carbon dioxide",
            "distractor1": "oxygen",
            "distractor2": "helium",
            "distractor3": "nitrogen",
        }
    ], rotations=1)
    openbookqa_examples = _openbookqa_train_examples_from_rows([
        {
            "question_stem": "Which force keeps planets in orbit around the Sun?",
            "choices": {"label": ["A", "B", "C", "D"], "text": ["magnetism", "gravity", "friction", "sound"]},
            "answerKey": "B",
        }
    ], rotations=1)

    assert {item["response"] for item in sciq_examples}.issubset({"A", "B", "C", "D"})
    assert {item["response"] for item in openbookqa_examples}.issubset({"A", "B", "C", "D"})
    assert len(sciq_examples) == 2
    assert len(openbookqa_examples) == 2
    assert all(item["category"] == "science" for item in sciq_examples + openbookqa_examples)
