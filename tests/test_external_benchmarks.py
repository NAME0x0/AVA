from ava.external_benchmarks import (
    ExternalBenchmarkTask,
    _match_multiple_choice,
    extract_gsm8k_answer,
    format_arc_prompt,
)


def test_extract_gsm8k_answer_prefers_delimiter() -> None:
    text = "Work: 3 + 4 = 7\n#### 7"
    assert extract_gsm8k_answer(text) == "7"


def test_extract_gsm8k_answer_falls_back_to_last_number() -> None:
    text = "The final value is 1,234 after subtraction."
    assert extract_gsm8k_answer(text) == "1234"


def test_format_arc_prompt_lists_labeled_options() -> None:
    prompt = format_arc_prompt("Which option is correct?", (("A", "left"), ("B", "right")))
    assert "Options:" in prompt
    assert "A. left" in prompt
    assert "B. right" in prompt


def test_match_multiple_choice_accepts_label_and_option_text() -> None:
    task = ExternalBenchmarkTask(
        benchmark="arc-challenge",
        task_id="1",
        category="science",
        prompt="prompt",
        metric="multiple_choice",
        expected="D",
        expected_text="Record the details of the investigation.",
        choices=(("A", "One"), ("D", "Record the details of the investigation.")),
    )
    assert _match_multiple_choice(task, "D")
    assert _match_multiple_choice(task, "Record the details of the investigation.")
    assert not _match_multiple_choice(task, "A")
