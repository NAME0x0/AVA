import json
import shutil
from pathlib import Path

from ava.external_benchmarks import (
    ExternalBenchmarkTask,
    _build_sparse_support_index,
    _match_multiple_choice,
    _predict_multiple_choice_from_support,
    _predict_multiple_choice_hybrid_from_support,
    _predict_multiple_choice_support_ensemble,
    extract_gsm8k_answer,
    format_arc_prompt,
)
from ava.retrieval import SupportExample, load_support_examples


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


def test_predict_multiple_choice_from_support_maps_answer_text_to_label() -> None:
    label, scoring = _predict_multiple_choice_from_support(
        prompt="Which gas do plants take in for photosynthesis?",
        choices=(("A", "oxygen"), ("B", "carbon dioxide"), ("C", "helium")),
        support_examples=[
            SupportExample(
                prompt="What gas do plants take in from the air for photosynthesis?",
                response="carbon dioxide",
                category="science",
            )
        ],
        category_hint="science",
        category_gated=True,
        top_k=1,
    )
    assert label == "B"
    assert scoring["mode"] == "support_mc"
    assert scoring["selected"] == "B"


def test_predict_multiple_choice_from_support_maps_rotated_label_to_current_choice_text() -> None:
    label, scoring = _predict_multiple_choice_from_support(
        prompt="Which gas do plants take in for photosynthesis?",
        choices=(("A", "oxygen"), ("B", "carbon dioxide"), ("C", "helium")),
        support_examples=[
            SupportExample(
                prompt=(
                    "Which gas do plants take in from the air for photosynthesis?\n\n"
                    "Options:\n"
                    "A. carbon dioxide\n"
                    "B. oxygen\n"
                    "C. helium\n\n"
                    "Reply with only the correct option label."
                ),
                response="A",
                category="science",
            )
        ],
        category_hint="science",
        category_gated=True,
        top_k=1,
    )
    assert label == "B"
    assert scoring["selected"] == "B"


def test_predict_multiple_choice_hybrid_from_support_uses_support_prompt_tokens() -> None:
    label, scoring = _predict_multiple_choice_hybrid_from_support(
        prompt="The majority of the elements on the periodic table are what?",
        choices=(("A", "gases"), ("B", "metals"), ("C", "liquids"), ("D", "nonmetals")),
        support_examples=[
            SupportExample(
                prompt="All elements found on the left side of the periodic table are metals.",
                response="They are solids at room temperature.",
                category="science",
            ),
            SupportExample(
                prompt="Helium is an example of a noble gas.",
                response="It rarely reacts with other elements.",
                category="science",
            ),
        ],
        category_hint="science",
        category_gated=True,
        top_k=2,
    )
    assert label == "B"
    assert scoring["mode"] == "hybrid_support_mc"
    assert scoring["selected"] == "B"
    assert scoring["scores"]["B"] > scoring["scores"]["A"]


def test_predict_multiple_choice_support_ensemble_prefers_higher_margin_bank() -> None:
    primary_examples = [
        SupportExample(
            prompt="Helium is an example of a noble gas.",
            response="It rarely reacts with other elements.",
            category="science",
        )
    ]
    challenger_examples = [
        SupportExample(
            prompt="What gas do plants take in from the air for photosynthesis?",
            response="carbon dioxide",
            category="science",
        )
    ]
    label, scoring = _predict_multiple_choice_support_ensemble(
        prompt="Which gas do plants take in for photosynthesis?",
        choices=(("A", "oxygen"), ("B", "carbon dioxide"), ("C", "helium")),
        support_resource={
            "kind": "banks",
            "manifest": {"primary_bank": "primary", "margin_threshold": 0.001, "strategy": "margin_gate"},
            "banks": {
                "primary": {
                    "path": "primary",
                    "examples": primary_examples,
                    "index": _build_sparse_support_index(primary_examples),
                },
                "challenger": {
                    "path": "challenger",
                    "examples": challenger_examples,
                    "index": _build_sparse_support_index(challenger_examples),
                },
            },
            "primary_examples": primary_examples,
        },
        category_hint="science",
        category_gated=True,
        top_k=2,
    )
    assert label == "B"
    assert scoring["selected_bank"] == "challenger"
    assert scoring["mode"] == "hybrid_support_ensemble"


def test_predict_multiple_choice_from_support_uses_explicit_categories_from_corpus() -> None:
    workspace = Path("sessions") / "test-external-support-index"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    corpus_path = workspace / "examples.jsonl"
    rows = [
        {"kind": "arc_mc", "category": "science", "prompt": "Which gas do plants take in from the air for photosynthesis?\n\nOptions:\nA. oxygen\nB. carbon dioxide\nC. helium\n\nReply with only the correct option label.", "response": "B"},
        {"kind": "gsm8k_train", "category": "math", "prompt": "What is 9 + 3?", "response": "12"},
    ]
    corpus_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    examples = load_support_examples(corpus_path)
    support_index = _build_sparse_support_index(examples)
    label, scoring = _predict_multiple_choice_from_support(
        prompt="Which gas do plants take in for photosynthesis?",
        choices=(("A", "oxygen"), ("B", "carbon dioxide"), ("C", "helium")),
        support_examples=examples,
        support_index=support_index,
        category_hint="science",
        category_gated=True,
        top_k=1,
    )
    assert label == "B"
    assert scoring["candidate_count"] == 1

    shutil.rmtree(workspace)

