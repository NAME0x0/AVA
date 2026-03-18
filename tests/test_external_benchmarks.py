import json
import shutil
from pathlib import Path

from ava.config import ModelConfig
from ava.external_benchmarks import (
    ExternalBenchmarkTask,
    _build_sparse_support_index,
    _match_multiple_choice,
    _predict_multiple_choice_dense_hybrid_from_support,
    _predict_multiple_choice_dense_rerank_from_support,
    _predict_multiple_choice_from_support,
    _predict_multiple_choice_hybrid_from_support,
    _predict_multiple_choice_sparse_dense_router,
    _predict_multiple_choice_support_ensemble,
    _prepare_retrieval_prompt_for_task,
    extract_gsm8k_answer,
    format_arc_prompt,
)
from ava.model import build_model
from ava.retrieval import SupportExample, load_support_examples
from ava.tokenizer import ByteTokenizer


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
            "manifest": {
                "primary_bank": "primary",
                "margin_threshold": 0.001,
                "strategy": "margin_gate",
            },
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
        {
            "kind": "arc_mc",
            "category": "science",
            "prompt": "Which gas do plants take in from the air for photosynthesis?\n\nOptions:\nA. oxygen\nB. carbon dioxide\nC. helium\n\nReply with only the correct option label.",
            "response": "B",
        },
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


def test_predict_multiple_choice_dense_hybrid_from_support_uses_shortlist() -> None:
    class FakeDenseRetriever:
        model_name = "fake/dense"
        device = "cpu"

        def shortlist(
            self, prompt: str, *, top_k: int, category_hint: str | None, category_gated: bool
        ):
            assert "photosynthesis" in prompt
            assert top_k == 4
            return (
                [
                    SupportExample(
                        prompt="What gas do plants take in from the air for photosynthesis?",
                        response="carbon dioxide",
                        category="science",
                    )
                ],
                {
                    "candidate_count": 1,
                    "top_k": top_k,
                    "matches": [
                        {
                            "score": 0.99,
                            "prompt": "What gas do plants take in from the air for photosynthesis?",
                            "response": "carbon dioxide",
                            "category": "science",
                        }
                    ],
                },
            )

    label, scoring = _predict_multiple_choice_dense_hybrid_from_support(
        prompt="Which gas do plants take in for photosynthesis?",
        choices=(("A", "oxygen"), ("B", "carbon dioxide"), ("C", "helium")),
        dense_retriever=FakeDenseRetriever(),
        category_hint="science",
        category_gated=True,
        shortlist_top_k=4,
    )
    assert label == "B"
    assert scoring["mode"] == "dense_hybrid_support_mc"
    assert scoring["dense_model"] == "fake/dense"
    assert scoring["dense_shortlist"]


def test_predict_multiple_choice_dense_rerank_from_support_uses_reranked_shortlist() -> None:
    class FakeDenseRetriever:
        model_name = "fake/dense"
        device = "cpu"

        def shortlist(
            self, prompt: str, *, top_k: int, category_hint: str | None, category_gated: bool
        ):
            return (
                [
                    SupportExample(
                        prompt="Which gas is most common in the air people breathe?",
                        response="oxygen",
                        category="science",
                    ),
                    SupportExample(
                        prompt="What gas do plants take in from the air for photosynthesis?",
                        response="carbon dioxide",
                        category="science",
                    ),
                ],
                {
                    "candidate_count": 2,
                    "top_k": top_k,
                    "matches": [
                        {
                            "score": 0.8,
                            "prompt": "Which gas do plants take in from the air for photosynthesis?",
                            "response": "oxygen",
                            "category": "science",
                        },
                        {
                            "score": 0.7,
                            "prompt": "What gas do plants take in from the air for photosynthesis?",
                            "response": "carbon dioxide",
                            "category": "science",
                        },
                    ],
                },
            )

    class FakeSupportReranker:
        model_name = "fake/reranker"
        device = "cpu"

        def rerank(
            self,
            prompt: str,
            examples,
            *,
            top_k: int,
            category_hint: str | None,
            category_gated: bool,
        ):
            assert top_k == 1
            reordered = [examples[1]]
            return reordered, {
                "candidate_count": 2,
                "top_k": top_k,
                "matches": [
                    {
                        "score": 1.2,
                        "prompt": reordered[0].prompt,
                        "response": reordered[0].response,
                        "category": reordered[0].category,
                    },
                ],
            }

    label, scoring = _predict_multiple_choice_dense_rerank_from_support(
        prompt="Which gas do plants take in for photosynthesis?",
        choices=(("A", "oxygen"), ("B", "carbon dioxide"), ("C", "helium")),
        dense_retriever=FakeDenseRetriever(),
        support_reranker=FakeSupportReranker(),
        category_hint="science",
        category_gated=True,
        shortlist_top_k=4,
        rerank_top_k=1,
    )
    assert label == "B"
    assert scoring["mode"] == "dense_rerank_support_mc"
    assert scoring["reranker_model"] == "fake/reranker"
    assert scoring["rerank_matches"][0]["response"] == "carbon dioxide"


def test_prepare_retrieval_prompt_for_task_supports_prompt_mode() -> None:
    tokenizer = ByteTokenizer()
    model = build_model(
        ModelConfig(block_size=256, n_layer=1, n_head=2, n_embd=8, dropout=0.0, bias=False),
        tokenizer.vocab_size,
    )
    direct_completion, retrieval = _prepare_retrieval_prompt_for_task(
        model,
        tokenizer,
        prompt="Ava had 3 apples and bought 2 more. How many apples does Ava have now?",
        support_examples=[
            SupportExample(
                prompt="Mia had 4 pencils and bought 3 more. How many pencils does Mia have now?",
                response="7",
                category="math",
            )
        ],
        retrieval_mode="prompt_support",
        category_hint="math",
        category_gated=True,
        nearest_threshold=0.45,
        nearest_margin=0.0,
        support_top_k=1,
    )
    assert direct_completion is None
    assert retrieval["enabled"] is True
    assert retrieval["references"]
    assert retrieval["mode"] == "prompt_support"
    assert "Question:" in str(retrieval["prompt"])


def test_predict_multiple_choice_sparse_dense_router_prefers_dense_on_low_margin_sparse() -> None:
    class FakeDenseRetriever:
        model_name = "fake/dense"
        device = "cpu"

        def shortlist(
            self, prompt: str, *, top_k: int, category_hint: str | None, category_gated: bool
        ):
            return (
                [
                    SupportExample(
                        prompt="Which characteristic of a cheetah is more likely to be learned rather than inherited?",
                        response="hunting style",
                        category="science",
                    )
                ],
                {
                    "candidate_count": 1,
                    "top_k": top_k,
                    "matches": [
                        {
                            "score": 0.82,
                            "prompt": "Which characteristic of a cheetah is more likely to be learned rather than inherited?",
                            "response": "hunting style",
                            "category": "science",
                        }
                    ],
                },
            )

    support_examples = [
        SupportExample(
            prompt="Which characteristic of a cheetah is more likely to be inherited?",
            response="fur pattern",
            category="science",
        ),
        SupportExample(
            prompt="Which characteristic of a cheetah is more likely to be learned rather than inherited?",
            response="hunting style",
            category="science",
        ),
    ]
    label, scoring = _predict_multiple_choice_sparse_dense_router(
        prompt="Which characteristic of a cheetah is more likely to be learned rather than inherited?",
        choices=(("A", "fur pattern"), ("B", "hunting style"), ("C", "tail length")),
        support_resource={
            "kind": "banks",
            "manifest": {
                "primary_bank": "primary",
                "margin_threshold": 0.001,
                "strategy": "margin_gate",
            },
            "banks": {
                "primary": {
                    "path": "primary",
                    "examples": support_examples[:1],
                    "index": _build_sparse_support_index(support_examples[:1]),
                },
                "challenger": {
                    "path": "challenger",
                    "examples": support_examples[:1],
                    "index": _build_sparse_support_index(support_examples[:1]),
                },
            },
            "primary_examples": support_examples[:1],
        },
        dense_retriever=FakeDenseRetriever(),
        category_hint="science",
        category_gated=True,
        shortlist_top_k=4,
        dense_score_min=0.7,
        sparse_margin_max=2.0,
    )
    assert label == "B"
    assert scoring["mode"] == "sparse_dense_router_support_mc"
    assert scoring["router_decision"] == "dense"
    assert scoring["dense_top_score"] == 0.82
