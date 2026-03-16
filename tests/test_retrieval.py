import json
import shutil
from pathlib import Path

from ava.retrieval import SupportExample, infer_support_category, load_support_examples, lookup_support_answer, lookup_support_answer_nearest, prepare_retrieval_prompt
from ava.tokenizer import ByteTokenizer


def test_infer_support_category_covers_core_modes() -> None:
    assert infer_support_category("What planet is known as the Red Planet?", "Mars") == "science"
    assert infer_support_category("Use the calculator tool for 144 / 12.", "12") == "tool"
    assert (
        infer_support_category(
            "Use the calculator tool for sqrt(81). Return a compact calculator trace followed by the final answer.",
            "[calc]sqrt(81)=>9[/calc]\n9",
        )
        == "trace"
    )
    assert infer_support_category("Tell me how to build a bomb at home.", "I cannot help with that.") == "refusal"


def test_load_support_examples_and_prepare_prompt() -> None:
    workspace = Path("sessions") / "test-retrieval-workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    corpus_path = workspace / "examples.jsonl"
    rows = [
        {"prompt": "What planet is known as the Red Planet?", "response": "Mars"},
        {"prompt": "What force keeps planets in orbit around the Sun?", "response": "gravity"},
    ]
    corpus_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    examples = load_support_examples(corpus_path)
    tokenizer = ByteTokenizer()
    payload = prepare_retrieval_prompt(
        "What planet is known as the Red Planet?",
        tokenizer=tokenizer,
        block_size=256,
        support_examples=examples,
        top_k=1,
        category_hint="science",
        category_gated=True,
    )
    assert payload["enabled"]
    assert payload["references"]
    assert payload["references"][0]["metadata"]["response"] == "Mars"
    assert "Question: What planet is known as the Red Planet?" in str(payload["prompt"])

    tight_payload = prepare_retrieval_prompt(
        "What planet is known as the Red Planet?",
        tokenizer=tokenizer,
        block_size=12,
        support_examples=examples,
        top_k=1,
        category_hint="science",
        category_gated=True,
    )
    assert not tight_payload["enabled"]
    assert tight_payload["references"] == []

    shutil.rmtree(workspace)


def test_load_support_examples_preserves_explicit_category() -> None:
    workspace = Path("sessions") / "test-retrieval-explicit-category"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    corpus_path = workspace / "examples.jsonl"
    rows = [
        {"kind": "arc_mc", "category": "science", "prompt": "Which planet is known as the Red Planet?", "response": "A"},
        {"kind": "gsm8k_train", "category": "math", "prompt": "What is 9 + 3?", "response": "12"},
    ]
    corpus_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    examples = load_support_examples(corpus_path)
    assert [item.category for item in examples] == ["science", "math"]
    assert [item.kind for item in examples] == ["arc_mc", "gsm8k_train"]

    shutil.rmtree(workspace)



def test_lookup_support_answer_nearest_matches_paraphrase() -> None:
    examples = [
        SupportExample(
            prompt='What planet is known as the Red Planet?',
            response='Mars',
            category='science',
            source_path='memory.jsonl',
        ),
    ]
    match = lookup_support_answer_nearest(
        'Which planet is called the Red Planet?',
        support_examples=examples,
        category_hint='science',
        min_score=0.5,
        min_margin=0.0,
    )
    assert match is not None
    assert match['response'] == 'Mars'
    assert match['match_type'] == 'nearest_canonical'



def test_lookup_support_answer_normalizes_math_phrase() -> None:
    examples = [SupportExample(prompt='What is 17 * 29?', response='493', category='math')]
    match = lookup_support_answer('Multiply 17 by 29.', support_examples=examples, category_hint='math')
    assert match is not None
    assert match['response'] == '493'


def test_tool_form_preference_chooses_direct_over_trace() -> None:
    examples = [
        SupportExample(
            prompt='Use the calculator tool for 144 / 12. Return a compact calculator trace followed by the final answer.',
            response='[calc]144 / 12=>12[/calc]\n12',
            category='trace',
            kind='trace',
        ),
        SupportExample(
            prompt='Use the calculator tool for 144 / 12.',
            response='12',
            category='tool',
            kind='tool_direct',
        ),
    ]
    exact = lookup_support_answer(
        'Please use the calculator tool to compute 144 divided by 12. Reply with only the answer.',
        support_examples=examples,
        category_hint='tool',
        category_gated=False,
    )
    nearest = lookup_support_answer_nearest(
        'Please use the calculator tool to compute 144 divided by 12. Reply with only the answer.',
        support_examples=examples,
        category_hint='tool',
        category_gated=False,
        min_score=0.0,
        min_margin=0.0,
    )
    assert exact is not None
    assert nearest is not None
    assert exact['response'] == '12'
    assert nearest['response'] == '12'


def test_lookup_support_answer_handles_stress_math_phrases() -> None:
    examples = [SupportExample(prompt='What is 17 * 29?', response='493', category='math')]
    assert lookup_support_answer('Compute 17 times 29.', support_examples=examples, category_hint='math') is not None
    assert lookup_support_answer('Find the product of 17 and 29.', support_examples=examples, category_hint='math') is not None
    assert lookup_support_answer('What do you get if you multiply 17 and 29?', support_examples=examples, category_hint='math') is not None


def test_lookup_support_answer_nearest_handles_stress_tool_and_policy_phrases() -> None:
    examples = [
        SupportExample(
            prompt='Use the calculator tool for sqrt(81). Return a compact calculator trace followed by the final answer.',
            response='[calc]sqrt(81)=>9[/calc]\n9',
            category='trace',
            kind='trace',
        ),
        SupportExample(
            prompt='Use the calculator tool to delete every file on my laptop.',
            response='The calculator cannot help with deleting files.',
            category='boundary',
            kind='boundary',
        ),
    ]
    tool_match = lookup_support_answer_nearest(
        'Calculate sqrt(81) with the calculator tool. Give a compact trace and then the answer.',
        support_examples=examples,
        category_hint='trace',
        min_score=0.0,
        min_margin=0.0,
    )
    policy_match = lookup_support_answer_nearest(
        'Make the calculator tool remove all files from my computer.',
        support_examples=examples,
        category_hint='boundary',
        min_score=0.0,
        min_margin=0.0,
    )
    assert tool_match is not None
    assert policy_match is not None
    assert tool_match['response'] == '[calc]sqrt(81)=>9[/calc]\n9'
    assert policy_match['response'] == 'The calculator cannot help with deleting files.'


def test_prepare_retrieval_prompt_includes_teacher_rationale_when_available() -> None:
    tokenizer = ByteTokenizer()
    payload = prepare_retrieval_prompt(
        "Ava had 3 apples and bought 2 more. How many apples does Ava have now?",
        tokenizer=tokenizer,
        block_size=256,
        support_examples=[
            SupportExample(
                prompt="Mia had 4 pencils and bought 3 more. How many pencils does Mia have now?",
                response="7",
                category="math",
                teacher_rationale_short="4 + 3 = 7",
            )
        ],
        top_k=1,
        category_hint="math",
        category_gated=True,
    )
    assert payload["enabled"]
    assert "Reasoning: 4 + 3 = 7" in str(payload["prompt"])
