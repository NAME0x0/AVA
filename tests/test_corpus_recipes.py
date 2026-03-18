import json
import shutil
import sys
import types
from pathlib import Path

import ava.corpus_recipes as corpus_recipes
from ava.corpus_recipes import (
    _gsm8k_to_text,
    _join_records,
    _mbpp_to_text,
    _openbookqa_to_text,
    _sciq_to_text,
    _tiny_stories_to_text,
    materialize_open_mix_corpus,
)


def test_join_records_uses_visible_separator() -> None:
    joined = _join_records(["alpha", "", "beta"])
    assert joined == "alpha\n\n<sep>\n\nbeta"


def test_row_formatters_keep_domain_structure() -> None:
    assert _tiny_stories_to_text([{"text": "A short story."}]) == ["A short story."]
    math_rows = _gsm8k_to_text([{"question": "2 + 2?", "answer": "4"}])
    science_rows = _sciq_to_text(
        [
            {
                "support": "Plants use sunlight.",
                "question": "What do plants use?",
                "correct_answer": "sunlight",
            }
        ]
    )
    mc_rows = _openbookqa_to_text(
        [
            {
                "question_stem": "Which force keeps planets in orbit?",
                "choices": {"label": ["A", "B"], "text": ["sound", "gravity"]},
                "answerKey": "B",
            }
        ]
    )
    code_rows = _mbpp_to_text(
        [{"text": "Return the sum.", "code": "def add(a, b):\n    return a + b"}]
    )

    assert math_rows[0].startswith("Math problem:\n2 + 2?")
    assert "Worked solution:\n4" in math_rows[0]
    assert science_rows[0].startswith("Science note:")
    assert science_rows[1].startswith("Science question:")
    assert "Correct answer:\ngravity" in mc_rows[0]
    assert code_rows[0].startswith("Programming task:")


def test_materialize_open_mix_corpus_writes_expected_files(monkeypatch) -> None:
    fake = types.ModuleType("datasets")

    def load_dataset(name: str, *args, split: str):
        if name == "roneneldan/TinyStories":
            return [{"text": "Story one."}, {"text": "Story two."}]
        if name == "gsm8k":
            return [{"question": "3 + 4?", "answer": "7"}]
        if name == "sciq":
            return [
                {
                    "support": "Water boils at 100C.",
                    "question": "When does water boil?",
                    "correct_answer": "100C",
                }
            ]
        if name == "allenai/openbookqa":
            return [
                {
                    "question_stem": "Which star is at the center of the solar system?",
                    "choices": {
                        "label": ["A", "B", "C", "D"],
                        "text": ["Moon", "Sun", "Mars", "Earth"],
                    },
                    "answerKey": "B",
                }
            ]
        if name == "mbpp":
            return [{"text": "Square a number.", "code": "def square(x):\n    return x * x"}]
        raise AssertionError(f"unexpected dataset: {name} {args} {split}")

    fake.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)

    root = Path("sessions") / "test-open-mix-corpus"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    manifest = materialize_open_mix_corpus(
        root, english_limit=2, math_limit=1, science_limit=2, code_limit=1
    )

    assert manifest["english_records"] == 2
    assert manifest["math_records"] == 1
    assert manifest["science_records"] == 3
    assert manifest["code_records"] == 1
    assert (root / "english.txt").read_text(encoding="utf-8").startswith("Story one.")
    assert "Math problem:" in (root / "math.txt").read_text(encoding="utf-8")
    assert "Science multiple choice:" in (root / "science.txt").read_text(encoding="utf-8")
    assert "Python solution:" in (root / "code.txt").read_text(encoding="utf-8")
    assert (root / "manifest.json").is_file()
    assert (root / "README.md").is_file()

    shutil.rmtree(root)


def test_materialize_posttrain_mix_corpus_samples_sources_deterministically(monkeypatch) -> None:
    root = Path("sessions") / "test-posttrain-mix"
    sources_root = Path("sessions") / "test-posttrain-mix-sources"
    for path in (root, sources_root):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    source_paths: dict[str, Path] = {}
    source_limits: dict[str, int] = {}
    source_repeats: dict[str, int] = {}
    for index, source_name in enumerate(corpus_recipes.DEFAULT_POSTTRAIN_SOURCE_LIMITS):
        source_dir = sources_root / source_name
        source_dir.mkdir(parents=True, exist_ok=True)
        source_path = source_dir / "examples.jsonl"
        rows = [
            {
                "prompt": f"{source_name} prompt {row}" + ("\u2028wrapped" if row == 0 else ""),
                "response": f"answer {row}",
                "kind": f"kind_{index}",
            }
            for row in range(4)
        ]
        source_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        source_paths[source_name] = source_path
        source_limits[source_name] = 2
        source_repeats[source_name] = 2 if index % 2 == 0 else 1

    monkeypatch.setattr(corpus_recipes, "POSTTRAIN_SOURCE_PATHS", source_paths)
    monkeypatch.setattr(corpus_recipes, "DEFAULT_POSTTRAIN_SOURCE_LIMITS", source_limits)

    manifest = corpus_recipes.materialize_posttrain_mix_corpus(
        root,
        source_limits=source_limits,
        source_repeats=source_repeats,
        seed_value=42,
    )

    expected_total = sum(2 * source_repeats[name] for name in source_limits)
    assert manifest["total_examples"] == expected_total
    assert manifest["source_counts"] == {name: 2 * source_repeats[name] for name in source_limits}
    examples = (root / "examples.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(examples) == manifest["total_examples"]
    assert all(json.loads(line) for line in examples)

    shutil.rmtree(root)
    shutil.rmtree(sources_root)
