import shutil
from pathlib import Path

from ava.data import load_supervised_examples


def test_load_supervised_examples_preserves_teacher_metadata() -> None:
    root = Path("sessions") / "test-teacher-metadata"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "examples.jsonl").write_text(
        "\n".join(
            [
                '{"prompt":"Question one","response":"A","teacher_model":"codex","format_contract":"label_only","verifier_status":"pass","tags":["science_mc","science"]}',
                '{"prompt":"Question two","response":"19"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    examples = load_supervised_examples(root)
    assert examples[0]["teacher_model"] == "codex"
    assert examples[0]["format_contract"] == "label_only"
    assert examples[0]["verifier_status"] == "pass"
    assert examples[0]["tags"] == ["science_mc", "science"]
    assert examples[1]["prompt"] == "Question two"
    shutil.rmtree(root)
