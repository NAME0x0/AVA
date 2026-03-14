import shutil
from pathlib import Path

from ava.config import load_experiment_config
from ava.tokenizer import load_tokenizer
from ava.train import (
    _load_supervised_dataset,
    _split_supervised_samples,
    dry_run_summary,
    summarize_corpus,
    summarize_supervised_dataset,
)


def test_dry_run_summary_matches_config() -> None:
    config = load_experiment_config("configs/base.yaml")
    summary = dry_run_summary(config)
    assert summary["name"] == "ava-59m-byte"
    assert summary["parameters"] > 0


def test_summarize_corpus_reports_files_and_tokens() -> None:
    summary = summarize_corpus(Path("corpora") / "smoke")
    assert summary["file_count"] >= 3
    assert summary["text_count"] >= 3
    assert summary["token_count"] > 0


def test_supervised_loader_skips_samples_with_no_target_tokens() -> None:
    tokenizer = load_tokenizer()
    corpus_root = Path("sessions") / "test-supervised-loader"
    if corpus_root.exists():
        shutil.rmtree(corpus_root)
    corpus_root.mkdir(parents=True, exist_ok=True)
    (corpus_root / "examples.jsonl").write_text(
        "\n".join(
            [
                '{"prompt":"This prompt is intentionally long long long long long long long long long long.","response":"ok"}',
                '{"prompt":"Short prompt","response":"good"}',
            ]
        ),
        encoding="utf-8",
    )
    samples, total_tokens = _load_supervised_dataset(corpus_root, block_size=40, tokenizer=tokenizer)
    assert total_tokens > 0
    assert len(samples) == 1
    assert any(token_id != -100 for token_id in samples[0]["target_ids"])
    shutil.rmtree(corpus_root)


def test_supervised_split_shuffles_deterministically() -> None:
    samples = [{"input_ids": [index], "target_ids": [index]} for index in range(10)]
    train_samples, val_samples = _split_supervised_samples(samples)
    assert len(train_samples) == 8
    assert len(val_samples) == 2
    combined = train_samples + val_samples
    assert {item["input_ids"][0] for item in combined} == set(range(10))
    assert combined != samples
    train_again, val_again = _split_supervised_samples(samples)
    assert train_samples == train_again
    assert val_samples == val_again


def test_summarize_supervised_dataset_reports_truncation() -> None:
    tokenizer = load_tokenizer()
    corpus_root = Path("sessions") / "test-supervised-summary"
    if corpus_root.exists():
        shutil.rmtree(corpus_root)
    corpus_root.mkdir(parents=True, exist_ok=True)
    (corpus_root / "examples.jsonl").write_text(
        "\n".join(
            [
                '{"prompt":"Use the calculator tool for factorial(5). Return a compact calculator trace followed by the final answer.","response":"[calc]factorial(5)=>120[/calc]\\n120"}',
                '{"prompt":"Short prompt","response":"good"}',
            ]
        ),
        encoding="utf-8",
    )
    stats = summarize_supervised_dataset(corpus_root, block_size=40, tokenizer=tokenizer)
    assert stats["total_examples"] == 2
    assert stats["truncated_examples"] == 1
    assert stats["kept_examples"] == 1
    assert stats["skipped_no_target_examples"] == 1
    shutil.rmtree(corpus_root)
