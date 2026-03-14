import shutil
from pathlib import Path

from ava.config import load_experiment_config
from ava.train import _load_supervised_dataset, dry_run_summary, summarize_corpus


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
    samples, total_tokens = _load_supervised_dataset(corpus_root, block_size=40)
    assert total_tokens > 0
    assert len(samples) == 1
    assert any(token_id != -100 for token_id in samples[0]["target_ids"])
    shutil.rmtree(corpus_root)

