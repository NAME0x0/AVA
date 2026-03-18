import shutil
from pathlib import Path

import torch

from ava.config import ModelConfig, load_experiment_config
from ava.model import build_model
from ava.tokenizer import (
    ByteBPETokenizer,
    load_hf_subword_tokenizer,
    load_tokenizer,
    token_piece_bytes,
)
from ava.tokenizer_artifacts import build_hf_bpe_artifact
from ava.train import (
    _configure_trainable_parameters,
    _expand_state_dict_for_tokenizer,
    _load_init_checkpoint,
    _load_supervised_dataset,
    _resize_state_dict_for_block_size,
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
    samples, total_tokens = _load_supervised_dataset(
        corpus_root, block_size=40, tokenizer=tokenizer
    )
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


def test_expand_state_dict_for_byte_bpe_initializes_new_tokens_from_bytes() -> None:
    tokenizer = ByteBPETokenizer([{"left": "61", "right": "62", "merged": "6162"}])
    embedding = __import__("torch").arange(260 * 2, dtype=__import__("torch").float32).view(260, 2)
    state = {"wte.weight": embedding, "lm_head.weight": embedding.clone()}
    expanded = _expand_state_dict_for_tokenizer(state, tokenizer)
    new_row = expanded["wte.weight"][260]
    expected = (embedding[101] + embedding[102]) / 2
    assert expanded["wte.weight"].shape[0] == tokenizer.vocab_size
    assert __import__("torch").allclose(new_row, expected)
    assert __import__("torch").allclose(expanded["lm_head.weight"][260], expected)


def test_expand_state_dict_for_hf_bpe_rebuilds_non_aligned_vocab() -> None:
    root = Path("sessions") / "test-hf-bpe-remap"
    artifact_path = root / "hf_bpe.json"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "examples.jsonl").write_text(
        "\n".join(
            [
                '{"prompt":"What planet is known as the Red Planet?","response":"Mars"}',
                '{"prompt":"Use the calculator tool for 144 / 12.","response":"12"}',
            ]
        ),
        encoding="utf-8",
    )
    build_hf_bpe_artifact(root, artifact_path, target_vocab_size=96, min_frequency=1)
    tokenizer = load_hf_subword_tokenizer(artifact_path)
    embedding = torch.arange(260 * 2, dtype=torch.float32).view(260, 2)
    state = {"wte.weight": embedding, "lm_head.weight": embedding.clone()}
    expanded = _expand_state_dict_for_tokenizer(state, tokenizer)
    bos_id = tokenizer.token_to_id["<bos>"]
    assert torch.allclose(expanded["wte.weight"][bos_id], embedding.mean(dim=0))
    mars_token_id = next(
        token_id for token_id in tokenizer.encode("Mars") if token_piece_bytes(tokenizer, token_id)
    )
    piece = token_piece_bytes(tokenizer, mars_token_id)
    assert piece is not None
    expected = embedding[[4 + value for value in piece]].mean(dim=0)
    assert torch.allclose(expanded["wte.weight"][mars_token_id], expected)
    shutil.rmtree(root)


def test_resize_state_dict_for_block_size_interpolates_endpoints() -> None:
    torch = __import__("torch")
    positional = torch.arange(4 * 2, dtype=torch.float32).view(4, 2)
    state = {"wpe.weight": positional}
    resized = _resize_state_dict_for_block_size(state, block_size=8)
    assert resized["wpe.weight"].shape == (8, 2)
    assert torch.allclose(resized["wpe.weight"][0], positional[0])
    assert torch.allclose(resized["wpe.weight"][-1], positional[-1])


def test_configure_trainable_parameters_filters_by_pattern() -> None:
    model = build_model(
        ModelConfig(block_size=32, n_layer=1, n_head=2, n_embd=8, dropout=0.0, bias=False), 260
    )
    parameters, names, count = _configure_trainable_parameters(model, ("wpe.weight", "ln_f"))
    assert set(names) == {"wpe.weight", "ln_f.weight", "ln_f.bias"}
    assert count == sum(parameter.numel() for parameter in parameters)
    assert model.wpe.weight.requires_grad is True
    assert model.wte.weight.requires_grad is False


def test_load_init_checkpoint_allows_transformer_to_looped_warm_start() -> None:
    tokenizer = load_tokenizer()
    workspace = Path("sessions") / "test-loop-init"
    checkpoint_path = workspace / "base.pt"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    base_model = build_model(
        ModelConfig(block_size=32, n_layer=1, n_head=2, n_embd=8, dropout=0.0, bias=False),
        tokenizer.vocab_size,
    )
    torch = __import__("torch")
    torch.save(
        {
            "model": base_model.state_dict(),
            "config": {"tokenizer": {"kind": "byte"}},
        },
        checkpoint_path,
    )

    looped_model = build_model(
        ModelConfig(
            block_size=32,
            n_layer=1,
            n_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            architecture="looped",
            loop_repeats=3,
        ),
        tokenizer.vocab_size,
    )
    init_kind = _load_init_checkpoint(looped_model, tokenizer, checkpoint_path)
    assert init_kind == "byte"
    assert looped_model.loop_step_embeddings is not None
    assert torch.count_nonzero(looped_model.loop_step_embeddings.weight).item() == 0
    shutil.rmtree(workspace)


def test_expand_state_dict_for_hf_tokenizer_neutralizes_special_tokens() -> None:
    root = Path("sessions") / "test-hf-special-neutralize"
    artifact_path = root / "hf_bpe.json"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "examples.jsonl").write_text(
        "\n".join(
            [
                '{"prompt":"What planet is known as the Red Planet?","response":"Mars"}',
                '{"prompt":"Use the calculator tool for 144 / 12.","response":"12"}',
            ]
        ),
        encoding="utf-8",
    )
    build_hf_bpe_artifact(root, artifact_path, target_vocab_size=96, min_frequency=1)
    tokenizer = load_hf_subword_tokenizer(artifact_path)
    state = {"wte.weight": torch.arange(260 * 4, dtype=torch.float32).reshape(260, 4)}
    expanded = _expand_state_dict_for_tokenizer(state, tokenizer)
    fallback = state["wte.weight"].mean(dim=0)
    for token_name in ("<pad>", "<bos>", "<eos>", "<sep>"):
        token_id = tokenizer.token_to_id[token_name]
        assert torch.allclose(expanded["wte.weight"][token_id], fallback)
    shutil.rmtree(root)


def test_load_init_checkpoint_allows_transformer_to_recurrent_depth_warm_start() -> None:
    tokenizer = load_tokenizer()
    workspace = Path("sessions") / "test-recurrent-depth-init"
    checkpoint_path = workspace / "base.pt"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    base_model = build_model(
        ModelConfig(block_size=32, n_layer=4, n_head=2, n_embd=8, dropout=0.0, bias=False),
        tokenizer.vocab_size,
    )
    torch = __import__("torch")
    torch.save(
        {
            "model": base_model.state_dict(),
            "config": {"tokenizer": {"kind": "byte"}, "model": {"architecture": "transformer"}},
        },
        checkpoint_path,
    )

    recurrent_model = build_model(
        ModelConfig(
            block_size=32,
            n_layer=2,
            n_head=2,
            n_embd=8,
            dropout=0.0,
            bias=False,
            architecture="recurrent_depth",
            loop_repeats=3,
            recurrent_prelude_layers=1,
            recurrent_coda_layers=1,
        ),
        tokenizer.vocab_size,
    )
    init_kind = _load_init_checkpoint(recurrent_model, tokenizer, checkpoint_path)
    assert init_kind == "byte"
    assert len(recurrent_model.prelude) == 1
    assert len(recurrent_model.coda) == 1
    assert recurrent_model.loop_step_embeddings is not None
    assert torch.count_nonzero(recurrent_model.loop_step_embeddings.weight).item() == 0
    shutil.rmtree(workspace)
