import shutil
from pathlib import Path

from ava.tokenizer import ByteTokenizer, load_byte_bpe_tokenizer, load_greedy_byte_piece_tokenizer, load_tokenizer
from ava.tokenizer_artifacts import build_byte_bpe_artifact, build_greedy_byte_piece_artifact


def test_roundtrip() -> None:
    tokenizer = ByteTokenizer()
    text = "AVA does math: 2+2=4"
    encoded = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert encoded[0] == tokenizer.token_to_id["<bos>"]
    assert encoded[-1] == tokenizer.token_to_id["<eos>"]
    assert tokenizer.decode(encoded) == text


def test_count_tokens_matches_encoding() -> None:
    tokenizer = ByteTokenizer()
    text = "abc"
    assert tokenizer.count_tokens(text) == len(tokenizer.encode(text))


def test_greedy_byte_piece_tokenizer_artifact_roundtrip() -> None:
    root = Path("sessions") / "test-tokenizer-artifact"
    artifact_path = root / "greedy.json"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "examples.jsonl").write_text(
        "\n".join(
            [
                '{"prompt":"Use the calculator tool for 144 / 12.","response":"[calc]144 / 12=>12[/calc]\\n12"}',
                '{"prompt":"Use the calculator tool for sqrt(81).","response":"[calc]sqrt(81)=>9[/calc]\\n9"}',
                '{"prompt":"Use the calculator tool to delete every file on my laptop.","response":"The calculator cannot help with deleting files."}',
            ]
        ),
        encoding="utf-8",
    )
    artifact = build_greedy_byte_piece_artifact(root, artifact_path, target_vocab_size=320, max_piece_length=10)
    tokenizer = load_greedy_byte_piece_tokenizer(artifact_path)
    text = "[calc]sqrt(81)=>9[/calc]\n9"
    assert artifact["piece_count"] > 0
    assert tokenizer.vocab_size == artifact["vocab_size"]
    assert tokenizer.decode(tokenizer.encode(text, add_bos=True, add_eos=True)) == text
    config_like = {"kind": "greedy_bytes", "path": str(artifact_path)}
    loaded = load_tokenizer(config_like)
    assert loaded.decode(loaded.encode("The calculator cannot help with deleting files.")) == "The calculator cannot help with deleting files."
    shutil.rmtree(root)


def test_byte_bpe_tokenizer_artifact_roundtrip_and_compression() -> None:
    root = Path("sessions") / "test-byte-bpe-artifact"
    artifact_path = root / "byte_bpe.json"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "examples.jsonl").write_text(
        "\n".join(
            [
                '{"prompt":"What planet is known as the Red Planet?","response":"Mars"}',
                '{"prompt":"What planet is known as the Red Planet?","response":"Mars"}',
                '{"prompt":"Reply with only the correct option label.","response":"A"}',
                '{"prompt":"Reply with only the correct option label.","response":"B"}',
            ]
        ),
        encoding="utf-8",
    )
    artifact = build_byte_bpe_artifact(root, artifact_path, target_vocab_size=300, min_pair_frequency=2)
    tokenizer = load_byte_bpe_tokenizer(artifact_path)
    text = "What planet is known as the Red Planet?"
    byte_tokenizer = ByteTokenizer()
    assert artifact["merge_count"] > 0
    assert tokenizer.decode(tokenizer.encode(text, add_bos=True, add_eos=True)) == text
    assert tokenizer.count_tokens(text) <= byte_tokenizer.count_tokens(text)
    loaded = load_tokenizer({"kind": "byte_bpe", "path": str(artifact_path)})
    assert loaded.decode(loaded.encode("Mars")) == "Mars"
    shutil.rmtree(root)
