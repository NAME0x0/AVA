import shutil
from pathlib import Path

from ava.tokenizer import ByteTokenizer, load_greedy_byte_piece_tokenizer, load_tokenizer
from ava.tokenizer_artifacts import build_greedy_byte_piece_artifact


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
