from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ava.data import load_supervised_examples, load_text_corpus
from ava.tokenizer import SPECIAL_TOKENS


def _render_supervised_example(prompt: str, response: str) -> str:
    return f"Question: {prompt}\nAnswer: {response}"


def corpus_texts_for_tokenizer(corpus_root: str | Path) -> list[str]:
    root_path = Path(corpus_root)
    texts = load_text_corpus(root_path)
    for example in load_supervised_examples(root_path):
        texts.append(_render_supervised_example(example["prompt"], example["response"]))
    return texts


def _candidate_piece_counts(texts: list[str], max_piece_length: int) -> Counter[bytes]:
    counts: Counter[bytes] = Counter()
    for text in texts:
        payload = text.encode("utf-8")
        if len(payload) < 2:
            continue
        max_length = min(max_piece_length, len(payload))
        for length in range(2, max_length + 1):
            for index in range(0, len(payload) - length + 1):
                counts[payload[index : index + length]] += 1
    return counts


def build_greedy_byte_piece_artifact(
    corpus_root: str | Path,
    output_path: str | Path,
    *,
    target_vocab_size: int = 512,
    max_piece_length: int = 12,
    min_frequency: int = 2,
) -> dict[str, object]:
    piece_budget = target_vocab_size - len(SPECIAL_TOKENS) - 256
    if piece_budget <= 0:
        raise ValueError("target_vocab_size must exceed special tokens plus 256 byte fallback tokens")

    texts = corpus_texts_for_tokenizer(corpus_root)
    if not texts:
        raise RuntimeError(f"no texts available to build tokenizer under {corpus_root}")

    counts = _candidate_piece_counts(texts, max_piece_length)
    ranked = [
        (piece, count)
        for piece, count in counts.items()
        if count >= min_frequency and len(piece) >= 2
    ]
    ranked.sort(
        key=lambda item: (
            item[1] * (len(item[0]) - 1),
            item[1],
            len(item[0]),
            item[0],
        ),
        reverse=True,
    )

    selected: list[str] = []
    seen: set[bytes] = set()
    for piece, _count in ranked:
        if piece in seen:
            continue
        seen.add(piece)
        selected.append(piece.hex())
        if len(selected) >= piece_budget:
            break

    artifact = {
        "kind": "greedy_bytes",
        "pieces_hex": selected,
        "vocab_size": len(SPECIAL_TOKENS) + len(selected) + 256,
        "piece_count": len(selected),
        "byte_fallback": True,
        "special_tokens": list(SPECIAL_TOKENS),
        "corpus_root": str(Path(corpus_root)),
        "target_vocab_size": target_vocab_size,
        "max_piece_length": max_piece_length,
        "min_frequency": min_frequency,
        "text_count": len(texts),
        "character_count": sum(len(text) for text in texts),
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact
