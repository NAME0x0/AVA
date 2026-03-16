from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ava.data import load_supervised_examples, load_text_corpus
from ava.env import huggingface_token, load_project_env
from ava.tokenizer import SPECIAL_TOKENS

try:
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers.decoders import Metaspace as HFMetaspaceDecoder
    from tokenizers.models import BPE as HFBPEModel
    from tokenizers.models import Unigram as HFUnigramModel
    from tokenizers.pre_tokenizers import Metaspace as HFMetaspace
    from tokenizers.trainers import BpeTrainer as HFBpeTrainer
    from tokenizers.trainers import UnigramTrainer as HFUnigramTrainer

    HF_TOKENIZERS_AVAILABLE = True
except ImportError:
    HFTokenizer = None
    HFMetaspaceDecoder = None
    HFBPEModel = None
    HFUnigramModel = None
    HFMetaspace = None
    HFBpeTrainer = None
    HFUnigramTrainer = None
    HF_TOKENIZERS_AVAILABLE = False

try:
    from transformers import AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    AutoTokenizer = None
    TRANSFORMERS_AVAILABLE = False


HF_EXTRA_SPECIAL_TOKENS = ("<unk>",)
HF_ALL_SPECIAL_TOKENS = list(SPECIAL_TOKENS) + list(HF_EXTRA_SPECIAL_TOKENS)


def _render_supervised_example(prompt: str, response: str) -> str:
    return f"Question: {prompt}\nAnswer: {response}"


def corpus_texts_for_tokenizer(corpus_root: str | Path) -> list[str]:
    root_path = Path(corpus_root)
    texts = load_text_corpus(root_path)
    for example in load_supervised_examples(root_path):
        texts.append(_render_supervised_example(str(example["prompt"]), str(example["response"])))
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


def _initial_bpe_sequences(texts: list[str]) -> list[list[bytes]]:
    sequences: list[list[bytes]] = []
    for text in texts:
        payload = text.encode("utf-8")
        if not payload:
            continue
        sequences.append([bytes([value]) for value in payload])
    return sequences


def _pair_counts(sequences: list[list[bytes]]) -> Counter[tuple[bytes, bytes]]:
    counts: Counter[tuple[bytes, bytes]] = Counter()
    for sequence in sequences:
        for index in range(len(sequence) - 1):
            counts[(sequence[index], sequence[index + 1])] += 1
    return counts


def _merge_pair_in_sequences(sequences: list[list[bytes]], pair: tuple[bytes, bytes]) -> list[list[bytes]]:
    merged_piece = pair[0] + pair[1]
    updated: list[list[bytes]] = []
    for sequence in sequences:
        rebuilt: list[bytes] = []
        index = 0
        while index < len(sequence):
            if index < len(sequence) - 1 and sequence[index] == pair[0] and sequence[index + 1] == pair[1]:
                rebuilt.append(merged_piece)
                index += 2
                continue
            rebuilt.append(sequence[index])
            index += 1
        updated.append(rebuilt)
    return updated


def build_byte_bpe_artifact(
    corpus_root: str | Path,
    output_path: str | Path,
    *,
    target_vocab_size: int = 512,
    min_pair_frequency: int = 2,
    max_piece_length: int = 12,
) -> dict[str, object]:
    merge_budget = target_vocab_size - len(SPECIAL_TOKENS) - 256
    if merge_budget <= 0:
        raise ValueError("target_vocab_size must exceed special tokens plus 256 base byte tokens")

    texts = corpus_texts_for_tokenizer(corpus_root)
    if not texts:
        raise RuntimeError(f"no texts available to build tokenizer under {corpus_root}")

    sequences = _initial_bpe_sequences(texts)
    merges: list[dict[str, object]] = []
    while len(merges) < merge_budget:
        counts = _pair_counts(sequences)
        candidates = [(pair, count) for pair, count in counts.items() if count >= min_pair_frequency]
        if not candidates:
            break
        candidates.sort(
            key=lambda item: (
                item[1],
                len(item[0][0]) + len(item[0][1]),
                item[0][0],
                item[0][1],
            ),
            reverse=True,
        )
        best_pair: tuple[bytes, bytes] | None = None
        best_count = 0
        merged = b""
        for candidate_pair, candidate_count in candidates:
            candidate_merged = candidate_pair[0] + candidate_pair[1]
            if len(candidate_merged) > max_piece_length:
                continue
            best_pair = candidate_pair
            best_count = candidate_count
            merged = candidate_merged
            break
        if best_pair is None:
            break
        merges.append(
            {
                "left": best_pair[0].hex(),
                "right": best_pair[1].hex(),
                "merged": merged.hex(),
                "count": best_count,
            }
        )
        sequences = _merge_pair_in_sequences(sequences, best_pair)

    artifact = {
        "kind": "byte_bpe",
        "merges": merges,
        "vocab_size": len(SPECIAL_TOKENS) + 256 + len(merges),
        "merge_count": len(merges),
        "special_tokens": list(SPECIAL_TOKENS),
        "base_bytes": 256,
        "corpus_root": str(Path(corpus_root)),
        "target_vocab_size": target_vocab_size,
        "min_pair_frequency": min_pair_frequency,
        "max_piece_length": max_piece_length,
        "text_count": len(texts),
        "character_count": sum(len(text) for text in texts),
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def _build_hf_tokenizer(kind: str, texts: list[str], target_vocab_size: int, min_frequency: int) -> dict[str, object]:
    if not HF_TOKENIZERS_AVAILABLE:
        raise RuntimeError("tokenizers is required to build hf_bpe or hf_unigram tokenizers")
    if kind == "hf_bpe":
        tokenizer = HFTokenizer(HFBPEModel(unk_token="<unk>"))
        trainer = HFBpeTrainer(
            vocab_size=target_vocab_size,
            min_frequency=min_frequency,
            special_tokens=HF_ALL_SPECIAL_TOKENS,
        )
    elif kind == "hf_unigram":
        tokenizer = HFTokenizer(HFUnigramModel())
        trainer = HFUnigramTrainer(
            vocab_size=target_vocab_size,
            special_tokens=HF_ALL_SPECIAL_TOKENS,
            unk_token="<unk>",
        )
    else:
        raise ValueError(f"unsupported HF tokenizer kind: {kind}")
    tokenizer.pre_tokenizer = HFMetaspace(replacement="▁", prepend_scheme="always")
    tokenizer.decoder = HFMetaspaceDecoder(replacement="▁", prepend_scheme="always")
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer_json = json.loads(tokenizer.to_str())
    vocab = tokenizer.get_vocab(with_added_tokens=True)
    return {
        "kind": kind,
        "tokenizer_json": tokenizer_json,
        "vocab_size": len(vocab),
        "special_tokens": list(HF_ALL_SPECIAL_TOKENS),
        "metaspace_replacement": "▁",
    }


def build_hf_bpe_artifact(
    corpus_root: str | Path,
    output_path: str | Path,
    *,
    target_vocab_size: int = 512,
    min_frequency: int = 2,
) -> dict[str, object]:
    texts = corpus_texts_for_tokenizer(corpus_root)
    if not texts:
        raise RuntimeError(f"no texts available to build tokenizer under {corpus_root}")
    artifact = _build_hf_tokenizer("hf_bpe", texts, target_vocab_size, min_frequency)
    artifact.update(
        {
            "corpus_root": str(Path(corpus_root)),
            "target_vocab_size": target_vocab_size,
            "min_frequency": min_frequency,
            "text_count": len(texts),
            "character_count": sum(len(text) for text in texts),
        }
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    return artifact


def build_hf_unigram_artifact(
    corpus_root: str | Path,
    output_path: str | Path,
    *,
    target_vocab_size: int = 512,
) -> dict[str, object]:
    texts = corpus_texts_for_tokenizer(corpus_root)
    if not texts:
        raise RuntimeError(f"no texts available to build tokenizer under {corpus_root}")
    artifact = _build_hf_tokenizer("hf_unigram", texts, target_vocab_size, min_frequency=1)
    artifact.update(
        {
            "corpus_root": str(Path(corpus_root)),
            "target_vocab_size": target_vocab_size,
            "text_count": len(texts),
            "character_count": sum(len(text) for text in texts),
        }
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    return artifact


def build_hf_remote_artifact(
    repo_id: str,
    output_path: str | Path,
    *,
    revision: str | None = None,
    trust_remote_code: bool = False,
) -> dict[str, object]:
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers is required to import Hugging Face tokenizers")
    load_project_env()
    token = huggingface_token()
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        revision=revision,
        token=token,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    additions: dict[str, str] = {}
    if getattr(tokenizer, "pad_token", None) != "<pad>":
        additions["pad_token"] = "<pad>"
    if getattr(tokenizer, "bos_token", None) != "<bos>":
        additions["bos_token"] = "<bos>"
    if getattr(tokenizer, "eos_token", None) != "<eos>":
        additions["eos_token"] = "<eos>"
    if getattr(tokenizer, "sep_token", None) != "<sep>":
        additions["sep_token"] = "<sep>"
    if additions:
        tokenizer.add_special_tokens(additions)
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise RuntimeError(f"{repo_id} did not produce a fast tokenizer backend")
    artifact = {
        "kind": "hf_auto",
        "tokenizer_json": json.loads(backend.to_str()),
        "vocab_size": len(tokenizer),
        "special_tokens": sorted(set(list(getattr(tokenizer, "all_special_tokens", [])) + list(HF_ALL_SPECIAL_TOKENS))),
        "metaspace_replacement": "▁",
        "source_repo": repo_id,
        "source_revision": revision,
        "source_class": tokenizer.__class__.__name__,
        "trust_remote_code": trust_remote_code,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    return artifact
