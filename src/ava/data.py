from __future__ import annotations

import json
from pathlib import Path

from ava.tokenizer import ByteTokenizer

TEXT_SUFFIXES = {".txt", ".md"}
JSONL_SUFFIXES = {".jsonl"}
SUPERVISED_METADATA_KEYS = (
    "kind",
    "category",
    "difficulty",
    "format_contract",
    "teacher_model",
    "source_type",
    "teacher_rationale_short",
    "verifier_status",
    "student_failure",
    "repair_note",
    "tags",
)


def discover_corpus_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    paths = [path for path in root_path.rglob("*") if path.is_file()]
    return sorted(paths)


def _text_from_json_line(payload: dict[str, object]) -> str:
    candidates = (
        payload.get("text"),
        payload.get("prompt"),
        payload.get("response"),
        payload.get("question"),
        payload.get("answer"),
        payload.get("rationale"),
    )
    return "\n".join(str(item) for item in candidates if item)


def _normalize_supervised_payload(payload: dict[str, object]) -> dict[str, object] | None:
    prompt = payload.get("prompt") or payload.get("question")
    response = payload.get("response") or payload.get("answer")
    if not prompt or not response:
        return None
    example: dict[str, object] = {
        "prompt": str(prompt),
        "response": str(response),
    }
    for key in SUPERVISED_METADATA_KEYS:
        value = payload.get(key)
        if value is None:
            continue
        if key == "tags" and isinstance(value, list):
            example[key] = [str(item) for item in value]
            continue
        example[key] = str(value)
    return example


def load_text_corpus(root: str | Path) -> list[str]:
    texts: list[str] = []
    for path in discover_corpus_files(root):
        if path.suffix in TEXT_SUFFIXES:
            texts.append(path.read_text(encoding="utf-8"))
        elif path.suffix in JSONL_SUFFIXES:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    text = _text_from_json_line(payload)
                    if text:
                        texts.append(text)
    return texts


def load_supervised_examples(root: str | Path) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for path in discover_corpus_files(root):
        if path.suffix not in JSONL_SUFFIXES:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            example = _normalize_supervised_payload(payload)
            if example is not None:
                examples.append(example)
    return examples


def encode_corpus(texts: list[str], tokenizer: ByteTokenizer) -> list[int]:
    token_ids: list[int] = []
    for text in texts:
        token_ids.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
    return token_ids
