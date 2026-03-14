from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<sep>")
SPECIAL_TOKEN_OFFSET = len(SPECIAL_TOKENS)


@dataclass
class ByteTokenizer:
    token_to_id: dict[str, int] = field(init=False)
    id_to_token: dict[int, str] = field(init=False)
    vocab_size: int = field(init=False)

    def __post_init__(self) -> None:
        self.token_to_id = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        self.id_to_token = {index: token for index, token in enumerate(SPECIAL_TOKENS)}
        self.vocab_size = SPECIAL_TOKEN_OFFSET + 256

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        token_ids: list[int] = []
        if add_bos:
            token_ids.append(self.token_to_id["<bos>"])
        token_ids.extend(byte + SPECIAL_TOKEN_OFFSET for byte in text.encode("utf-8"))
        if add_eos:
            token_ids.append(self.token_to_id["<eos>"])
        return token_ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        raw_bytes = bytearray()
        text_fragments: list[str] = []
        for token_id in token_ids:
            if token_id < SPECIAL_TOKEN_OFFSET:
                if not skip_special_tokens:
                    text_fragments.append(self.id_to_token[token_id])
                continue
            raw_bytes.append(token_id - SPECIAL_TOKEN_OFFSET)
        decoded = raw_bytes.decode("utf-8", errors="ignore")
        if not text_fragments:
            return decoded
        return "".join(text_fragments) + decoded

    def count_tokens(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> int:
        return len(self.encode(text, add_bos=add_bos, add_eos=add_eos))


@dataclass
class GreedyBytePieceTokenizer:
    pieces_hex: list[str]
    token_to_id: dict[str, int] = field(init=False)
    id_to_token: dict[int, str] = field(init=False)
    vocab_size: int = field(init=False)
    byte_token_offset: int = field(init=False)
    max_piece_length: int = field(init=False)
    piece_to_id: dict[bytes, int] = field(init=False)
    id_to_piece: dict[int, bytes] = field(init=False)
    pieces: list[bytes] = field(init=False)

    def __post_init__(self) -> None:
        self.token_to_id = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        self.id_to_token = {index: token for index, token in enumerate(SPECIAL_TOKENS)}

        seen: set[bytes] = set()
        pieces: list[bytes] = []
        for piece_hex in self.pieces_hex:
            piece = bytes.fromhex(piece_hex)
            if len(piece) < 2 or piece in seen:
                continue
            seen.add(piece)
            pieces.append(piece)

        self.pieces = pieces
        self.piece_to_id = {
            piece: SPECIAL_TOKEN_OFFSET + index for index, piece in enumerate(self.pieces)
        }
        self.id_to_piece = {
            SPECIAL_TOKEN_OFFSET + index: piece for index, piece in enumerate(self.pieces)
        }
        self.byte_token_offset = SPECIAL_TOKEN_OFFSET + len(self.pieces)
        self.max_piece_length = max((len(piece) for piece in self.pieces), default=1)
        self.vocab_size = self.byte_token_offset + 256

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        token_ids: list[int] = []
        if add_bos:
            token_ids.append(self.token_to_id["<bos>"])

        payload = text.encode("utf-8")
        index = 0
        while index < len(payload):
            max_length = min(self.max_piece_length, len(payload) - index)
            matched_id: int | None = None
            matched_length = 0
            for length in range(max_length, 1, -1):
                piece = payload[index : index + length]
                token_id = self.piece_to_id.get(piece)
                if token_id is not None:
                    matched_id = token_id
                    matched_length = length
                    break
            if matched_id is not None:
                token_ids.append(matched_id)
                index += matched_length
                continue
            token_ids.append(self.byte_token_offset + payload[index])
            index += 1

        if add_eos:
            token_ids.append(self.token_to_id["<eos>"])
        return token_ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        raw_bytes = bytearray()
        text_fragments: list[str] = []
        for token_id in token_ids:
            if token_id < SPECIAL_TOKEN_OFFSET:
                if not skip_special_tokens:
                    text_fragments.append(self.id_to_token[token_id])
                continue
            if token_id < self.byte_token_offset:
                raw_bytes.extend(self.id_to_piece[token_id])
                continue
            raw_bytes.append(token_id - self.byte_token_offset)
        decoded = raw_bytes.decode("utf-8", errors="ignore")
        if not text_fragments:
            return decoded
        return "".join(text_fragments) + decoded

    def count_tokens(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> int:
        return len(self.encode(text, add_bos=add_bos, add_eos=add_eos))

    def to_artifact(self) -> dict[str, object]:
        return {
            "kind": "greedy_bytes",
            "pieces_hex": [piece.hex() for piece in self.pieces],
            "vocab_size": self.vocab_size,
            "piece_count": len(self.pieces),
            "byte_fallback": True,
            "special_tokens": list(SPECIAL_TOKENS),
        }


def load_greedy_byte_piece_tokenizer(path: str | Path) -> GreedyBytePieceTokenizer:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("kind") != "greedy_bytes":
        raise ValueError(f"unsupported tokenizer artifact kind: {payload.get('kind')}")
    return GreedyBytePieceTokenizer(list(payload.get("pieces_hex", [])))


def _config_value(config: object | None, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def load_tokenizer(config: object | None = None) -> ByteTokenizer | GreedyBytePieceTokenizer:
    kind = str(_config_value(config, "kind", "byte"))
    if kind == "byte":
        return ByteTokenizer()
    if kind == "greedy_bytes":
        artifact_path = _config_value(config, "path")
        if not artifact_path:
            raise RuntimeError("tokenizer.kind=greedy_bytes requires tokenizer.path")
        return load_greedy_byte_piece_tokenizer(artifact_path)
    raise ValueError(f"unsupported tokenizer kind: {kind}")
