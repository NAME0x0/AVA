from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<sep>")
SPECIAL_TOKEN_OFFSET = len(SPECIAL_TOKENS)


def _special_token_maps() -> tuple[dict[str, int], dict[int, str]]:
    token_to_id = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    id_to_token = {index: token for index, token in enumerate(SPECIAL_TOKENS)}
    return token_to_id, id_to_token


@dataclass
class ByteTokenizer:
    token_to_id: dict[str, int] = field(init=False)
    id_to_token: dict[int, str] = field(init=False)
    vocab_size: int = field(init=False)

    def __post_init__(self) -> None:
        self.token_to_id, self.id_to_token = _special_token_maps()
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
        self.token_to_id, self.id_to_token = _special_token_maps()

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


@dataclass
class ByteBPETokenizer:
    merges: list[dict[str, str]]
    token_to_id: dict[str, int] = field(init=False)
    id_to_token: dict[int, str] = field(init=False)
    vocab_size: int = field(init=False)
    merge_token_offset: int = field(init=False)
    piece_to_id: dict[bytes, int] = field(init=False)
    id_to_piece: dict[int, bytes] = field(init=False)
    merge_ranks: dict[tuple[bytes, bytes], int] = field(init=False)
    merged_pieces: list[bytes] = field(init=False)

    def __post_init__(self) -> None:
        self.token_to_id, self.id_to_token = _special_token_maps()
        self.piece_to_id = {}
        self.id_to_piece = {}
        for byte in range(256):
            piece = bytes([byte])
            token_id = SPECIAL_TOKEN_OFFSET + byte
            self.piece_to_id[piece] = token_id
            self.id_to_piece[token_id] = piece
        self.merge_ranks = {}
        self.merged_pieces = []
        next_token_id = SPECIAL_TOKEN_OFFSET + 256
        for rank, merge in enumerate(self.merges):
            left = bytes.fromhex(str(merge["left"]))
            right = bytes.fromhex(str(merge["right"]))
            merged = bytes.fromhex(str(merge.get("merged", (left + right).hex())))
            self.merge_ranks[(left, right)] = rank
            if merged in self.piece_to_id:
                continue
            self.piece_to_id[merged] = next_token_id
            self.id_to_piece[next_token_id] = merged
            self.merged_pieces.append(merged)
            next_token_id += 1
        self.merge_token_offset = SPECIAL_TOKEN_OFFSET + 256
        self.vocab_size = next_token_id

    def _merge_bytes(self, payload: bytes) -> list[bytes]:
        pieces = [bytes([value]) for value in payload]
        if len(pieces) < 2:
            return pieces
        while True:
            best_rank: int | None = None
            best_pair: tuple[bytes, bytes] | None = None
            for index in range(len(pieces) - 1):
                pair = (pieces[index], pieces[index + 1])
                rank = self.merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            merged_piece = best_pair[0] + best_pair[1]
            updated: list[bytes] = []
            index = 0
            while index < len(pieces):
                if index < len(pieces) - 1 and pieces[index] == best_pair[0] and pieces[index + 1] == best_pair[1]:
                    updated.append(merged_piece)
                    index += 2
                    continue
                updated.append(pieces[index])
                index += 1
            pieces = updated
        return pieces

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
        for piece in self._merge_bytes(text.encode("utf-8")):
            token_ids.append(self.piece_to_id[piece])
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
            raw_bytes.extend(self.id_to_piece[token_id])
        decoded = raw_bytes.decode("utf-8", errors="ignore")
        if not text_fragments:
            return decoded
        return "".join(text_fragments) + decoded

    def count_tokens(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> int:
        return len(self.encode(text, add_bos=add_bos, add_eos=add_eos))

    def to_artifact(self) -> dict[str, object]:
        return {
            "kind": "byte_bpe",
            "merges": list(self.merges),
            "vocab_size": self.vocab_size,
            "merge_count": len(self.merges),
            "special_tokens": list(SPECIAL_TOKENS),
            "base_bytes": 256,
        }


def load_greedy_byte_piece_tokenizer(path: str | Path) -> GreedyBytePieceTokenizer:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("kind") != "greedy_bytes":
        raise ValueError(f"unsupported tokenizer artifact kind: {payload.get('kind')}")
    return GreedyBytePieceTokenizer(list(payload.get("pieces_hex", [])))


def load_byte_bpe_tokenizer(path: str | Path) -> ByteBPETokenizer:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("kind") != "byte_bpe":
        raise ValueError(f"unsupported tokenizer artifact kind: {payload.get('kind')}")
    return ByteBPETokenizer(list(payload.get("merges", [])))


def token_piece_bytes(tokenizer: Any, token_id: int) -> bytes | None:
    if token_id < SPECIAL_TOKEN_OFFSET:
        return None
    if isinstance(tokenizer, ByteTokenizer):
        return bytes([token_id - SPECIAL_TOKEN_OFFSET])
    if isinstance(tokenizer, GreedyBytePieceTokenizer):
        if token_id < tokenizer.byte_token_offset:
            return tokenizer.id_to_piece[token_id]
        return bytes([token_id - tokenizer.byte_token_offset])
    if isinstance(tokenizer, ByteBPETokenizer):
        return tokenizer.id_to_piece[token_id]
    return None


def _config_value(config: object | None, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def load_tokenizer(config: object | None = None) -> ByteTokenizer | GreedyBytePieceTokenizer | ByteBPETokenizer:
    kind = str(_config_value(config, "kind", "byte"))
    if kind == "byte":
        return ByteTokenizer()
    if kind == "greedy_bytes":
        artifact_path = _config_value(config, "path")
        if not artifact_path:
            raise RuntimeError("tokenizer.kind=greedy_bytes requires tokenizer.path")
        return load_greedy_byte_piece_tokenizer(artifact_path)
    if kind == "byte_bpe":
        artifact_path = _config_value(config, "path")
        if not artifact_path:
            raise RuntimeError("tokenizer.kind=byte_bpe requires tokenizer.path")
        return load_byte_bpe_tokenizer(artifact_path)
    raise ValueError(f"unsupported tokenizer kind: {kind}")
