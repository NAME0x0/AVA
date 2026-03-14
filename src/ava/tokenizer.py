from __future__ import annotations

from dataclasses import dataclass, field


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
