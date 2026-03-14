from ava.tokenizer import ByteTokenizer


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
