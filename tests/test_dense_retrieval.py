from ava.dense_retrieval import prepare_dense_text, rank_dense_support_vectors


def test_prepare_dense_text_uses_e5_and_bge_conventions() -> None:
    assert (
        prepare_dense_text("what is gravity?", "intfloat/e5-small-v2", role="query")
        == "query: what is gravity?"
    )
    assert (
        prepare_dense_text("gravity is a force", "intfloat/e5-small-v2", role="support")
        == "passage: gravity is a force"
    )
    assert prepare_dense_text(
        "what is gravity?", "BAAI/bge-small-en-v1.5", role="query"
    ).startswith("Represent this sentence")


def test_rank_dense_support_vectors_orders_by_similarity() -> None:
    ranked = rank_dense_support_vectors(
        (1.0, 0.0),
        ((0.9, 0.1), (0.1, 0.9), (1.0, 0.0)),
        top_k=2,
    )
    assert [index for index, _score in ranked] == [2, 0]
