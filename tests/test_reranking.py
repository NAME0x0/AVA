from ava.reranking import SupportReranker
from ava.retrieval import SupportExample


class _FakeBackend:
    def predict(self, pairs, show_progress_bar=False):
        assert show_progress_bar is False
        scores = []
        for query, support in pairs:
            if "photosynthesis" in query and "carbon dioxide" in support:
                scores.append(0.9)
            elif "photosynthesis" in query and "oxygen" in support:
                scores.append(0.1)
            else:
                scores.append(0.0)
        return scores


def test_support_reranker_orders_matches_and_reports_payload() -> None:
    reranker = SupportReranker(model_name="fake/reranker", device="cpu", backend=_FakeBackend())
    examples = [
        SupportExample(
            prompt="Which gas do plants take in from the air for photosynthesis?",
            response="oxygen",
            category="science",
        ),
        SupportExample(
            prompt="What gas do plants take in from the air for photosynthesis?",
            response="carbon dioxide",
            category="science",
        ),
    ]
    ranked, payload = reranker.rerank(
        "Which gas do plants take in for photosynthesis?",
        examples,
        top_k=2,
        category_hint="science",
        category_gated=True,
    )
    assert ranked[0].response == "carbon dioxide"
    assert payload["model_name"] == "fake/reranker"
    assert payload["matches"][0]["response"] == "carbon dioxide"
    assert payload["matches"][0]["score"] > payload["matches"][1]["score"]
