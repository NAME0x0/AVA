from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any

from ava.env import huggingface_token, load_project_env
from ava.retrieval import SupportExample, _unwrap_prompt

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@dataclass(slots=True)
class DenseSupportRetriever:
    model_name: str
    device: str
    examples: tuple[SupportExample, ...]
    vectors: tuple[tuple[float, ...], ...]
    backend: Any = field(repr=False, compare=False)

    def shortlist(
        self,
        prompt: str,
        *,
        top_k: int = 64,
        category_hint: str | None = None,
        category_gated: bool = True,
    ) -> tuple[list[SupportExample], dict[str, object]]:
        if not self.examples:
            return [], {
                "model_name": self.model_name,
                "device": self.device,
                "candidate_count": 0,
                "top_k": top_k,
                "matches": [],
            }
        query_text = prepare_dense_text(_unwrap_prompt(prompt), self.model_name, role="query")
        query_vector = _normalize_vector(
            _to_vector(
                self.backend.encode(query_text, normalize_embeddings=True, show_progress_bar=False)
            )
        )
        allowed_indices = list(
            _allowed_indices(
                self.examples, category_hint=category_hint, category_gated=category_gated
            )
        )
        ranked = rank_dense_support_vectors(
            query_vector, self.vectors, allowed_indices=allowed_indices, top_k=top_k
        )
        matches = [
            {
                "score": round(float(score), 6),
                "prompt": self.examples[index].prompt,
                "response": self.examples[index].response,
                "category": self.examples[index].category,
                "kind": self.examples[index].kind,
                "source_path": self.examples[index].source_path,
            }
            for index, score in ranked
        ]
        return [self.examples[index] for index, _score in ranked], {
            "model_name": self.model_name,
            "device": self.device,
            "candidate_count": len(allowed_indices),
            "top_k": top_k,
            "matches": matches,
        }


def prepare_dense_text(text: str, model_name: str, *, role: str) -> str:
    lowered = model_name.lower()
    if "e5" in lowered:
        prefix = "query:" if role == "query" else "passage:"
        return f"{prefix} {text}".strip()
    if "bge" in lowered and role == "query":
        return f"Represent this sentence for searching relevant passages: {text}".strip()
    return text.strip()


def _support_dense_text(example: SupportExample, model_name: str) -> str:
    return prepare_dense_text(_unwrap_prompt(example.prompt), model_name, role="support")


def _to_vector(values: Any) -> tuple[float, ...]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], (list, tuple)):
        values = values[0]
    return tuple(float(value) for value in values)


def _normalize_vector(values: tuple[float, ...]) -> tuple[float, ...]:
    norm = sqrt(sum(value * value for value in values))
    if norm <= 0.0:
        return values
    return tuple(value / norm for value in values)


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(
        left_value * right_value for left_value, right_value in zip(left, right, strict=True)
    )


def _allowed_indices(
    examples: tuple[SupportExample, ...],
    *,
    category_hint: str | None,
    category_gated: bool,
) -> tuple[int, ...]:
    if not category_gated or not category_hint:
        return tuple(range(len(examples)))
    filtered = tuple(
        index for index, example in enumerate(examples) if example.category == category_hint
    )
    return filtered or tuple(range(len(examples)))


def rank_dense_support_vectors(
    query_vector: tuple[float, ...],
    support_vectors: tuple[tuple[float, ...], ...],
    *,
    allowed_indices: list[int] | tuple[int, ...] | None = None,
    top_k: int = 64,
) -> list[tuple[int, float]]:
    indices = (
        list(allowed_indices) if allowed_indices is not None else list(range(len(support_vectors)))
    )
    scored = [(index, _dot(query_vector, support_vectors[index])) for index in indices]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[: max(top_k, 0)]


def build_dense_support_retriever(
    support_examples: list[SupportExample],
    *,
    model_name: str = "BAAI/bge-small-en-v1.5",
    device: str = "cpu",
    trust_remote_code: bool = False,
) -> DenseSupportRetriever:
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise RuntimeError("sentence-transformers is required for dense retrieval")
    load_project_env()
    token = huggingface_token()
    try:
        backend = SentenceTransformer(
            model_name,
            device=device,
            token=token,
            trust_remote_code=trust_remote_code,
        )
    except TypeError:
        backend = SentenceTransformer(model_name, device=device, token=token)
    except Exception:
        try:
            backend = SentenceTransformer(
                model_name,
                device=device,
                token=token,
                trust_remote_code=trust_remote_code,
                local_files_only=True,
            )
        except TypeError:
            backend = SentenceTransformer(
                model_name, device=device, token=token, local_files_only=True
            )
    examples = tuple(support_examples)
    support_texts = [_support_dense_text(example, model_name) for example in examples]
    encoded = backend.encode(support_texts, normalize_embeddings=True, show_progress_bar=False)
    vectors = tuple(_normalize_vector(_to_vector(vector)) for vector in encoded)
    return DenseSupportRetriever(
        model_name=model_name,
        device=device,
        examples=examples,
        vectors=vectors,
        backend=backend,
    )
