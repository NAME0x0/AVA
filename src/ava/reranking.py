from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ava.env import huggingface_token, load_project_env
from ava.retrieval import SupportExample, _unwrap_prompt

try:
    from sentence_transformers import CrossEncoder

    CROSS_ENCODER_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    CrossEncoder = None
    CROSS_ENCODER_AVAILABLE = False


@dataclass(slots=True)
class SupportReranker:
    model_name: str
    device: str
    backend: Any = field(repr=False, compare=False)

    def rerank(
        self,
        prompt: str,
        examples: list[SupportExample],
        *,
        top_k: int = 8,
        category_hint: str | None = None,
        category_gated: bool = True,
    ) -> tuple[list[SupportExample], dict[str, object]]:
        filtered = [
            example
            for example in examples
            if not category_gated or not category_hint or example.category == category_hint
        ]
        if not filtered:
            filtered = list(examples)
        if not filtered:
            return [], {
                'model_name': self.model_name,
                'device': self.device,
                'candidate_count': 0,
                'top_k': top_k,
                'matches': [],
            }
        query = _unwrap_prompt(prompt).strip()
        pairs = [(query, _support_rerank_text(example)) for example in filtered]
        scores = self.backend.predict(pairs, show_progress_bar=False)
        if hasattr(scores, 'tolist'):
            scores = scores.tolist()
        normalized_scores: list[float] = []
        for value in scores:
            if isinstance(value, (list, tuple)):
                normalized_scores.append(float(value[0]))
            else:
                normalized_scores.append(float(value))
        ranked = sorted(
            zip(filtered, normalized_scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )[: max(top_k, 0)]
        matches = [
            {
                'score': round(float(score), 6),
                'prompt': example.prompt,
                'response': example.response,
                'category': example.category,
                'kind': example.kind,
                'source_path': example.source_path,
            }
            for example, score in ranked
        ]
        return [example for example, _score in ranked], {
            'model_name': self.model_name,
            'device': self.device,
            'candidate_count': len(filtered),
            'top_k': top_k,
            'matches': matches,
        }


def _support_rerank_text(example: SupportExample) -> str:
    return f"{_unwrap_prompt(example.prompt).strip()}\n\nAnswer: {example.response.strip()}".strip()


def build_support_reranker(
    *,
    model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    device: str = 'cpu',
    trust_remote_code: bool = False,
) -> SupportReranker:
    if not CROSS_ENCODER_AVAILABLE:
        raise RuntimeError('sentence-transformers is required for support reranking')
    load_project_env()
    token = huggingface_token()
    try:
        backend = CrossEncoder(
            model_name,
            device=device,
            token=token,
            trust_remote_code=trust_remote_code,
        )
    except TypeError:
        backend = CrossEncoder(model_name, device=device)
    except Exception:
        try:
            backend = CrossEncoder(
                model_name,
                device=device,
                token=token,
                trust_remote_code=trust_remote_code,
                local_files_only=True,
            )
        except TypeError:
            backend = CrossEncoder(model_name, device=device, local_files_only=True)
    return SupportReranker(model_name=model_name, device=device, backend=backend)
