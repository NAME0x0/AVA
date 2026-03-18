from __future__ import annotations

import json
import re
from collections import Counter
from math import log
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

HF_DATASETS_CACHE = Path.home() / ".cache" / "huggingface" / "datasets"

from ava.config import ExperimentConfig
from ava.dense_retrieval import DenseSupportRetriever, build_dense_support_retriever
from ava.reranking import SupportReranker, build_support_reranker
from ava.eval import _resolve_device
from ava.memory_transfer import _generate_completion
from ava.model import TORCH_AVAILABLE, build_model, torch
from ava.retrieval import (
    SupportExample,
    _canonical_lookup_text,
    load_support_examples,
    lookup_support_answer,
    lookup_support_answer_nearest,
    prepare_retrieval_prompt,
)
from ava.tokenizer import load_tokenizer

try:
    import pyarrow.ipc as pa_ipc
except ImportError:  # pragma: no cover - exercised only when optional deps are missing
    pa_ipc = None

try:
    from datasets import load_dataset

    DATASETS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when optional deps are missing
    DATASETS_AVAILABLE = False

if TORCH_AVAILABLE:
    import torch.nn.functional as F
else:  # pragma: no cover - exercised only when torch is unavailable
    F = None


@dataclass(frozen=True, slots=True)
class ExternalBenchmarkTask:
    benchmark: str
    task_id: str
    category: str
    prompt: str
    metric: str
    expected: str
    choices: tuple[tuple[str, str], ...] = ()
    expected_text: str | None = None


@dataclass(frozen=True, slots=True)
class ExternalBenchmarkResult:
    benchmark: str
    task_id: str
    category: str
    prompt: str
    expected: str
    completion: str
    matched: bool
    retrieval: dict[str, object] | None = None
    expected_text: str | None = None
    choices: tuple[tuple[str, str], ...] = ()
    scoring: dict[str, object] | None = None


def _slice_split(split: str, limit: int | None, offset: int = 0) -> str:
    if "[" in split:
        return split
    start = max(offset, 0)
    if limit is None:
        return f"{split}[{start}:]" if start else split
    end = start + max(limit, 0)
    return f"{split}[{start}:{end}]"


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def extract_gsm8k_answer(text: str) -> str:
    if "####" in text:
        text = text.split("####", 1)[1]
    matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if not matches:
        return text.strip()
    return matches[-1].replace(",", "")


def format_arc_prompt(question: str, choices: tuple[tuple[str, str], ...]) -> str:
    lines = [question.strip(), "", "Options:"]
    for label, choice in choices:
        lines.append(f"{label}. {choice}")
    lines.extend(["", "Reply with only the correct option label."])
    return "\n".join(lines)


def _match_multiple_choice(task: ExternalBenchmarkTask, completion: str) -> bool:
    normalized = completion.strip()
    label_match = re.search(r"\b([A-Z])\b", normalized.upper())
    if label_match and label_match.group(1) == task.expected.upper():
        return True
    expected_text = _normalize_text(task.expected_text or "")
    if not expected_text:
        return False
    completion_text = _normalize_text(normalized)
    return completion_text == expected_text or expected_text in completion_text


def _match_exact(task: ExternalBenchmarkTask, completion: str) -> bool:
    if task.benchmark == "gsm8k":
        return extract_gsm8k_answer(completion) == extract_gsm8k_answer(task.expected)
    return _normalize_text(completion) == _normalize_text(task.expected)



def _support_lookup_tokens(text: str) -> list[str]:
    return [token for token in _canonical_lookup_text(text).split() if token]


def _parse_multiple_choice_prompt(prompt: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    stripped = prompt.strip()
    if "\n\nOptions:\n" not in stripped:
        return stripped, ()
    stem, remainder = stripped.split("\n\nOptions:\n", 1)
    option_block = remainder.split("\n\nReply with only", 1)[0]
    choices: list[tuple[str, str]] = []
    for line in option_block.splitlines():
        match = re.match(r"^\s*([A-Za-z0-9]+)[\.)]\s*(.+?)\s*$", line)
        if match is None:
            continue
        choices.append((match.group(1).strip(), match.group(2).strip()))
    return stem.strip(), tuple(choices)


def _normalize_support_response(prompt: str, response: str) -> str:
    _, choices = _parse_multiple_choice_prompt(prompt)
    if not choices:
        return response.strip()
    response_key = response.strip()
    direct = {label: text for label, text in choices}
    if response_key in direct:
        return direct[response_key]
    upper = {label.upper(): text for label, text in choices}
    if response_key.upper() in upper:
        return upper[response_key.upper()]
    return response_key


def _support_signature(prompt: str, normalized_response: str) -> tuple[str, tuple[str, ...], str]:
    return (_normalize_text(prompt), (), _normalize_text(normalized_response))


@dataclass(frozen=True, slots=True)
class SparseSupportRow:
    question_tokens: tuple[str, ...]
    option_tokens: tuple[str, ...]
    response_tokens: tuple[str, ...]
    full_tokens: tuple[str, ...]
    prompt_counts: dict[str, int]
    question_counts: dict[str, int]
    option_counts: dict[str, int]
    full_counts: dict[str, int]
    response_token_set: frozenset[str]
    response: str
    category: str
    reference: dict[str, object]


@dataclass(frozen=True, slots=True)
class SparseSupportIndex:
    rows: tuple[SparseSupportRow, ...]
    idf: dict[str, float]
    category_rows: dict[str, tuple[int, ...]]
    prompt_inverted: dict[str, tuple[int, ...]]


def _build_sparse_support_index(examples: list[SupportExample]) -> SparseSupportIndex:
    document_frequency: Counter[str] = Counter()
    rows: list[SparseSupportRow] = []
    category_rows: dict[str, list[int]] = {}
    prompt_inverted: dict[str, list[int]] = {}

    for example in examples:
        stem, choices = _parse_multiple_choice_prompt(example.prompt)
        normalized_response = _normalize_support_response(example.prompt, example.response)
        question_tokens = tuple(_support_lookup_tokens(stem or example.prompt))
        option_tokens = tuple(_support_lookup_tokens(' '.join(choice_text for _label, choice_text in choices)))
        response_tokens = tuple(_support_lookup_tokens(normalized_response))
        prompt_tokens = tuple(_support_lookup_tokens(example.prompt))
        full_tokens = prompt_tokens + response_tokens
        row = SparseSupportRow(
            question_tokens=question_tokens,
            option_tokens=option_tokens,
            response_tokens=response_tokens,
            full_tokens=full_tokens,
            prompt_counts=dict(Counter(prompt_tokens)),
            question_counts=dict(Counter(question_tokens)),
            option_counts=dict(Counter(option_tokens)),
            full_counts=dict(Counter(full_tokens)),
            response_token_set=frozenset(response_tokens),
            response=normalized_response,
            category=example.category,
            reference={
                'prompt': example.prompt,
                'response': example.response,
                'normalized_response': normalized_response,
                'category': example.category,
                'kind': example.kind,
                'source_path': example.source_path,
                'reference_count': 1,
            },
        )
        row_index = len(rows)
        rows.append(row)
        category_rows.setdefault(row.category, []).append(row_index)
        for token in set(row.prompt_counts):
            prompt_inverted.setdefault(token, []).append(row_index)
        for token in set(row.full_tokens):
            document_frequency[token] += 1
    total_documents = len(rows)
    idf = {
        token: log((total_documents + 1) / (count + 1)) + 1.0
        for token, count in document_frequency.items()
    }
    return SparseSupportIndex(
        rows=tuple(rows),
        idf=idf,
        category_rows={key: tuple(value) for key, value in category_rows.items()},
        prompt_inverted={key: tuple(value) for key, value in prompt_inverted.items()},
    )


def _candidate_row_indices(
    support_index: SparseSupportIndex,
    query_tokens: list[str],
    *,
    category_hint: str | None,
    category_gated: bool,
    max_candidates: int = 8192,
) -> list[int]:
    if category_gated and category_hint and category_hint in support_index.category_rows:
        base_indices = list(support_index.category_rows[category_hint])
    else:
        base_indices = list(range(len(support_index.rows)))
    if len(base_indices) <= max_candidates:
        return base_indices
    allowed = set(base_indices)
    votes: Counter[int] = Counter()
    for token in set(query_tokens):
        for row_index in support_index.prompt_inverted.get(token, ()): 
            if row_index in allowed:
                votes[row_index] += 1
    if not votes:
        return base_indices[:max_candidates]
    ranked = [row_index for row_index, _count in votes.most_common(max_candidates)]
    if len(ranked) < max_candidates:
        seen = set(ranked)
        for row_index in base_indices:
            if row_index in seen:
                continue
            ranked.append(row_index)
            if len(ranked) >= max_candidates:
                break
    return ranked


def _sparse_support_score(query_counts: Counter[str], support_counts: dict[str, int], idf: dict[str, float]) -> float:
    shared = set(query_counts) & set(support_counts)
    if not shared:
        return 0.0
    numerator = sum(min(query_counts[token], support_counts[token]) * idf.get(token, 1.0) for token in shared)
    denominator = sum(query_counts[token] * idf.get(token, 1.0) for token in query_counts)
    return numerator / max(denominator, 1e-9)


def _support_response_choice_score(response_tokens: frozenset[str], choice_tokens: frozenset[str]) -> float:
    if not response_tokens or not choice_tokens:
        return 0.0
    return len(response_tokens & choice_tokens) / len(response_tokens | choice_tokens)


def _support_graph_edge_score(left_counts: dict[str, int], right_counts: dict[str, int], idf: dict[str, float]) -> float:
    shared = set(left_counts) & set(right_counts)
    if not shared:
        return 0.0
    numerator = sum(idf.get(token, 1.0) for token in shared)
    left_norm = sum(idf.get(token, 1.0) for token in left_counts)
    right_norm = sum(idf.get(token, 1.0) for token in right_counts)
    denominator = max((left_norm * right_norm) ** 0.5, 1e-9)
    return numerator / denominator


def _propagate_support_scores(
    support_index: SparseSupportIndex,
    row_indices: list[int],
    base_scores: dict[int, float],
    *,
    seed_k: int = 12,
    alpha: float = 0.35,
) -> dict[int, float]:
    if not row_indices:
        return {}
    scored_indices = sorted(row_indices, key=lambda index: base_scores.get(index, 0.0), reverse=True)
    seeds = [index for index in scored_indices[:seed_k] if base_scores.get(index, 0.0) > 0.0]
    if not seeds:
        return dict(base_scores)
    propagated = dict(base_scores)
    for candidate_index in row_indices:
        row = support_index.rows[candidate_index]
        bonus = 0.0
        for seed_index in seeds:
            edge = _support_graph_edge_score(row.question_counts, support_index.rows[seed_index].question_counts, support_index.idf)
            if edge <= 0.0:
                continue
            bonus += base_scores.get(seed_index, 0.0) * edge
        propagated[candidate_index] = propagated.get(candidate_index, 0.0) + (alpha * bonus)
    return propagated


def _top_support_details(
    support_index: SparseSupportIndex,
    scores: dict[int, float],
    row_indices: list[int],
    *,
    top_k: int,
) -> list[dict[str, object]]:
    details: list[dict[str, object]] = []
    ranked = sorted(row_indices, key=lambda index: scores.get(index, 0.0), reverse=True)
    for index in ranked[:top_k]:
        details.append({'score': round(float(scores.get(index, 0.0)), 6), **support_index.rows[index].reference})
    return details


def _prompt_query_tokens(prompt: str) -> tuple[list[str], list[str]]:
    stem, choices = _parse_multiple_choice_prompt(prompt)
    question_tokens = _support_lookup_tokens(stem or prompt)
    option_tokens = _support_lookup_tokens(' '.join(choice_text for _label, choice_text in choices))
    return question_tokens, option_tokens


def _predict_multiple_choice_from_support(
    *,
    prompt: str,
    choices: tuple[tuple[str, str], ...],
    support_examples: list[SupportExample] | None,
    support_index: SparseSupportIndex | None = None,
    category_hint: str | None,
    category_gated: bool,
    top_k: int = 8,
) -> tuple[str, dict[str, object]] | None:
    if support_index is None:
        examples = list(support_examples or [])
        if not examples:
            return None
        support_index = _build_sparse_support_index(examples)
    if not support_index.rows:
        return None

    question_tokens, option_tokens = _prompt_query_tokens(prompt)
    query_tokens = _support_lookup_tokens(prompt)
    query_counts = Counter(query_tokens)
    question_counts = Counter(question_tokens)
    option_counts = Counter(option_tokens)
    candidate_indices = _candidate_row_indices(
        support_index,
        query_tokens,
        category_hint=category_hint,
        category_gated=category_gated,
    )
    if not candidate_indices:
        return None
    scored = sorted(
        (
            (
                _sparse_support_score(query_counts, support_index.rows[index].prompt_counts, support_index.idf),
                index,
            )
            for index in candidate_indices
        ),
        key=lambda item: item[0],
        reverse=True,
    )[:top_k]
    choice_scores: Counter[str] = Counter({label: 0.0 for label, _choice_text in choices})
    support_details: list[dict[str, object]] = []
    choice_token_sets = {label: frozenset(_support_lookup_tokens(choice_text)) for label, choice_text in choices}
    for score, index in scored:
        row = support_index.rows[index]
        support_details.append({'score': round(float(score), 6), **row.reference})
        direct_label = next(
            (
                label
                for label, choice_text in choices
                if _normalize_text(choice_text) == _normalize_text(row.response)
            ),
            None,
        )
        if direct_label is not None:
            choice_scores[direct_label] += score * 1.1
            continue
        for label, _choice_text in choices:
            choice_scores[label] += score * _support_response_choice_score(row.response_token_set, choice_token_sets[label])

    if not choice_scores:
        return None
    best_label = max(choice_scores.items(), key=lambda item: item[1])[0]
    return best_label, {
        'mode': 'support_mc',
        'scores': {label: round(float(choice_scores[label]), 6) for label, _choice_text in choices},
        'selected': best_label,
        'supports': support_details,
        'candidate_count': len(candidate_indices),
    }



def _predict_multiple_choice_hybrid_from_support(
    *,
    prompt: str,
    choices: tuple[tuple[str, str], ...],
    support_examples: list[SupportExample] | None,
    support_index: SparseSupportIndex | None = None,
    category_hint: str | None,
    category_gated: bool,
    top_k: int = 8,
) -> tuple[str, dict[str, object]] | None:
    if support_index is None:
        examples = list(support_examples or [])
        if not examples:
            return None
        support_index = _build_sparse_support_index(examples)
    if not support_index.rows:
        return None

    question_tokens, option_tokens = _prompt_query_tokens(prompt)
    query_tokens = _support_lookup_tokens(prompt)
    query_counts = Counter(query_tokens)
    question_counts = Counter(question_tokens)
    option_counts = Counter(option_tokens)
    candidate_indices = _candidate_row_indices(
        support_index,
        query_tokens,
        category_hint=category_hint,
        category_gated=category_gated,
    )
    if not candidate_indices:
        return None
    base_scores = {
        index: _sparse_support_score(query_counts, support_index.rows[index].prompt_counts, support_index.idf)
        for index in candidate_indices
    }
    propagated_scores = _propagate_support_scores(support_index, candidate_indices, base_scores)

    choice_scores: Counter[str] = Counter({label: 0.0 for label, _choice_text in choices})
    choice_supports: dict[str, list[dict[str, object]]] = {}
    for label, choice_text in choices:
        choice_tokens = _support_lookup_tokens(choice_text)
        choice_query_counts = Counter(query_tokens + choice_tokens)
        choice_token_set = frozenset(choice_tokens)
        ranked: list[tuple[float, int]] = []
        for index in candidate_indices:
            row = support_index.rows[index]
            question_match = _sparse_support_score(question_counts, row.question_counts, support_index.idf)
            prompt_match = base_scores.get(index, 0.0)
            option_match = _sparse_support_score(option_counts, row.option_counts, support_index.idf) if option_counts else 0.0
            full_match = _sparse_support_score(choice_query_counts, row.full_counts, support_index.idf)
            response_match = _support_response_choice_score(row.response_token_set, choice_token_set)
            semantic_match = 1.0 if _normalize_text(row.response) == _normalize_text(choice_text) else 0.0
            total = (
                (0.35 * question_match)
                + (0.05 * prompt_match)
                + (0.05 * option_match)
                + (0.2 * full_match)
                + (0.1 * propagated_scores.get(index, 0.0))
                + (0.1 * response_match)
                + (0.3 * semantic_match)
            )
            ranked.append((total, index))
        ranked.sort(key=lambda item: item[0], reverse=True)
        top_rows = ranked[:top_k]
        choice_scores[label] = sum(score for score, _index in top_rows)
        choice_supports[label] = [
            {'score': round(float(score), 6), **support_index.rows[index].reference}
            for score, index in top_rows
            if score > 0.0
        ]

    if not choice_scores:
        return None
    best_label = max(choice_scores.items(), key=lambda item: item[1])[0]
    combined_scores = {label: round(float(choice_scores[label]), 6) for label, _choice_text in choices}
    return best_label, {
        'mode': 'hybrid_support_mc',
        'scores': combined_scores,
        'selected': best_label,
        'supports': _top_support_details(support_index, propagated_scores, candidate_indices, top_k=top_k),
        'choice_supports': choice_supports,
        'candidate_count': len(candidate_indices),
    }




def _predict_model_retrieval_ensemble(
    model: Any,
    tokenizer: Any,
    device: str,
    *,
    prompt: str,
    choices: tuple[tuple[str, str], ...],
    support_examples: list[SupportExample] | None,
    support_index: SparseSupportIndex | None = None,
    category_hint: str | None,
    category_gated: bool,
    top_k: int = 8,
    model_weight: float = 0.4,
) -> tuple[str, dict[str, object]] | None:
    """True weighted ensemble of model logprobs and retrieval scores."""
    # Get model logprob scores
    _, model_scores = _predict_multiple_choice_label(
        model, tokenizer, device, prompt=prompt, choices=choices,
    )
    # Get retrieval scores
    retrieval_result = _predict_multiple_choice_hybrid_from_support(
        prompt=prompt,
        choices=choices,
        support_examples=support_examples,
        support_index=support_index,
        category_hint=category_hint,
        category_gated=category_gated,
        top_k=top_k,
    )
    if retrieval_result is None:
        # No retrieval available, use model only
        best_label = max(model_scores.items(), key=lambda item: item[1])[0]
        return best_label, {
            'mode': 'model_retrieval_ensemble',
            'model_scores': model_scores,
            'retrieval_scores': None,
            'combined_scores': model_scores,
            'selected': best_label,
            'model_weight': model_weight,
        }

    _retrieval_label, retrieval_scoring = retrieval_result
    retrieval_scores = dict(retrieval_scoring.get('scores', {}))

    # Normalize model logprobs to probabilities
    import math as _math
    model_vals = [float(model_scores.get(label, -100.0)) for label, _ in choices]
    max_logp = max(model_vals)
    model_probs = [_math.exp(v - max_logp) for v in model_vals]
    model_sum = sum(model_probs) or 1.0
    model_probs = [p / model_sum for p in model_probs]

    # Normalize retrieval scores to probabilities
    retrieval_vals = [float(retrieval_scores.get(label, 0.0)) for label, _ in choices]
    retrieval_sum = sum(retrieval_vals) or 1.0
    retrieval_probs = [v / retrieval_sum for v in retrieval_vals]

    # Weighted combination
    combined = {}
    for i, (label, _) in enumerate(choices):
        combined[label] = round(
            model_weight * model_probs[i] + (1.0 - model_weight) * retrieval_probs[i],
            6,
        )

    best_label = max(combined.items(), key=lambda item: item[1])[0]
    return best_label, {
        'mode': 'model_retrieval_ensemble',
        'model_scores': model_scores,
        'retrieval_scores': {label: round(float(retrieval_scores.get(label, 0.0)), 6) for label, _ in choices},
        'model_probs': {choices[i][0]: round(model_probs[i], 6) for i in range(len(choices))},
        'retrieval_probs': {choices[i][0]: round(retrieval_probs[i], 6) for i in range(len(choices))},
        'combined_scores': combined,
        'selected': best_label,
        'model_weight': model_weight,
        'supports': retrieval_scoring.get('supports', []),
        'candidate_count': retrieval_scoring.get('candidate_count'),
    }


def _predict_multiple_choice_dense_hybrid_from_support(
    *,
    prompt: str,
    choices: tuple[tuple[str, str], ...],
    dense_retriever: DenseSupportRetriever | None,
    category_hint: str | None,
    category_gated: bool,
    shortlist_top_k: int = 64,
    top_k: int = 8,
) -> tuple[str, dict[str, object]] | None:
    if dense_retriever is None:
        return None
    shortlisted_examples, dense_payload = dense_retriever.shortlist(
        prompt,
        top_k=shortlist_top_k,
        category_hint=category_hint,
        category_gated=category_gated,
    )
    if not shortlisted_examples:
        return None
    prediction = _predict_multiple_choice_hybrid_from_support(
        prompt=prompt,
        choices=choices,
        support_examples=shortlisted_examples,
        support_index=_build_sparse_support_index(shortlisted_examples),
        category_hint=category_hint,
        category_gated=category_gated,
        top_k=min(max(top_k, 1), len(shortlisted_examples)),
    )
    if prediction is None:
        return None
    label, scoring = prediction
    scoring["mode"] = "dense_hybrid_support_mc"
    scoring["dense_model"] = dense_retriever.model_name
    scoring["dense_device"] = dense_retriever.device
    scoring["dense_shortlist"] = dense_payload["matches"]
    scoring["dense_candidate_count"] = dense_payload["candidate_count"]
    scoring["dense_top_k"] = dense_payload["top_k"]
    return label, scoring


def _predict_multiple_choice_dense_rerank_from_support(
    *,
    prompt: str,
    choices: tuple[tuple[str, str], ...],
    dense_retriever: DenseSupportRetriever | None,
    support_reranker: SupportReranker | None,
    category_hint: str | None,
    category_gated: bool,
    shortlist_top_k: int = 64,
    rerank_top_k: int = 12,
    top_k: int = 8,
) -> tuple[str, dict[str, object]] | None:
    if dense_retriever is None or support_reranker is None:
        return None
    shortlisted_examples, dense_payload = dense_retriever.shortlist(
        prompt,
        top_k=shortlist_top_k,
        category_hint=category_hint,
        category_gated=category_gated,
    )
    if not shortlisted_examples:
        return None
    reranked_examples, rerank_payload = support_reranker.rerank(
        prompt,
        shortlisted_examples,
        top_k=min(max(rerank_top_k, 1), len(shortlisted_examples)),
        category_hint=category_hint,
        category_gated=category_gated,
    )
    if not reranked_examples:
        return None
    prediction = _predict_multiple_choice_hybrid_from_support(
        prompt=prompt,
        choices=choices,
        support_examples=reranked_examples,
        support_index=_build_sparse_support_index(reranked_examples),
        category_hint=category_hint,
        category_gated=category_gated,
        top_k=min(max(top_k, 1), len(reranked_examples)),
    )
    if prediction is None:
        return None
    label, scoring = prediction
    scoring["mode"] = "dense_rerank_support_mc"
    scoring["dense_model"] = dense_retriever.model_name
    scoring["dense_device"] = dense_retriever.device
    scoring["dense_shortlist"] = dense_payload["matches"]
    scoring["dense_candidate_count"] = dense_payload["candidate_count"]
    scoring["dense_top_k"] = dense_payload["top_k"]
    scoring["reranker_model"] = support_reranker.model_name
    scoring["reranker_device"] = support_reranker.device
    scoring["rerank_matches"] = rerank_payload["matches"]
    scoring["rerank_top_k"] = rerank_payload["top_k"]
    return label, scoring


def _predict_multiple_choice_sparse_dense_router(
    *,
    prompt: str,
    choices: tuple[tuple[str, str], ...],
    support_resource: dict[str, object] | None,
    dense_retriever: DenseSupportRetriever | None,
    support_reranker: SupportReranker | None = None,
    category_hint: str | None,
    category_gated: bool,
    shortlist_top_k: int = 64,
    rerank_top_k: int = 12,
    top_k: int = 8,
    dense_score_min: float = 0.7,
    dense_margin_min: float = 0.0,
    sparse_margin_max: float = 0.015,
    margin_gap_min: float = 0.0,
) -> tuple[str, dict[str, object]] | None:
    sparse_prediction = (
        _predict_multiple_choice_support_ensemble(
            prompt=prompt,
            choices=choices,
            support_resource=support_resource,
            category_hint=category_hint,
            category_gated=category_gated,
            top_k=top_k,
        )
        if support_resource is not None and support_resource.get('kind') == 'banks'
        else _predict_multiple_choice_hybrid_from_support(
            prompt=prompt,
            choices=choices,
            support_examples=(list(support_resource.get('examples', [])) if support_resource else None),
            support_index=(support_resource.get('index') if support_resource else None),
            category_hint=category_hint,
            category_gated=category_gated,
            top_k=top_k,
        )
    )
    dense_prediction = (
        _predict_multiple_choice_dense_rerank_from_support(
            prompt=prompt,
            choices=choices,
            dense_retriever=dense_retriever,
            support_reranker=support_reranker,
            category_hint=category_hint,
            category_gated=category_gated,
            shortlist_top_k=shortlist_top_k,
            rerank_top_k=rerank_top_k,
            top_k=top_k,
        )
        if support_reranker is not None
        else _predict_multiple_choice_dense_hybrid_from_support(
            prompt=prompt,
            choices=choices,
            dense_retriever=dense_retriever,
            category_hint=category_hint,
            category_gated=category_gated,
            shortlist_top_k=shortlist_top_k,
            top_k=top_k,
        )
    )
    if sparse_prediction is None:
        return dense_prediction
    if dense_prediction is None:
        return sparse_prediction

    sparse_label, sparse_scoring = sparse_prediction
    dense_label, dense_scoring = dense_prediction
    sparse_margin = _scoring_margin(sparse_scoring)
    dense_margin = _scoring_margin(dense_scoring)
    dense_top_score = _rerank_top_score(dense_scoring) if support_reranker is not None else _dense_shortlist_top_score(dense_scoring)

    if sparse_label == dense_label:
        router_decision = 'agreement'
        selected_label = sparse_label
        selected_scoring = sparse_scoring
    else:
        use_dense = (
            dense_top_score >= dense_score_min
            and dense_margin >= dense_margin_min
            and sparse_margin <= sparse_margin_max
            and (dense_margin - sparse_margin) >= margin_gap_min
        )
        router_decision = 'dense' if use_dense else 'sparse'
        selected_label = dense_label if use_dense else sparse_label
        selected_scoring = dense_scoring if use_dense else sparse_scoring

    scoring = {
        'mode': 'sparse_dense_router_support_mc',
        'selected': selected_label,
        'router_decision': router_decision,
        'router_thresholds': {
            'dense_score_min': dense_score_min,
            'dense_margin_min': dense_margin_min,
            'sparse_margin_max': sparse_margin_max,
            'margin_gap_min': margin_gap_min,
        },
        'scores': dict(selected_scoring.get('scores', {})),
        'supports': list(selected_scoring.get('supports', [])),
        'choice_supports': dict(selected_scoring.get('choice_supports', {})),
        'dense_shortlist': list(dense_scoring.get('dense_shortlist', [])),
        'dense_model': dense_scoring.get('dense_model'),
        'dense_device': dense_scoring.get('dense_device'),
        'dense_candidate_count': dense_scoring.get('dense_candidate_count'),
        'dense_top_k': dense_scoring.get('dense_top_k'),
        'sparse_mode': sparse_scoring.get('mode'),
        'dense_mode': dense_scoring.get('mode'),
        'sparse_margin': round(float(sparse_margin), 6),
        'dense_margin': round(float(dense_margin), 6),
        'dense_top_score': round(float(dense_top_score), 6),
        'dense_score_source': 'rerank' if support_reranker is not None else 'dense_shortlist',
        'sparse_selected': sparse_label,
        'dense_selected': dense_label,
    }
    if 'selected_bank' in sparse_scoring:
        scoring['selected_bank'] = sparse_scoring.get('selected_bank')
        scoring['bank_results'] = dict(sparse_scoring.get('bank_results', {}))
    return selected_label, scoring


def _score_margin(scores: dict[str, float]) -> float:
    ordered = sorted(scores.values(), reverse=True)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    return float(ordered[0] - ordered[1])


def _scoring_margin(scoring: dict[str, object] | None) -> float:
    scores = dict(scoring.get('scores', {})) if scoring else {}
    numeric = {str(key): float(value) for key, value in scores.items()}
    return _score_margin(numeric)


def _dense_shortlist_top_score(scoring: dict[str, object] | None) -> float:
    if not scoring:
        return 0.0
    shortlist = list(scoring.get('dense_shortlist', []))
    if not shortlist:
        return 0.0
    return float(dict(shortlist[0]).get('score', 0.0))


def _rerank_top_score(scoring: dict[str, object] | None) -> float:
    if not scoring:
        return 0.0
    rerank_matches = list(scoring.get('rerank_matches', []))
    if not rerank_matches:
        return 0.0
    return float(dict(rerank_matches[0]).get('score', 0.0))


def _load_support_resource(path: str | Path) -> dict[str, object]:
    root = Path(path)
    manifest_path = (root / 'manifest.json') if root.is_dir() else root
    if manifest_path.suffix == '.json' and manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding='utf-8'))
        if payload.get('banks'):
            banks: dict[str, dict[str, object]] = {}
            for bank in payload['banks']:
                bank_path = Path(bank['path'])
                if not bank_path.is_absolute():
                    bank_path = (manifest_path.parent / bank_path)
                examples = load_support_examples(bank_path)
                banks[str(bank['name'])] = {
                    'path': str(bank_path),
                    'examples': examples,
                    'index': _build_sparse_support_index(examples),
                }
            primary_bank = str(payload.get('primary_bank') or payload['banks'][0]['name'])
            return {
                'kind': 'banks',
                'manifest': payload,
                'banks': banks,
                'primary_examples': list(banks[primary_bank]['examples']),
            }
    examples = load_support_examples(root)
    return {
        'kind': 'examples',
        'examples': examples,
        'index': _build_sparse_support_index(examples),
    }


def _predict_multiple_choice_support_ensemble(
    *,
    prompt: str,
    choices: tuple[tuple[str, str], ...],
    support_resource: dict[str, object] | None,
    category_hint: str | None,
    category_gated: bool,
    top_k: int = 8,
) -> tuple[str, dict[str, object]] | None:
    if support_resource is None or support_resource.get('kind') != 'banks':
        return None
    manifest = dict(support_resource['manifest'])
    banks = dict(support_resource['banks'])
    threshold = float(manifest.get('margin_threshold', 0.0))
    strategy = str(manifest.get('strategy', 'margin_gate'))
    primary_bank = str(manifest.get('primary_bank') or next(iter(banks)))
    bank_order = [primary_bank] + [name for name in banks if name != primary_bank]

    selected_name: str | None = None
    selected_payload: tuple[str, dict[str, object]] | None = None
    selected_margin = float('-inf')
    bank_results: dict[str, dict[str, object]] = {}

    for bank_name in bank_order:
        bank = banks[bank_name]
        prediction = _predict_multiple_choice_hybrid_from_support(
            prompt=prompt,
            choices=choices,
            support_examples=bank['examples'],
            support_index=bank['index'],
            category_hint=category_hint,
            category_gated=category_gated,
            top_k=top_k,
        )
        if prediction is None:
            bank_results[bank_name] = {'selected': None, 'margin': None, 'path': str(bank['path'])}
            continue
        label, scoring = prediction
        margin = _score_margin({str(key): float(value) for key, value in dict(scoring.get('scores', {})).items()})
        bank_results[bank_name] = {
            'selected': label,
            'margin': round(float(margin), 6),
            'path': str(bank['path']),
            'candidate_count': scoring.get('candidate_count'),
        }
        if selected_payload is None:
            selected_name = bank_name
            selected_payload = prediction
            selected_margin = margin
            continue
        if strategy == 'margin_gate' and margin >= selected_margin + threshold:
            selected_name = bank_name
            selected_payload = prediction
            selected_margin = margin

    if selected_payload is None or selected_name is None:
        return None
    label, scoring = selected_payload
    return label, {
        'mode': 'hybrid_support_ensemble',
        'strategy': strategy,
        'margin_threshold': threshold,
        'selected_bank': selected_name,
        'bank_results': bank_results,
        'scores': scoring.get('scores', {}),
        'selected': scoring.get('selected', label),
        'supports': scoring.get('supports', []),
        'choice_supports': scoring.get('choice_supports', {}),
        'candidate_count': scoring.get('candidate_count'),
    }

def _load_cached_rows(parts: tuple[str, ...], filename: str, limit: int | None, offset: int = 0) -> list[dict[str, object]] | None:
    if pa_ipc is None:
        return None
    base = HF_DATASETS_CACHE.joinpath(*parts)
    if not base.exists():
        return None
    for candidate in sorted(base.glob(f"*/{filename}")):
        with candidate.open("rb") as handle:
            table = pa_ipc.open_stream(handle).read_all()
        rows = table.to_pylist()
        start = max(offset, 0)
        end = None if limit is None else start + max(limit, 0)
        rows = rows[start:end]
        return rows
    return None


def load_gsm8k_tasks(*, split: str = "test", limit: int | None = 50, offset: int = 0) -> list[ExternalBenchmarkTask]:
    rows = _load_cached_rows(("gsm8k", "main", "0.0.0"), f"gsm8k-{split}.arrow", limit, offset)
    if rows is None:
        if not DATASETS_AVAILABLE:
            raise RuntimeError("datasets is required for external benchmarks.")
        rows = list(load_dataset("gsm8k", "main", split=_slice_split(split, limit, offset)))
    tasks: list[ExternalBenchmarkTask] = []
    for idx, row in enumerate(rows):
        tasks.append(
            ExternalBenchmarkTask(
                benchmark="gsm8k",
                task_id=str(idx),
                category="math",
                prompt=f"{row['question'].strip()}\n\nReply with only the final answer.",
                metric="exact_match",
                expected=extract_gsm8k_answer(str(row["answer"])),
            )
        )
    return tasks


def load_arc_challenge_tasks(*, split: str = "validation", limit: int | None = 50, offset: int = 0) -> list[ExternalBenchmarkTask]:
    rows = _load_cached_rows(("allenai___ai2_arc", "ARC-Challenge", "0.0.0"), f"ai2_arc-{split}.arrow", limit, offset)
    if rows is None:
        if not DATASETS_AVAILABLE:
            raise RuntimeError("datasets is required for external benchmarks.")
        rows = list(load_dataset("allenai/ai2_arc", "ARC-Challenge", split=_slice_split(split, limit, offset)))
    tasks: list[ExternalBenchmarkTask] = []
    for row in rows:
        labels = tuple(str(item) for item in row["choices"]["label"])
        texts = tuple(str(item) for item in row["choices"]["text"])
        choices = tuple(zip(labels, texts, strict=True))
        answer_key = str(row["answerKey"])
        answer_text = dict(choices)[answer_key]
        tasks.append(
            ExternalBenchmarkTask(
                benchmark="arc-challenge",
                task_id=str(row["id"]),
                category="science",
                prompt=format_arc_prompt(str(row["question"]), choices),
                metric="multiple_choice",
                expected=answer_key,
                choices=choices,
                expected_text=answer_text,
            )
        )
    return tasks


def load_piqa_tasks(*, split: str = "validation", limit: int | None = 50, offset: int = 0) -> list[ExternalBenchmarkTask]:
    if not DATASETS_AVAILABLE:
        raise RuntimeError("datasets is required for external benchmarks.")
    dataset = load_dataset("piqa", split=_slice_split(split, limit, offset))
    tasks: list[ExternalBenchmarkTask] = []
    for idx, row in enumerate(dataset):
        choices = (("A", str(row["sol1"])), ("B", str(row["sol2"])))
        answer_key = "A" if int(row["label"]) == 0 else "B"
        answer_text = dict(choices)[answer_key]
        tasks.append(
            ExternalBenchmarkTask(
                benchmark="piqa",
                task_id=str(idx),
                category="commonsense",
                prompt=format_arc_prompt(str(row["goal"]), choices),
                metric="multiple_choice",
                expected=answer_key,
                choices=choices,
                expected_text=answer_text,
            )
        )
    return tasks


def load_external_benchmark_tasks(
    benchmark: str,
    *,
    split: str | None = None,
    limit: int | None = 50,
    offset: int = 0,
) -> list[ExternalBenchmarkTask]:
    key = benchmark.lower()
    if key == "gsm8k":
        return load_gsm8k_tasks(split=split or "test", limit=limit, offset=offset)
    if key in {"arc", "arc-challenge"}:
        return load_arc_challenge_tasks(split=split or "validation", limit=limit, offset=offset)
    if key == "piqa":
        return load_piqa_tasks(split=split or "validation", limit=limit, offset=offset)
    raise ValueError(f"unsupported external benchmark: {benchmark}")


def _match_task(task: ExternalBenchmarkTask, completion: str) -> bool:
    if task.metric == "multiple_choice":
        return _match_multiple_choice(task, completion)
    return _match_exact(task, completion)


def _prepare_retrieval_prompt_for_task(
    model: Any,
    tokenizer: Any,
    *,
    prompt: str,
    support_examples: list[SupportExample] | None,
    retrieval_mode: str,
    category_hint: str | None,
    category_gated: bool,
    nearest_threshold: float,
    nearest_margin: float,
    support_top_k: int,
) -> tuple[str | None, dict[str, object]]:
    base_prompt = prepare_retrieval_prompt(prompt, tokenizer=tokenizer, block_size=model.config.block_size)
    if retrieval_mode == "baseline":
        return None, {
            **base_prompt,
            "enabled": False,
            "mode": "baseline",
            "references": list(base_prompt.get("references", [])),
        }

    if retrieval_mode == "direct":
        direct = lookup_support_answer(
            prompt,
            support_examples=support_examples,
            category_hint=category_hint,
            category_gated=category_gated,
        )
        if direct is not None:
            return str(direct["response"]), {
                **base_prompt,
                "enabled": True,
                "mode": "direct",
                "direct_match": direct,
                "references": [direct["reference"]],
            }
        return None, {
            **base_prompt,
            "enabled": False,
            "mode": "direct",
            "direct_match": None,
            "references": [],
        }

    if retrieval_mode == "nearest":
        nearest = lookup_support_answer_nearest(
            prompt,
            support_examples=support_examples,
            category_hint=category_hint,
            category_gated=category_gated,
            min_score=nearest_threshold,
            min_margin=nearest_margin,
        )
        if nearest is not None:
            return str(nearest["response"]), {
                **base_prompt,
                "enabled": True,
                "mode": "nearest",
                "nearest_match": nearest,
                "references": [nearest["reference"]],
                "nearest_threshold": nearest_threshold,
                "nearest_margin": nearest_margin,
            }
        return None, {
            **base_prompt,
            "enabled": False,
            "mode": "nearest",
            "nearest_match": None,
            "references": [],
            "nearest_threshold": nearest_threshold,
            "nearest_margin": nearest_margin,
        }

    if retrieval_mode == "prompt_support":
        retrieval = prepare_retrieval_prompt(
            prompt,
            tokenizer=tokenizer,
            block_size=model.config.block_size,
            support_examples=support_examples,
            top_k=support_top_k,
            category_hint=category_hint,
            category_gated=category_gated,
        )
        retrieval["mode"] = "prompt_support"
        return None, retrieval

    raise ValueError(f"unknown external retrieval mode: {retrieval_mode}")


@torch.no_grad()
def _score_completion_ids(
    model: Any,
    device: str,
    *,
    prompt_ids: list[int],
    candidate_ids: list[int],
) -> float:
    if not candidate_ids:
        return float("-inf")
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    total = 0.0
    for token_id in candidate_ids:
        idx_cond = idx[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
        total += float(log_probs[0, token_id].item())
        next_token = torch.tensor([[token_id]], dtype=torch.long, device=device)
        idx = torch.cat((idx, next_token), dim=1)
    return total / len(candidate_ids)


@torch.no_grad()
def _predict_multiple_choice_label(
    model: Any,
    tokenizer: Any,
    device: str,
    *,
    prompt: str,
    choices: tuple[tuple[str, str], ...],
) -> tuple[str, dict[str, float]]:
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    scores: dict[str, float] = {}
    single_token: list[tuple[str, int]] = []
    multi_token: list[tuple[str, list[int]]] = []
    for label, _choice_text in choices:
        candidate_ids = tokenizer.encode(label)
        if len(candidate_ids) == 1:
            single_token.append((label, candidate_ids[0]))
        else:
            multi_token.append((label, candidate_ids))

    if single_token:
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        idx_cond = idx[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)[0]
        for label, token_id in single_token:
            scores[label] = round(float(log_probs[token_id].item()), 6)

    for label, candidate_ids in multi_token:
        scores[label] = round(
            _score_completion_ids(
                model,
                device,
                prompt_ids=prompt_ids,
                candidate_ids=candidate_ids,
            ),
            6,
        )
    best_label = max(scores.items(), key=lambda item: item[1])[0]
    return best_label, scores


@torch.no_grad()
def evaluate_external_benchmark(
    checkpoint_path: str | Path,
    *,
    benchmark: str,
    split: str | None = None,
    limit: int | None = 50,
    offset: int = 0,
    requested_device: str = "cuda",
    max_new_tokens: int = 48,
    support_corpus: str | Path | None = None,
    dense_support_corpus: str | Path | None = None,
    retrieval_mode: str = "baseline",
    category_gated: bool = True,
    nearest_threshold: float = 0.45,
    nearest_margin: float = 0.0,
    support_top_k: int = 2,
    dense_encoder_model: str = "BAAI/bge-small-en-v1.5",
    dense_encoder_device: str = "cpu",
    dense_candidate_top_k: int = 64,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    reranker_device: str = "cpu",
    rerank_top_k: int = 12,
    router_dense_score_min: float = 0.7,
    router_dense_margin_min: float = 0.0,
    router_sparse_margin_max: float = 0.015,
    router_margin_gap_min: float = 0.0,
    repeat_override: int | None = None,
) -> dict[str, object]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for evaluation.")

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    config = ExperimentConfig.from_dict(checkpoint["config"])
    tokenizer = load_tokenizer(config.tokenizer)
    model = build_model(config.model, tokenizer.vocab_size)
    model.load_state_dict(checkpoint["model"])

    device, warnings = _resolve_device(requested_device)
    model = model.to(device)
    model.eval()

    # Test-time compute scaling: override recurrent loop count
    if repeat_override is not None:
        model._inference_repeat_override = repeat_override

    tasks = load_external_benchmark_tasks(benchmark, split=split, limit=limit, offset=offset)
    support_resource = _load_support_resource(support_corpus) if support_corpus else None
    if support_resource is None:
        support_examples = None
        support_index = None
    elif support_resource.get("kind") == "banks":
        support_examples = list(support_resource.get("primary_examples", []))
        primary_bank = str(dict(support_resource["manifest"]).get("primary_bank") or next(iter(dict(support_resource["banks"]))))
        support_index = dict(support_resource["banks"])[primary_bank]["index"]
    else:
        support_examples = list(support_resource.get("examples", []))
        support_index = support_resource.get("index")

    dense_support_resource = _load_support_resource(dense_support_corpus) if dense_support_corpus else support_resource
    if dense_support_resource is None:
        dense_support_examples = None
    elif dense_support_resource.get("kind") == "banks":
        dense_support_examples = list(dense_support_resource.get("primary_examples", []))
    else:
        dense_support_examples = list(dense_support_resource.get("examples", []))

    dense_retriever: DenseSupportRetriever | None = None
    support_reranker: SupportReranker | None = None
    if retrieval_mode in {"dense_hybrid_support_mc", "dense_rerank_support_mc", "sparse_dense_router_support_mc", "sparse_dense_rerank_router_support_mc"}:
        if not dense_support_examples:
            raise RuntimeError(f"{retrieval_mode} requires dense support examples")
        dense_retriever = build_dense_support_retriever(
            dense_support_examples,
            model_name=dense_encoder_model,
            device=dense_encoder_device,
        )
    if retrieval_mode in {"dense_rerank_support_mc", "sparse_dense_rerank_router_support_mc"}:
        support_reranker = build_support_reranker(
            model_name=reranker_model,
            device=reranker_device,
        )

    results: list[ExternalBenchmarkResult] = []
    for task in tasks:
        scoring: dict[str, object] | None = None
        if task.metric == "multiple_choice":
            if retrieval_mode == "model_retrieval_ensemble":
                support_prediction = _predict_model_retrieval_ensemble(
                    model,
                    tokenizer,
                    device,
                    prompt=task.prompt,
                    choices=task.choices,
                    support_examples=support_examples,
                    support_index=support_index,
                    category_hint=task.category,
                    category_gated=category_gated,
                )
                if support_prediction is not None:
                    completion, scoring = support_prediction
                    retrieval = {
                        "enabled": True,
                        "mode": retrieval_mode,
                        "references": list(scoring.get("supports", [])),
                        "category_hint": task.category,
                    }
                else:
                    retrieval = {"enabled": False, "mode": retrieval_mode, "references": [], "category_hint": task.category}
                    completion, label_scores = _predict_multiple_choice_label(
                        model, tokenizer, device, prompt=task.prompt, choices=task.choices,
                    )
                    scoring = {"mode": "label_logprob", "scores": label_scores, "selected": completion}
            elif retrieval_mode in {"support_mc", "hybrid_support_mc", "hybrid_support_ensemble", "dense_hybrid_support_mc", "dense_rerank_support_mc", "sparse_dense_router_support_mc", "sparse_dense_rerank_router_support_mc"}:
                if retrieval_mode == "sparse_dense_rerank_router_support_mc":
                    support_prediction = _predict_multiple_choice_sparse_dense_router(
                        prompt=task.prompt,
                        choices=task.choices,
                        support_resource=support_resource,
                        dense_retriever=dense_retriever,
                        support_reranker=support_reranker,
                        category_hint=task.category,
                        category_gated=category_gated,
                        shortlist_top_k=dense_candidate_top_k,
                        rerank_top_k=rerank_top_k,
                        dense_score_min=router_dense_score_min,
                        dense_margin_min=router_dense_margin_min,
                        sparse_margin_max=router_sparse_margin_max,
                        margin_gap_min=router_margin_gap_min,
                    )
                elif retrieval_mode == "sparse_dense_router_support_mc":
                    support_prediction = _predict_multiple_choice_sparse_dense_router(
                        prompt=task.prompt,
                        choices=task.choices,
                        support_resource=support_resource,
                        dense_retriever=dense_retriever,
                        category_hint=task.category,
                        category_gated=category_gated,
                        shortlist_top_k=dense_candidate_top_k,
                        dense_score_min=router_dense_score_min,
                        dense_margin_min=router_dense_margin_min,
                        sparse_margin_max=router_sparse_margin_max,
                        margin_gap_min=router_margin_gap_min,
                    )
                elif retrieval_mode == "dense_rerank_support_mc":
                    support_prediction = _predict_multiple_choice_dense_rerank_from_support(
                        prompt=task.prompt,
                        choices=task.choices,
                        dense_retriever=dense_retriever,
                        support_reranker=support_reranker,
                        category_hint=task.category,
                        category_gated=category_gated,
                        shortlist_top_k=dense_candidate_top_k,
                        rerank_top_k=rerank_top_k,
                    )
                elif retrieval_mode == "dense_hybrid_support_mc":
                    support_prediction = _predict_multiple_choice_dense_hybrid_from_support(
                        prompt=task.prompt,
                        choices=task.choices,
                        dense_retriever=dense_retriever,
                        category_hint=task.category,
                        category_gated=category_gated,
                        shortlist_top_k=dense_candidate_top_k,
                    )
                elif retrieval_mode == "hybrid_support_ensemble":
                    support_prediction = _predict_multiple_choice_support_ensemble(
                        prompt=task.prompt,
                        choices=task.choices,
                        support_resource=support_resource,
                        category_hint=task.category,
                        category_gated=category_gated,
                    )
                elif retrieval_mode == "hybrid_support_mc":
                    support_prediction = _predict_multiple_choice_hybrid_from_support(
                        prompt=task.prompt,
                        choices=task.choices,
                        support_examples=support_examples,
                        support_index=support_index,
                        category_hint=task.category,
                        category_gated=category_gated,
                    )
                else:
                    support_prediction = _predict_multiple_choice_from_support(
                        prompt=task.prompt,
                        choices=task.choices,
                        support_examples=support_examples,
                        support_index=support_index,
                        category_hint=task.category,
                        category_gated=category_gated,
                    )
                if support_prediction is not None:
                    completion, scoring = support_prediction
                    references = list(scoring.get("dense_shortlist", scoring.get("supports", [])))
                    retrieval = {
                        "enabled": True,
                        "mode": retrieval_mode,
                        "references": references,
                        "category_hint": task.category,
                    }
                    if "dense_shortlist" in scoring:
                        retrieval["rerank_supports"] = list(scoring.get("supports", []))
                        retrieval["dense_model"] = scoring.get("dense_model")
                else:
                    retrieval = {
                        "enabled": False,
                        "mode": retrieval_mode,
                        "references": [],
                        "category_hint": task.category,
                    }
                    completion, label_scores = _predict_multiple_choice_label(
                        model,
                        tokenizer,
                        device,
                        prompt=task.prompt,
                        choices=task.choices,
                    )
                    scoring = {"mode": "label_logprob", "scores": label_scores, "selected": completion}
            else:
                direct_completion, retrieval = _prepare_retrieval_prompt_for_task(
                    model,
                    tokenizer,
                    prompt=task.prompt,
                    support_examples=support_examples,
                    retrieval_mode=retrieval_mode,
                    category_hint=task.category,
                    category_gated=category_gated,
                    nearest_threshold=nearest_threshold,
                    nearest_margin=nearest_margin,
                    support_top_k=support_top_k,
                )
                if direct_completion is not None:
                    completion = direct_completion
                    scoring = {"mode": "retrieval", "selected": completion}
                else:
                    completion, label_scores = _predict_multiple_choice_label(
                        model,
                        tokenizer,
                        device,
                        prompt=str(retrieval["prompt"]),
                        choices=task.choices,
                    )
                    scoring = {"mode": "label_logprob", "scores": label_scores, "selected": completion}
        else:
            completion, retrieval = _generate_completion(
                model,
                tokenizer,
                device,
                prompt=task.prompt,
                max_new_tokens=max_new_tokens,
                support_examples=support_examples,
                retrieval_mode=retrieval_mode,
                category_hint=task.category,
                category_gated=category_gated,
                nearest_threshold=nearest_threshold,
                nearest_margin=nearest_margin,
                support_top_k=support_top_k,
            )
        results.append(
            ExternalBenchmarkResult(
                benchmark=task.benchmark,
                task_id=task.task_id,
                category=task.category,
                prompt=task.prompt,
                expected=task.expected,
                completion=completion,
                matched=_match_task(task, completion),
                retrieval=retrieval,
                expected_text=task.expected_text,
                choices=task.choices,
                scoring=scoring,
            )
        )

    correct = sum(1 for item in results if item.matched)
    total = len(results)
    by_category: dict[str, dict[str, object]] = {}
    for category in sorted({item.category for item in results}):
        subset = [item for item in results if item.category == category]
        subset_correct = sum(1 for item in subset if item.matched)
        by_category[category] = {
            "correct": subset_correct,
            "total": len(subset),
            "accuracy": round(subset_correct / max(len(subset), 1), 3),
        }

    return {
        "benchmark": benchmark,
        "split": split,
        "limit": limit,
        "offset": offset,
        "checkpoint": str(checkpoint_path),
        "config_name": config.name,
        "requested_device": requested_device,
        "device_used": device,
        "warnings": warnings,
        "support_corpus": str(support_corpus) if support_corpus else None,
        "dense_support_corpus": str(dense_support_corpus) if dense_support_corpus else None,
        "support_index_rows": len(support_index.rows) if support_index is not None else 0,
        "retrieval_mode": retrieval_mode,
        "category_gated": category_gated,
        "nearest_threshold": nearest_threshold,
        "nearest_margin": nearest_margin,
        "dense_encoder_model": dense_encoder_model if retrieval_mode in {"dense_hybrid_support_mc", "dense_rerank_support_mc", "sparse_dense_router_support_mc", "sparse_dense_rerank_router_support_mc"} else None,
        "dense_encoder_device": dense_encoder_device if retrieval_mode in {"dense_hybrid_support_mc", "dense_rerank_support_mc", "sparse_dense_router_support_mc", "sparse_dense_rerank_router_support_mc"} else None,
        "dense_candidate_top_k": dense_candidate_top_k if retrieval_mode in {"dense_hybrid_support_mc", "dense_rerank_support_mc", "sparse_dense_router_support_mc", "sparse_dense_rerank_router_support_mc"} else None,
        "reranker_model": reranker_model if retrieval_mode in {"dense_rerank_support_mc", "sparse_dense_rerank_router_support_mc"} else None,
        "reranker_device": reranker_device if retrieval_mode in {"dense_rerank_support_mc", "sparse_dense_rerank_router_support_mc"} else None,
        "rerank_top_k": rerank_top_k if retrieval_mode in {"dense_rerank_support_mc", "sparse_dense_rerank_router_support_mc"} else None,
        "router_dense_score_min": router_dense_score_min if retrieval_mode in {"sparse_dense_router_support_mc", "sparse_dense_rerank_router_support_mc"} else None,
        "router_dense_margin_min": router_dense_margin_min if retrieval_mode in {"sparse_dense_router_support_mc", "sparse_dense_rerank_router_support_mc"} else None,
        "router_sparse_margin_max": router_sparse_margin_max if retrieval_mode in {"sparse_dense_router_support_mc", "sparse_dense_rerank_router_support_mc"} else None,
        "router_margin_gap_min": router_margin_gap_min if retrieval_mode in {"sparse_dense_router_support_mc", "sparse_dense_rerank_router_support_mc"} else None,
        "repeat_override": repeat_override,
        "correct": correct,
        "total": total,
        "accuracy": round(correct / max(total, 1), 3),
        "by_category": by_category,
        "results": [asdict(item) for item in results],
    }


def summarize_external_benchmark(payload: dict[str, object]) -> str:
    return json.dumps(
        {
            "benchmark": payload["benchmark"],
            "split": payload["split"],
            "limit": payload["limit"],
            "retrieval_mode": payload["retrieval_mode"],
            "accuracy": payload["accuracy"],
            "correct": payload["correct"],
            "total": payload["total"],
            "support_corpus": payload["support_corpus"],
        },
        indent=2,
    )
