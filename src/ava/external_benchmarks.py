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


def _slice_split(split: str, limit: int | None) -> str:
    if limit is None or "[" in split:
        return split
    return f"{split}[:{limit}]"


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


def _build_sparse_support_rows(examples: list[SupportExample]) -> tuple[list[dict[str, object]], dict[str, float]]:
    document_frequency: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    for example in examples:
        tokens = _support_lookup_tokens(example.prompt)
        rows.append({
            "tokens": tokens,
            "response": example.response,
            "reference": {
                "prompt": example.prompt,
                "response": example.response,
                "category": example.category,
                "kind": example.kind,
                "source_path": example.source_path,
            },
        })
        for token in set(tokens):
            document_frequency[token] += 1
    total_documents = len(rows)
    idf = {
        token: log((total_documents + 1) / (count + 1)) + 1.0
        for token, count in document_frequency.items()
    }
    return rows, idf


def _sparse_support_score(query_tokens: list[str], support_tokens: list[str], idf: dict[str, float]) -> float:
    query_counts = Counter(query_tokens)
    support_counts = Counter(support_tokens)
    shared = set(query_counts) & set(support_counts)
    if not shared:
        return 0.0
    numerator = sum(min(query_counts[token], support_counts[token]) * idf.get(token, 1.0) for token in shared)
    denominator = sum(query_counts[token] * idf.get(token, 1.0) for token in query_counts)
    return numerator / max(denominator, 1e-9)


def _support_response_choice_score(response: str, choice_text: str) -> float:
    response_tokens = set(_support_lookup_tokens(response))
    choice_tokens = set(_support_lookup_tokens(choice_text))
    if not response_tokens or not choice_tokens:
        return 0.0
    return len(response_tokens & choice_tokens) / len(response_tokens | choice_tokens)


def _predict_multiple_choice_from_support(
    *,
    prompt: str,
    choices: tuple[tuple[str, str], ...],
    support_examples: list[SupportExample] | None,
    category_hint: str | None,
    category_gated: bool,
    top_k: int = 8,
) -> tuple[str, dict[str, object]] | None:
    examples = list(support_examples or [])
    if category_gated and category_hint:
        filtered = [item for item in examples if item.category == category_hint]
        if filtered:
            examples = filtered
    if not examples:
        return None

    rows, idf = _build_sparse_support_rows(examples)
    query_tokens = _support_lookup_tokens(prompt)
    scored = sorted(
        ((
            _sparse_support_score(query_tokens, row["tokens"], idf),
            row,
        ) for row in rows),
        key=lambda item: item[0],
        reverse=True,
    )[:top_k]
    choice_scores: Counter[str] = Counter({label: 0.0 for label, _choice_text in choices})
    support_details: list[dict[str, object]] = []
    for score, row in scored:
        support_details.append({"score": round(score, 6), **row["reference"]})
        response = str(row["response"]).strip()
        direct_label = next((label for label, _choice_text in choices if label.upper() == response.upper()), None)
        if direct_label is not None:
            choice_scores[direct_label] += score
            continue
        for label, choice_text in choices:
            choice_scores[label] += score * _support_response_choice_score(response, choice_text)

    if not choice_scores:
        return None
    best_label = max(choice_scores.items(), key=lambda item: item[1])[0]
    return best_label, {
        "mode": "support_mc",
        "scores": {label: round(float(choice_scores[label]), 6) for label, _choice_text in choices},
        "selected": best_label,
        "supports": support_details,
    }


def _load_cached_rows(parts: tuple[str, ...], filename: str, limit: int | None) -> list[dict[str, object]] | None:
    if pa_ipc is None:
        return None
    base = HF_DATASETS_CACHE.joinpath(*parts)
    if not base.exists():
        return None
    for candidate in sorted(base.glob(f"*/{filename}")):
        with candidate.open("rb") as handle:
            table = pa_ipc.open_stream(handle).read_all()
        rows = table.to_pylist()
        if limit is not None:
            rows = rows[:limit]
        return rows
    return None


def load_gsm8k_tasks(*, split: str = "test", limit: int | None = 50) -> list[ExternalBenchmarkTask]:
    rows = _load_cached_rows(("gsm8k", "main", "0.0.0"), f"gsm8k-{split}.arrow", limit)
    if rows is None:
        if not DATASETS_AVAILABLE:
            raise RuntimeError("datasets is required for external benchmarks.")
        rows = list(load_dataset("gsm8k", "main", split=_slice_split(split, limit)))
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


def load_arc_challenge_tasks(*, split: str = "validation", limit: int | None = 50) -> list[ExternalBenchmarkTask]:
    rows = _load_cached_rows(("allenai___ai2_arc", "ARC-Challenge", "0.0.0"), f"ai2_arc-{split}.arrow", limit)
    if rows is None:
        if not DATASETS_AVAILABLE:
            raise RuntimeError("datasets is required for external benchmarks.")
        rows = list(load_dataset("allenai/ai2_arc", "ARC-Challenge", split=_slice_split(split, limit)))
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


def load_piqa_tasks(*, split: str = "validation", limit: int | None = 50) -> list[ExternalBenchmarkTask]:
    if not DATASETS_AVAILABLE:
        raise RuntimeError("datasets is required for external benchmarks.")
    dataset = load_dataset("piqa", split=_slice_split(split, limit))
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
) -> list[ExternalBenchmarkTask]:
    key = benchmark.lower()
    if key == "gsm8k":
        return load_gsm8k_tasks(split=split or "test", limit=limit)
    if key in {"arc", "arc-challenge"}:
        return load_arc_challenge_tasks(split=split or "validation", limit=limit)
    if key == "piqa":
        return load_piqa_tasks(split=split or "validation", limit=limit)
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
    requested_device: str = "cuda",
    max_new_tokens: int = 48,
    support_corpus: str | Path | None = None,
    retrieval_mode: str = "baseline",
    category_gated: bool = True,
    nearest_threshold: float = 0.45,
    nearest_margin: float = 0.0,
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

    tasks = load_external_benchmark_tasks(benchmark, split=split, limit=limit)
    support_examples: list[SupportExample] | None = load_support_examples(support_corpus) if support_corpus else None

    results: list[ExternalBenchmarkResult] = []
    for task in tasks:
        scoring: dict[str, object] | None = None
        if task.metric == "multiple_choice":
            if retrieval_mode == "support_mc":
                support_prediction = _predict_multiple_choice_from_support(
                    prompt=task.prompt,
                    choices=task.choices,
                    support_examples=support_examples,
                    category_hint=task.category,
                    category_gated=category_gated,
                )
                if support_prediction is not None:
                    completion, scoring = support_prediction
                    retrieval = {
                        "enabled": True,
                        "mode": "support_mc",
                        "references": list(scoring.get("supports", [])),
                        "category_hint": task.category,
                    }
                else:
                    retrieval = {
                        "enabled": False,
                        "mode": "support_mc",
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
        "checkpoint": str(checkpoint_path),
        "config_name": config.name,
        "requested_device": requested_device,
        "device_used": device,
        "warnings": warnings,
        "support_corpus": str(support_corpus) if support_corpus else None,
        "retrieval_mode": retrieval_mode,
        "category_gated": category_gated,
        "nearest_threshold": nearest_threshold,
        "nearest_margin": nearest_margin,
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
