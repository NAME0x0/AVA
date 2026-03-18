from __future__ import annotations

import json
from pathlib import Path
from random import Random
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POSTTRAIN_SOURCE_LIMITS: dict[str, int] = {
    "teacher_distill": 103,
    "public_benchmark": 6000,
    "public_science": 6000,
    "math_reasoning": 102,
    "general_sft": 59,
    "guardrails": 26,
    "tool_repair": 23,
    "public_reasoning": 103,
}
POSTTRAIN_SOURCE_PATHS: dict[str, Path] = {
    "teacher_distill": REPO_ROOT / "corpora" / "teacher_distill_v1" / "examples.jsonl",
    "public_benchmark": REPO_ROOT / "corpora" / "public_benchmark_distill_v1" / "examples.jsonl",
    "public_science": REPO_ROOT / "corpora" / "public_science_support_v1" / "examples.jsonl",
    "math_reasoning": REPO_ROOT / "corpora" / "gsm8k_reasoning_support_v1" / "examples.jsonl",
    "general_sft": REPO_ROOT / "corpora" / "general_sft" / "examples.jsonl",
    "guardrails": REPO_ROOT / "corpora" / "multiview_guardrail_v1" / "examples.jsonl",
    "tool_repair": REPO_ROOT / "corpora" / "tool_repair_nano_v1" / "examples.jsonl",
    "public_reasoning": REPO_ROOT / "corpora" / "public_reasoning_patch_v1" / "examples.jsonl",
}


def _join_records(records: list[str]) -> str:
    return "\n\n<sep>\n\n".join(record.strip() for record in records if record.strip())


def _tiny_stories_to_text(rows: list[dict[str, object]]) -> list[str]:
    return [str(row.get("text", "")).strip() for row in rows if str(row.get("text", "")).strip()]


def _gsm8k_to_text(rows: list[dict[str, object]]) -> list[str]:
    records: list[str] = []
    for row in rows:
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not question or not answer:
            continue
        records.append(f"Math problem:\n{question}\n\nWorked solution:\n{answer}")
    return records


def _sciq_to_text(rows: list[dict[str, object]]) -> list[str]:
    records: list[str] = []
    for row in rows:
        support = str(row.get("support", "")).strip()
        question = str(row.get("question", "")).strip()
        answer = str(row.get("correct_answer", "")).strip()
        if support:
            records.append(f"Science note:\n{support}")
        if question and answer:
            records.append(f"Science question:\n{question}\n\nAnswer:\n{answer}")
    return records


def _openbookqa_to_text(rows: list[dict[str, object]]) -> list[str]:
    records: list[str] = []
    for row in rows:
        question = str(row.get("question_stem", "")).strip()
        choices = row.get("choices") or {}
        labels = list(choices.get("label") or [])
        texts = list(choices.get("text") or [])
        answer_key = str(row.get("answerKey", "")).strip()
        answer_text = ""
        for label, choice in zip(labels, texts, strict=False):
            if str(label) == answer_key:
                answer_text = str(choice)
                break
        options = "\n".join(
            f"{label}. {choice}" for label, choice in zip(labels, texts, strict=False)
        )
        if question and options:
            records.append(
                f"Science multiple choice:\n{question}\n\nOptions:\n{options}\n\nCorrect answer:\n{answer_text or answer_key}"
            )
    return records


def _mbpp_to_text(rows: list[dict[str, object]]) -> list[str]:
    records: list[str] = []
    for row in rows:
        task = str(row.get("text", "")).strip()
        code = str(row.get("code", "")).strip()
        if not task or not code:
            continue
        records.append(f"Programming task:\n{task}\n\nPython solution:\n{code}")
    return records


def _rows(dataset: Any) -> list[dict[str, object]]:
    return [dict(row) for row in dataset]


def _read_jsonl_examples(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _sample_examples(
    rows: list[dict[str, object]], limit: int | None, *, seed_value: int
) -> list[dict[str, object]]:
    if limit is None or limit >= len(rows):
        return list(rows)
    rng = Random(seed_value)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    selected = sorted(indices[:limit])
    return [rows[index] for index in selected]


def _write_jsonl_examples(path: str | Path, rows: list[dict[str, object]]) -> None:
    content = "\n".join(json.dumps(row) for row in rows) + "\n"
    Path(path).write_text(content, encoding="utf-8")


def materialize_open_mix_corpus(
    root: str | Path,
    *,
    english_limit: int = 20_000,
    math_limit: int = 4_000,
    science_limit: int = 4_000,
    code_limit: int = 374,
) -> dict[str, object]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("datasets is required for open-mix corpus materialization") from exc

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    tiny_rows = _rows(load_dataset("roneneldan/TinyStories", split=f"train[:{english_limit}]"))
    gsm8k_rows = _rows(load_dataset("gsm8k", "main", split=f"train[:{math_limit}]"))
    sciq_rows = _rows(load_dataset("sciq", split=f"train[:{science_limit}]"))
    openbook_limit = max(1, science_limit // 2)
    openbook_rows = _rows(load_dataset("allenai/openbookqa", split=f"train[:{openbook_limit}]"))
    mbpp_rows = _rows(load_dataset("mbpp", split=f"train[:{code_limit}]"))

    english_records = _tiny_stories_to_text(tiny_rows)
    math_records = _gsm8k_to_text(gsm8k_rows)
    science_records = _sciq_to_text(sciq_rows) + _openbookqa_to_text(openbook_rows)
    code_records = _mbpp_to_text(mbpp_rows)

    files = {
        "english.txt": _join_records(english_records),
        "math.txt": _join_records(math_records),
        "science.txt": _join_records(science_records),
        "code.txt": _join_records(code_records),
    }
    for name, content in files.items():
        (root_path / name).write_text(content, encoding="utf-8")

    manifest = {
        "curriculum": "ava_v2_open_mix",
        "english_limit": english_limit,
        "math_limit": math_limit,
        "science_limit": science_limit,
        "code_limit": code_limit,
        "english_records": len(english_records),
        "math_records": len(math_records),
        "science_records": len(science_records),
        "code_records": len(code_records),
        "files": list(files),
    }
    (root_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (root_path / "README.md").write_text(
        "# AVA-v2 Open Mix Corpus\n\n"
        "A mixed raw-text continuation corpus built from TinyStories, GSM8K train, SciQ train, OpenBookQA train, and MBPP train.\n",
        encoding="utf-8",
    )
    return manifest


def materialize_posttrain_mix_corpus(
    root: str | Path,
    *,
    source_limits: dict[str, int] | None = None,
    source_repeats: dict[str, int] | None = None,
    seed_value: int = 1337,
) -> dict[str, object]:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    merged_rows: list[dict[str, object]] = []
    source_limits = source_limits or DEFAULT_POSTTRAIN_SOURCE_LIMITS
    source_repeats = source_repeats or {name: 1 for name in DEFAULT_POSTTRAIN_SOURCE_LIMITS}
    source_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}

    for offset, source_name in enumerate(DEFAULT_POSTTRAIN_SOURCE_LIMITS):
        path = POSTTRAIN_SOURCE_PATHS[source_name]
        rows = _read_jsonl_examples(path)
        sampled = _sample_examples(
            rows, source_limits.get(source_name), seed_value=seed_value + offset
        )
        repeat = max(1, int(source_repeats.get(source_name, 1)))
        repeated_rows = sampled * repeat
        source_counts[source_name] = len(repeated_rows)
        for row in repeated_rows:
            kind = str(row.get("kind", "unknown"))
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
        merged_rows.extend(repeated_rows)

    _write_jsonl_examples(root_path / "examples.jsonl", merged_rows)
    manifest = {
        "curriculum": "ava_v2_posttrain_mix",
        "seed": seed_value,
        "total_examples": len(merged_rows),
        "source_limits": source_limits,
        "source_repeats": source_repeats,
        "source_counts": source_counts,
        "by_kind": kind_counts,
        "examples_path": str(root_path / "examples.jsonl"),
    }
    (root_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (root_path / "README.md").write_text(
        "# AVA-v2 Post-Train Mix\n\n"
        "A deterministic supervised post-training mixture assembled from AVA teacher, public benchmark, science, math-reasoning, tool, compliance, and general instruction packets.\n",
        encoding="utf-8",
    )
    return manifest
