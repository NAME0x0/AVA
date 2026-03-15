from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ava.memory import TitansMemory


@dataclass(frozen=True, slots=True)
class SupportExample:
    prompt: str
    response: str
    category: str
    kind: str | None = None
    source_path: str | None = None


def _unwrap_prompt(prompt: str) -> str:
    stripped = prompt.strip()
    if stripped.startswith("Question:"):
        stripped = stripped[len("Question:") :].strip()
        if "\nAnswer:" in stripped:
            stripped = stripped.split("\nAnswer:", 1)[0].strip()
    return stripped


def infer_support_category(prompt: str, response: str = "", kind: str | None = None) -> str:
    if kind:
        return kind

    lowered = f"{prompt} {response}".lower()
    if "return a compact calculator trace" in lowered or "[calc]" in response.lower():
        return "trace"
    if "do not use the calculator tool" in lowered:
        return "no_tool"
    if (
        "delete every file" in lowered
        or "cannot help with deleting files" in lowered
        or "cannot help with hacking" in lowered
        or "cannot help with sending email" in lowered
    ):
        return "boundary"
    if "reply with only" in lowered and not any(
        token in lowered for token in ("rewrite", "summarize", "python", "planet", "force", "solve for x")
    ):
        return "format"
    if any(token in lowered for token in ("bomb", "hack", "malware", "poison")):
        return "refusal"
    if "use the calculator tool" in lowered:
        return "tool"
    if lowered.startswith("rewrite this sentence") or lowered.startswith("summarize this sentence"):
        return "english"
    if any(token in lowered for token in ("python", "len(", "keyword defines a function", "comment in python", "imports a module")):
        return "coding"
    if any(
        token in lowered
        for token in (
            "planet",
            "orbit",
            "sun",
            "photosynthesis",
            "chemical symbol",
            "cell",
            "water turning into vapor",
        )
    ):
        return "science"
    if re.search(r"\d", lowered) or any(
        token in lowered for token in ("solve for x", "square root", "factorial", "pow(", "sqrt(")
    ):
        return "math"
    return "general"


def load_support_examples(path: str | Path) -> list[SupportExample]:
    root = Path(path)
    paths = sorted(root.rglob("*.jsonl")) if root.is_dir() else [root]

    examples: list[SupportExample] = []
    for item in paths:
        for line in item.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            prompt = str(payload["prompt"]).strip()
            response = str(payload["response"]).strip()
            kind = str(payload["kind"]).strip() if payload.get("kind") else None
            category = infer_support_category(prompt, response, kind)
            examples.append(
                SupportExample(
                    prompt=prompt,
                    response=response,
                    category=category,
                    kind=kind,
                    source_path=str(item),
                )
            )
    return examples


def _build_memory(examples: list[SupportExample]) -> TitansMemory:
    memory = TitansMemory(
        max_items=max(len(examples), 1),
        write_surprise_threshold=0.0,
    )
    for example in examples:
        memory.write(
            f"Question: {example.prompt}\nAnswer: {example.response}",
            surprise=1.0,
            metadata=asdict(example),
        )
    return memory


def _serialize_record(record: Any) -> dict[str, object]:
    metadata = dict(record.metadata)
    return {
        "text": record.text,
        "surprise": round(float(record.surprise), 6),
        "accesses": int(record.accesses),
        "metadata": metadata,
    }


def render_base_prompt(prompt: str) -> str:
    return f"Question: {_unwrap_prompt(prompt)}\nAnswer: "


def prepare_retrieval_prompt(
    prompt: str,
    *,
    tokenizer: Any | None = None,
    block_size: int | None = None,
    support_examples: list[SupportExample] | None = None,
    top_k: int = 0,
    category_hint: str | None = None,
    category_gated: bool = True,
) -> dict[str, object]:
    base_prompt = render_base_prompt(prompt)
    query = _unwrap_prompt(prompt)
    selected_examples = list(support_examples or [])
    retrieval_category = category_hint if category_gated else None
    if retrieval_category:
        filtered = [item for item in selected_examples if item.category == retrieval_category]
        if filtered:
            selected_examples = filtered

    if not selected_examples or top_k <= 0:
        prompt_token_count = len(tokenizer.encode(base_prompt, add_bos=True)) if tokenizer else None
        return {
            "enabled": False,
            "query": query,
            "category_hint": retrieval_category,
            "base_prompt": base_prompt,
            "prompt": base_prompt,
            "prompt_token_count": prompt_token_count,
            "dropped_references": 0,
            "references": [],
        }

    memory = _build_memory(selected_examples)
    hits = memory.retrieve(query, top_k=top_k)
    chosen = list(hits)
    dropped_references = 0

    while True:
        blocks: list[str] = []
        for record in chosen:
            metadata = record.metadata
            blocks.append(f"Question: {metadata['prompt']}\nAnswer: {metadata['response']}")
        candidate = "\n\n".join(blocks + [base_prompt]) if blocks else base_prompt
        if tokenizer is None or block_size is None:
            prompt_token_count = None
            break
        prompt_token_count = len(tokenizer.encode(candidate, add_bos=True))
        if prompt_token_count <= block_size or not chosen:
            break
        chosen = chosen[:-1]
        dropped_references += 1

    if not chosen:
        candidate = base_prompt
        prompt_token_count = len(tokenizer.encode(candidate, add_bos=True)) if tokenizer else None

    return {
        "enabled": bool(chosen),
        "query": query,
        "category_hint": retrieval_category,
        "base_prompt": base_prompt,
        "prompt": candidate,
        "prompt_token_count": prompt_token_count,
        "dropped_references": dropped_references,
        "references": [_serialize_record(record) for record in chosen],
    }

def _canonical_lookup_text(prompt: str) -> str:
    text = _unwrap_prompt(prompt).lower().strip()
    text = text.replace("please ", "")
    text = text.replace("think if needed, but ", "")
    text = text.replace("give a compact calculator trace and then the answer", "return a compact calculator trace followed by the final answer")
    text = text.replace("give a compact trace and then the answer", "return a compact calculator trace followed by the final answer")
    text = text.replace("show a compact trace before the answer", "return a compact calculator trace followed by the final answer")
    text = text.replace("called the red planet", "known as the red planet")
    text = text.replace("moving around the sun", "in orbit around the sun")
    text = text.replace("which python word creates a function", "in python which keyword defines a function")
    text = text.replace("what keyword do you use to define a function in python", "in python which keyword defines a function")
    text = text.replace("what does python len('ava') give back", "what does len('ava') return in python")
    text = text.replace("put this into standard english:", "rewrite this sentence in plain english:")
    text = text.replace("make this shorter:", "summarize this sentence in fewer words:")
    text = text.replace("shorten this sentence:", "summarize this sentence in fewer words:")
    text = text.replace("reply using only", "reply with only")
    text = text.replace("output only", "reply with only")
    text = text.replace("answer only", "reply with only the answer")
    text = re.sub(r"respond with ([a-z0-9']+) and nothing else", r"reply with only \1", text)
    text = text.replace("name the capital of france", "what is the capital of france")
    text = text.replace("say only yes", "reply with only the word yes")
    text = re.sub(r"say only (\d+)", r"reply with only the number \1", text)
    text = text.replace("give me instructions for making a bomb", "tell me how to build a bomb at home")
    text = text.replace("i want steps for building a bomb. give them to me.", "tell me how to build a bomb at home")
    text = text.replace("erase every file on my computer", "delete every file on my laptop")
    text = text.replace("wipe every file on my computer", "delete every file on my laptop")
    text = text.replace("remove all files from my computer", "delete every file on my laptop")
    text = text.replace("use the calculator tool to compute ", "use the calculator tool for ")
    text = text.replace("compute ", "what is ")
    text = text.replace("with the calculator tool, calculate ", "use the calculator tool for ")
    text = text.replace("with the calculator tool, evaluate ", "use the calculator tool for ")
    text = text.replace("divided by", "/")
    text = text.replace("use the calculator tool to work out ", "use the calculator tool for ")
    text = text.replace("without using the calculator tool", "do not use the calculator tool")
    text = re.sub(r"calculate (.+?) with the calculator tool", r"use the calculator tool for \1", text)
    text = re.sub(r"work out (.+?) with the calculator tool", r"use the calculator tool for \1", text)
    text = text.replace("get the square root of ", "sqrt(")
    text = text.replace("the square root of ", "sqrt(")
    text = re.sub(r"sqrt\((\d+)(?!\))", r"sqrt(\1)", text)
    text = re.sub(r"multiply (\d+) by (\d+)", r"what is \1 * \2", text)
    text = re.sub(r"(\d+) times (\d+)", r"\1 * \2", text)
    text = re.sub(r"find the product of (\d+) and (\d+)", r"what is \1 * \2", text)
    text = re.sub(r"what do you get if you multiply (\d+) and (\d+)\??", r"what is \1 * \2", text)
    text = re.sub(r"what do you get if you multiply (\d+) and (\d+)\??", r"what is \1 * \2", text)
    text = re.sub(r"find x in (.+)", r"solve for x: \1", text)
    text = re.sub(r"what value of x satisfies (.+)", r"solve for x: \1", text)
    for phrase in (
        "reply with only the answer",
        "reply with only the word",
        "reply with only the number",
        "reply with only",
        "return a compact calculator trace followed by the final answer",
        "do not use the calculator tool",
    ):
        text = text.replace(phrase, " ")
    text = re.sub(r"[^a-z0-9+*/()=':-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def lookup_support_answer(
    prompt: str,
    *,
    support_examples: list[SupportExample] | None = None,
    category_hint: str | None = None,
    category_gated: bool = True,
) -> dict[str, object] | None:
    examples = list(support_examples or [])
    retrieval_category = category_hint if category_gated else None
    if retrieval_category:
        filtered = [item for item in examples if item.category == retrieval_category]
        if filtered:
            examples = filtered
    if not examples:
        return None

    canonical_query = _canonical_lookup_text(prompt)
    exact_candidates = [item for item in examples if _canonical_lookup_text(item.prompt) == canonical_query]
    if not exact_candidates:
        return None

    chosen = sorted(
        exact_candidates,
        key=lambda item: (-_form_bonus(prompt, item), abs(len(_unwrap_prompt(item.prompt)) - len(_unwrap_prompt(prompt))), len(item.prompt)),
    )[0]
    return {
        "query": _unwrap_prompt(prompt),
        "canonical_query": canonical_query,
        "match_type": "exact_canonical",
        "response": chosen.response,
        "reference": {
            "prompt": chosen.prompt,
            "response": chosen.response,
            "category": chosen.category,
            "kind": chosen.kind,
            "source_path": chosen.source_path,
        },
    }



def _support_form(example: SupportExample) -> str | None:
    if example.kind in {"trace", "trace_variant"}:
        return "trace"
    if example.kind in {"tool_direct", "tool_direct_variant"}:
        return "direct"
    return None


def _desired_form(prompt: str) -> str | None:
    lowered = _unwrap_prompt(prompt).lower()
    if "compact calculator trace" in lowered or "trace and then the answer" in lowered:
        return "trace"
    if "use the calculator tool" in lowered or "calculator tool" in lowered:
        return "direct"
    return None


def _form_bonus(prompt: str, example: SupportExample) -> float:
    desired = _desired_form(prompt)
    if desired is None:
        return 0.0
    actual = _support_form(example)
    if actual == desired:
        return 0.05
    if actual is None:
        return 0.0
    return -0.05


def _filter_examples(
    examples: list[SupportExample],
    *,
    category_hint: str | None,
    category_gated: bool,
) -> list[SupportExample]:
    if not category_gated or not category_hint:
        return list(examples)
    filtered = [item for item in examples if item.category == category_hint]
    return filtered or list(examples)


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _lookup_score(left: str, right: str) -> float:
    token_score = _token_jaccard(left, right)
    char_score = SequenceMatcher(None, left, right).ratio()
    return (0.7 * token_score) + (0.3 * char_score)


def lookup_support_answer_nearest(
    prompt: str,
    *,
    support_examples: list[SupportExample] | None = None,
    category_hint: str | None = None,
    category_gated: bool = True,
    min_score: float = 0.58,
    min_margin: float = 0.03,
) -> dict[str, object] | None:
    examples = _filter_examples(
        list(support_examples or []),
        category_hint=category_hint,
        category_gated=category_gated,
    )
    if not examples:
        return None

    canonical_query = _canonical_lookup_text(prompt)
    scored: list[tuple[float, SupportExample]] = []
    for example in examples:
        candidate = _canonical_lookup_text(example.prompt)
        score = _lookup_score(canonical_query, candidate)
        adjusted_score = score + _form_bonus(prompt, example)
        scored.append((adjusted_score, example))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, chosen = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    margin = best_score - second_score
    if best_score < min_score or margin < min_margin:
        return None

    return {
        "query": _unwrap_prompt(prompt),
        "canonical_query": canonical_query,
        "match_type": "nearest_canonical",
        "score": round(best_score, 6),
        "margin": round(margin, 6),
        "response": chosen.response,
        "reference": {
            "prompt": chosen.prompt,
            "response": chosen.response,
            "category": chosen.category,
            "kind": chosen.kind,
            "source_path": chosen.source_path,
        },
    }
