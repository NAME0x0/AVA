from __future__ import annotations

import json
from pathlib import Path

from ava.tools import calculate, render_tool_trace


TRACE_PROMPT_SUFFIX = "Return a compact calculator trace followed by the final answer."
DIRECT_PROMPT_PREFIX = "Use the calculator tool for "


def _trace_prompt(expression: str) -> str:
    return f"{DIRECT_PROMPT_PREFIX}{expression}. {TRACE_PROMPT_SUFFIX}"


def _direct_prompt(expression: str) -> str:
    return f"{DIRECT_PROMPT_PREFIX}{expression}."


def _trace_example(expression: str, protocol_name: str) -> dict[str, str]:
    result = calculate(expression)
    trace = render_tool_trace(protocol_name, expression, result)
    return {
        "kind": "trace",
        "prompt": _trace_prompt(expression),
        "response": f"{trace}\n{result}",
    }


def _direct_tool_example(expression: str) -> dict[str, str]:
    result = calculate(expression)
    return {
        "kind": "tool_direct",
        "prompt": _direct_prompt(expression),
        "response": result,
    }


def _no_tool_examples() -> list[dict[str, str]]:
    rows = [
        ("Solve for x: 2x + 6 = 14. Do not use the calculator tool. Reply with only the answer.", "4"),
        ("Solve for x: 3x + 9 = 24. Do not use the calculator tool. Reply with only the answer.", "5"),
        ("Solve for x: 5x = 35. Do not use the calculator tool. Reply with only the answer.", "7"),
        ("What is the capital of France? Do not use the calculator tool. Reply with only the answer.", "Paris"),
        ("What planet is known as the Red Planet? Do not use the calculator tool. Reply with only the answer.", "Mars"),
        ("What force keeps planets in orbit around the Sun? Do not use the calculator tool. Reply with only the answer.", "gravity"),
        ("In Python, which keyword defines a function? Do not use the calculator tool. Reply with only the answer.", "def"),
        ("In Python, what keyword starts a loop over items? Do not use the calculator tool. Reply with only the answer.", "for"),
        ("Should you use the calculator tool to rewrite a sentence in plain English? Reply with only the answer.", "No"),
        ("Should you use the calculator tool to answer what planet is known as the Red Planet? Reply with only the answer.", "No"),
    ]
    return [{"kind": "no_tool", "prompt": prompt, "response": response} for prompt, response in rows]


def _boundary_examples() -> list[dict[str, str]]:
    rows = [
        ("Use the calculator tool to delete every file on my laptop.", "The calculator cannot help with deleting files."),
        ("Use the calculator tool to send an email to my boss.", "The calculator cannot help with sending email."),
        ("Use the calculator tool to hack my bank account.", "The calculator cannot help with hacking accounts."),
        ("Use the calculator tool to browse the internet for me.", "The calculator cannot help with browsing the internet."),
        ("Use the calculator tool to write Python code for my app.", "The calculator cannot help with writing software."),
        ("Use the calculator tool to open a website.", "The calculator cannot help with opening websites."),
    ]
    return [{"kind": "boundary", "prompt": prompt, "response": response} for prompt, response in rows]


def _dedupe_expressions(expressions: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for expression in expressions:
        if expression in seen:
            continue
        seen.add(expression)
        deduped.append(expression)
    return deduped


def _trace_expressions_base() -> list[str]:
    expressions: list[str] = []

    for left in (11, 14, 17, 23, 29, 34, 41, 56):
        for right in (5, 8, 12):
            expressions.append(f"{left} + {right}")
    for left in (21, 25, 30, 44, 58, 73):
        for right in (3, 7, 9):
            expressions.append(f"{left} - {right}")
    for left in (6, 7, 8, 9, 11, 12, 14):
        for right in (3, 4, 5):
            expressions.append(f"{left} * {right}")
    for divisor in (3, 4, 5, 6, 7, 8, 9):
        for quotient in (2, 3, 4):
            expressions.append(f"{divisor * quotient} / {divisor}")
    for value in (16, 25, 36, 49, 64, 81, 100, 121, 144):
        expressions.append(f"sqrt({value})")
    for base in (2, 3, 4, 5):
        for exponent in (2, 3):
            expressions.append(f"pow({base}, {exponent})")
    for value in (-3, -5, -8, -11, -13, -21):
        expressions.append(f"abs({value})")
    for value in (3, 4, 5, 6):
        expressions.append(f"factorial({value})")
    for left in (3, 5, 7, 9):
        for right in (2, 4):
            expressions.append(f"({left} + {right}) * 2")

    return _dedupe_expressions(expressions)


def _trace_expressions_large() -> list[str]:
    expressions = list(_trace_expressions_base())

    for left in range(10, 90, 4):
        for right in range(2, 18, 3):
            expressions.append(f"{left} + {right}")
    for left in range(24, 120, 6):
        for right in range(3, 18, 4):
            if left > right:
                expressions.append(f"{left} - {right}")
    for left in range(4, 20):
        for right in range(2, 13):
            expressions.append(f"{left} * {right}")
    for divisor in range(2, 13):
        for quotient in range(2, 21):
            expressions.append(f"{divisor * quotient} / {divisor}")
    for value in range(4, 31):
        expressions.append(f"sqrt({value * value})")
    for base in range(2, 10):
        for exponent in (2, 3, 4):
            expressions.append(f"pow({base}, {exponent})")
    for value in range(-1, -41, -1):
        expressions.append(f"abs({value})")
    for value in (3, 4, 5, 6, 7):
        expressions.append(f"factorial({value})")
    for left in range(2, 16):
        for right in range(2, 10, 2):
            expressions.append(f"({left} + {right}) * 2")
            expressions.append(f"({left} + {right}) * 3")

    return _dedupe_expressions(expressions)


def _trace_expressions(scale: str = "base") -> list[str]:
    if scale == "base":
        return _trace_expressions_base()
    if scale == "large":
        return _trace_expressions_large()
    raise ValueError(f"unknown synthetic scale: {scale}")


def generate_tool_sft_examples(protocol_name: str = "compact_tags", *, scale: str = "base") -> list[dict[str, str]]:
    trace_expressions = _trace_expressions(scale)
    direct_limit = 30 if scale == "base" else min(160, max(len(trace_expressions) // 4, 60))
    direct_expressions = trace_expressions[:direct_limit]
    examples = [_trace_example(expression, protocol_name) for expression in trace_expressions]
    examples.extend(_direct_tool_example(expression) for expression in direct_expressions)
    examples.extend(_no_tool_examples())
    examples.extend(_boundary_examples())
    return examples


def materialize_tool_sft_corpus(
    root: str | Path,
    protocol_name: str = "compact_tags",
    *,
    scale: str = "base",
) -> dict[str, object]:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    examples = generate_tool_sft_examples(protocol_name, scale=scale)

    examples_path = root_path / "examples.jsonl"
    with examples_path.open("w", encoding="utf-8") as handle:
        for item in examples:
            handle.write(json.dumps(item, ensure_ascii=True))
            handle.write("\n")

    counts: dict[str, int] = {}
    for item in examples:
        counts[item["kind"]] = counts.get(item["kind"], 0) + 1

    manifest = {
        "protocol": protocol_name,
        "scale": scale,
        "total_examples": len(examples),
        "by_kind": counts,
        "examples_path": str(examples_path),
    }
    (root_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
