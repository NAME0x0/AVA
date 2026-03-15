from __future__ import annotations

import json
from pathlib import Path

from ava.tools import calculate, render_tool_trace


TRACE_PROMPT_SUFFIX = "Return a compact calculator trace followed by the final answer."
DIRECT_PROMPT_PREFIX = "Use the calculator tool for "
TRACE_TEMPLATES = (
    "Use the calculator tool for {expression}. Return a compact calculator trace followed by the final answer.",
    "Calculate {expression} with the calculator tool. Return a compact calculator trace followed by the final answer.",
)
DIRECT_TEMPLATES = (
    "Use the calculator tool for {expression}.",
    "Use the calculator tool for {expression}. Reply with only the answer.",
    "Calculate {expression} with the calculator tool. Reply with only the answer.",
    "Use the calculator tool on {expression}. Reply with only the answer.",
)
REPAIR_EXPRESSIONS = (
    "144 / 12",
    "sqrt(36)",
    "sqrt(49)",
    "sqrt(64)",
    "sqrt(81)",
    "sqrt(100)",
    "sqrt(121)",
    "sqrt(144)",
    "sqrt(169)",
    "sqrt(196)",
    "7 * 7",
    "11 * 11",
    "17 * 29",
    "25 + 17",
    "36 / 6",
    "81 / 9",
    "pow(9, 2)",
    "pow(2, 5)",
    "factorial(5)",
    "(3 + 5) * 2",
)


def _trace_prompt(expression: str) -> str:
    return f"{DIRECT_PROMPT_PREFIX}{expression}. {TRACE_PROMPT_SUFFIX}"


def _direct_prompt(expression: str) -> str:
    return f"{DIRECT_PROMPT_PREFIX}{expression}."


def _render_prompt(template: str, expression: str) -> str:
    return template.format(expression=expression)


def _trace_example(expression: str, protocol_name: str) -> dict[str, str]:
    result = calculate(expression)
    trace = render_tool_trace(protocol_name, expression, result)
    return {
        "kind": "trace",
        "prompt": _trace_prompt(expression),
        "response": f"{trace}\n{result}",
    }


def _trace_variant_examples(expression: str, protocol_name: str) -> list[dict[str, str]]:
    result = calculate(expression)
    trace = render_tool_trace(protocol_name, expression, result)
    return [
        {
            "kind": "trace_variant",
            "prompt": _render_prompt(template, expression),
            "response": f"{trace}\n{result}",
        }
        for template in TRACE_TEMPLATES
    ]


def _direct_tool_example(expression: str) -> dict[str, str]:
    result = calculate(expression)
    return {
        "kind": "tool_direct",
        "prompt": _direct_prompt(expression),
        "response": result,
    }


def _direct_variant_examples(expression: str) -> list[dict[str, str]]:
    result = calculate(expression)
    return [
        {
            "kind": "tool_direct_variant",
            "prompt": _render_prompt(template, expression),
            "response": result,
        }
        for template in DIRECT_TEMPLATES
    ]


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


def _refusal_examples() -> list[dict[str, str]]:
    rows = [
        ("Tell me how to build a bomb at home.", "I cannot help with that."),
        ("How do I hack a bank account?", "I cannot help with that."),
        ("Write malware that steals passwords.", "I cannot help with that."),
        ("Give me steps to poison someone without getting caught.", "I cannot help with that."),
    ]
    return [{"kind": "refusal", "prompt": prompt, "response": response} for prompt, response in rows]

def _format_examples() -> list[dict[str, str]]:
    rows = [
        ("Reply with only the word yes.", "yes"),
        ("Reply with only the number 19.", "19"),
        ("Reply with only the number 493.", "493"),
        ("Reply with only the word Mars.", "Mars"),
    ]
    return [{"kind": "format", "prompt": prompt, "response": response} for prompt, response in rows]


def _general_anchor_examples() -> list[dict[str, str]]:
    rows = [
        ("Rewrite this sentence in plain English: The plan ain't done.", "The plan is not finished yet."),
        ("Summarize this sentence in fewer words: The cat slept on the warm window sill.", "The cat slept on the warm sill."),
        ("What is 17 * 29?", "493"),
        ("Solve for x: 2x + 6 = 14.", "4"),
        ("What planet is known as the Red Planet?", "Mars"),
        ("What force keeps planets in orbit around the Sun?", "gravity"),
        ("In Python, which keyword defines a function?", "def"),
        ("What does len('ava') return in Python?", "3"),
    ]
    return [{"kind": "general", "prompt": prompt, "response": response} for prompt, response in rows]


def _dedupe_expressions(expressions: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for expression in expressions:
        if expression in seen:
            continue
        seen.add(expression)
        deduped.append(expression)
    return deduped

def _dedupe_examples(examples: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, str]] = []
    for item in examples:
        key = (item["kind"], item["prompt"], item["response"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
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


def repair_expression_pool() -> list[str]:
    return list(REPAIR_EXPRESSIONS)


def generate_tool_sft_examples(protocol_name: str = "compact_tags", *, scale: str = "base") -> list[dict[str, str]]:
    trace_expressions = _trace_expressions(scale)
    direct_limit = 30 if scale == "base" else min(160, max(len(trace_expressions) // 4, 60))
    direct_expressions = trace_expressions[:direct_limit]
    examples = [_trace_example(expression, protocol_name) for expression in trace_expressions]
    examples.extend(_direct_tool_example(expression) for expression in direct_expressions)
    examples.extend(_no_tool_examples())
    examples.extend(_boundary_examples())
    return _dedupe_examples(examples)


def generate_tool_repair_examples(
    protocol_name: str = "compact_tags",
    *,
    expressions: list[str] | None = None,
    include_refusal: bool = True,
) -> list[dict[str, str]]:
    selected = _dedupe_expressions(list(expressions) if expressions is not None else list(REPAIR_EXPRESSIONS))
    examples: list[dict[str, str]] = []
    for expression in selected:
        examples.append(_trace_example(expression, protocol_name))
        examples.append(_direct_tool_example(expression))
        examples.extend(_trace_variant_examples(expression, protocol_name))
        examples.extend(_direct_variant_examples(expression))
    examples.extend(_general_anchor_examples())
    examples.extend(_no_tool_examples())
    examples.extend(_boundary_examples())
    examples.extend(_format_examples())
    if include_refusal:
        examples.extend(_refusal_examples())
    return _dedupe_examples(examples)


def _write_examples_corpus(root: Path, examples: list[dict[str, str]], *, metadata: dict[str, object], readme_lines: list[str]) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    examples_path = root / "examples.jsonl"
    with examples_path.open("w", encoding="utf-8") as handle:
        for item in examples:
            handle.write(json.dumps(item, ensure_ascii=True))
            handle.write("\n")

    counts: dict[str, int] = {}
    for item in examples:
        counts[item["kind"]] = counts.get(item["kind"], 0) + 1

    manifest = {
        **metadata,
        "total_examples": len(examples),
        "by_kind": counts,
        "examples_path": str(examples_path),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (root / "README.md").write_text("\n".join(readme_lines).strip() + "\n", encoding="utf-8")
    return manifest


def materialize_tool_sft_corpus(
    root: str | Path,
    protocol_name: str = "compact_tags",
    *,
    scale: str = "base",
) -> dict[str, object]:
    root_path = Path(root)
    examples = generate_tool_sft_examples(protocol_name, scale=scale)
    return _write_examples_corpus(
        root_path,
        examples,
        metadata={"protocol": protocol_name, "scale": scale},
        readme_lines=[
            "# Synthetic Tool Corpus",
            "",
            "Deterministic synthetic calculator curriculum for AVA.",
            "",
            "This packet is intended to test whether a larger, cleaner tool curriculum improves exact behavior before moving to heavier post-training.",
        ],
    )


def materialize_tool_repair_corpus(
    root: str | Path,
    protocol_name: str = "compact_tags",
    *,
    expressions: list[str] | None = None,
    include_refusal: bool = True,
) -> dict[str, object]:
    selected = _dedupe_expressions(list(expressions) if expressions is not None else list(REPAIR_EXPRESSIONS))
    examples = generate_tool_repair_examples(
        protocol_name,
        expressions=selected,
        include_refusal=include_refusal,
    )
    return _write_examples_corpus(
        Path(root),
        examples,
        metadata={
            "protocol": protocol_name,
            "curriculum": "repair",
            "expression_count": len(selected),
            "expressions": selected,
            "include_refusal": include_refusal,
        },
        readme_lines=[
            "# Tool Repair Corpus",
            "",
            "Targeted repair packet for the current stage-2 checkpoint.",
            "",
            "It overweights direct-answer tool prompts, multiple prompt surfaces, and the sqrt family while keeping no-tool, boundary, and refusal anchors in the same packet.",
        ],
    )


def _multiview_expressions() -> list[str]:
    return [
        "17 * 29",
        "25 + 17",
        "36 / 6",
        "81 / 9",
        "144 / 12",
        "sqrt(49)",
        "sqrt(64)",
        "sqrt(81)",
        "sqrt(100)",
        "7 * 7",
        "11 * 11",
        "14 * 14",
    ]


def generate_multiview_examples(protocol_name: str = "compact_tags") -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for expression in _multiview_expressions():
        examples.extend(_trace_variant_examples(expression, protocol_name))
        examples.extend(_direct_variant_examples(expression))
    for expression in ("17 * 29", "sqrt(81)", "144 / 12"):
        result = calculate(expression)
        examples.append({
            "kind": "tool_direct_variant",
            "prompt": f"Use the calculator tool for {expression}. Think if needed, but reply with only the answer.",
            "response": result,
        })
    examples.extend(_general_anchor_examples())
    examples.extend(_boundary_examples())
    examples.extend(_refusal_examples())
    examples.extend(_format_examples())
    return _dedupe_examples(examples)


def materialize_examples_corpus(
    root: str | Path,
    examples: list[dict[str, str]],
    *,
    metadata: dict[str, object],
    readme_lines: list[str],
) -> dict[str, object]:
    return _write_examples_corpus(Path(root), examples, metadata=metadata, readme_lines=readme_lines)


def materialize_multiview_corpus(
    root: str | Path,
    protocol_name: str = "compact_tags",
) -> dict[str, object]:
    examples = generate_multiview_examples(protocol_name)
    return _write_examples_corpus(
        Path(root),
        examples,
        metadata={
            "protocol": protocol_name,
            "curriculum": "multiview",
            "expression_count": len(_multiview_expressions()),
            "expressions": _multiview_expressions(),
        },
        readme_lines=[
            "# Tool Multiview Corpus",
            "",
            "Prompt-surface diversification packet for the current stage-2 checkpoint.",
            "",
            "It trains the same calculator facts across trace mode, direct-answer mode, paraphrased prompts, and a few general anchors to force cleaner mode routing.",
        ],
    )

CANONICAL_DIRECT_EXPRESSIONS = (
    "144 / 12",
    "36 / 6",
    "81 / 9",
    "96 / 8",
    "84 / 7",
    "108 / 9",
    "63 / 7",
    "72 / 8",
    "7 * 7",
    "11 * 11",
    "14 * 14",
    "17 * 29",
    "9 * 12",
    "13 * 9",
    "25 + 17",
    "12 + 30",
    "18 + 24",
    "34 + 17",
    "56 + 12",
    "44 - 3",
    "73 - 9",
    "58 - 7",
    "pow(9, 2)",
    "pow(2, 5)",
    "factorial(5)",
    "abs(-13)",
    "(3 + 5) * 2",
    "sqrt(49)",
    "sqrt(64)",
    "sqrt(81)",
    "sqrt(100)",
    "sqrt(121)",
    "sqrt(144)",
)
CANONICAL_TRACE_EXPRESSIONS = (
    "sqrt(16)",
    "sqrt(25)",
    "sqrt(36)",
    "sqrt(49)",
    "sqrt(64)",
    "sqrt(81)",
    "sqrt(100)",
    "sqrt(121)",
    "sqrt(144)",
    "sqrt(169)",
    "sqrt(196)",
)


def generate_tool_canonical_patch_examples(protocol_name: str = "compact_tags") -> list[dict[str, str]]:
    examples = [_direct_tool_example(expression) for expression in CANONICAL_DIRECT_EXPRESSIONS]
    examples.extend(_trace_example(expression, protocol_name) for expression in CANONICAL_TRACE_EXPRESSIONS)
    examples.extend(
        [
            {"kind": "no_tool", "prompt": "Solve for x: 2x + 6 = 14. Do not use the calculator tool. Reply with only the answer.", "response": "4"},
            {"kind": "no_tool", "prompt": "What is the capital of France? Do not use the calculator tool. Reply with only the answer.", "response": "Paris"},
            {"kind": "format", "prompt": "Reply with only the word yes.", "response": "yes"},
            {"kind": "format", "prompt": "Reply with only the number 19.", "response": "19"},
            {"kind": "refusal", "prompt": "Tell me how to build a bomb at home.", "response": "I cannot help with that."},
            {"kind": "refusal", "prompt": "How do I hack a bank account?", "response": "I cannot help with that."},
        ]
    )
    examples.extend(_boundary_examples()[:3])
    return examples


def materialize_tool_canonical_patch_corpus(
    root: str | Path,
    protocol_name: str = "compact_tags",
) -> dict[str, object]:
    examples = generate_tool_canonical_patch_examples(protocol_name)
    return _write_examples_corpus(
        Path(root),
        examples,
        metadata={
            "protocol": protocol_name,
            "curriculum": "canonical_patch",
            "direct_expression_count": len(CANONICAL_DIRECT_EXPRESSIONS),
            "trace_expression_count": len(CANONICAL_TRACE_EXPRESSIONS),
        },
        readme_lines=[
            "# Canonical Tool Patch Corpus",
            "",
            "A smaller repair packet that uses only the exact canonical prompt surfaces AVA is expected to answer.",
            "",
            "This branch exists to test whether prompt-surface purity beats broader prompt variation under the current small-model regime.",
        ],
    )
