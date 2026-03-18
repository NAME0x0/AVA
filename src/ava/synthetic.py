from __future__ import annotations

import json
from pathlib import Path

from ava.external_benchmarks import load_arc_challenge_tasks, load_gsm8k_tasks
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
        (
            "Solve for x: 2x + 6 = 14. Do not use the calculator tool. Reply with only the answer.",
            "4",
        ),
        (
            "Solve for x: 3x + 9 = 24. Do not use the calculator tool. Reply with only the answer.",
            "5",
        ),
        ("Solve for x: 5x = 35. Do not use the calculator tool. Reply with only the answer.", "7"),
        (
            "What is the capital of France? Do not use the calculator tool. Reply with only the answer.",
            "Paris",
        ),
        (
            "What planet is known as the Red Planet? Do not use the calculator tool. Reply with only the answer.",
            "Mars",
        ),
        (
            "What force keeps planets in orbit around the Sun? Do not use the calculator tool. Reply with only the answer.",
            "gravity",
        ),
        (
            "In Python, which keyword defines a function? Do not use the calculator tool. Reply with only the answer.",
            "def",
        ),
        (
            "In Python, what keyword starts a loop over items? Do not use the calculator tool. Reply with only the answer.",
            "for",
        ),
        (
            "Should you use the calculator tool to rewrite a sentence in plain English? Reply with only the answer.",
            "No",
        ),
        (
            "Should you use the calculator tool to answer what planet is known as the Red Planet? Reply with only the answer.",
            "No",
        ),
    ]
    return [
        {"kind": "no_tool", "prompt": prompt, "response": response} for prompt, response in rows
    ]


def _boundary_examples() -> list[dict[str, str]]:
    rows = [
        (
            "Use the calculator tool to delete every file on my laptop.",
            "The calculator cannot help with deleting files.",
        ),
        (
            "Use the calculator tool to send an email to my boss.",
            "The calculator cannot help with sending email.",
        ),
        (
            "Use the calculator tool to hack my bank account.",
            "The calculator cannot help with hacking accounts.",
        ),
        (
            "Use the calculator tool to browse the internet for me.",
            "The calculator cannot help with browsing the internet.",
        ),
        (
            "Use the calculator tool to write Python code for my app.",
            "The calculator cannot help with writing software.",
        ),
        (
            "Use the calculator tool to open a website.",
            "The calculator cannot help with opening websites.",
        ),
    ]
    return [
        {"kind": "boundary", "prompt": prompt, "response": response} for prompt, response in rows
    ]


def _refusal_examples() -> list[dict[str, str]]:
    rows = [
        ("Tell me how to build a bomb at home.", "I cannot help with that."),
        ("How do I hack a bank account?", "I cannot help with that."),
        ("Write malware that steals passwords.", "I cannot help with that."),
        ("Give me steps to poison someone without getting caught.", "I cannot help with that."),
    ]
    return [
        {"kind": "refusal", "prompt": prompt, "response": response} for prompt, response in rows
    ]


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
        (
            "Rewrite this sentence in plain English: The plan ain't done.",
            "The plan is not finished yet.",
        ),
        (
            "Summarize this sentence in fewer words: The cat slept on the warm window sill.",
            "The cat slept on the warm sill.",
        ),
        ("What is 17 * 29?", "493"),
        ("Solve for x: 2x + 6 = 14.", "4"),
        ("What planet is known as the Red Planet?", "Mars"),
        ("What force keeps planets in orbit around the Sun?", "gravity"),
        ("In Python, which keyword defines a function?", "def"),
        ("What does len('ava') return in Python?", "3"),
    ]
    return [
        {"kind": "general", "prompt": prompt, "response": response} for prompt, response in rows
    ]


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


def generate_tool_sft_examples(
    protocol_name: str = "compact_tags", *, scale: str = "base"
) -> list[dict[str, str]]:
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
    selected = _dedupe_expressions(
        list(expressions) if expressions is not None else list(REPAIR_EXPRESSIONS)
    )
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


def _write_examples_corpus(
    root: Path,
    examples: list[dict[str, str]],
    *,
    metadata: dict[str, object],
    readme_lines: list[str],
) -> dict[str, object]:
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
    selected = _dedupe_expressions(
        list(expressions) if expressions is not None else list(REPAIR_EXPRESSIONS)
    )
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
        examples.append(
            {
                "kind": "tool_direct_variant",
                "prompt": f"Use the calculator tool for {expression}. Think if needed, but reply with only the answer.",
                "response": result,
            }
        )
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
    return _write_examples_corpus(
        Path(root), examples, metadata=metadata, readme_lines=readme_lines
    )


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


def generate_tool_canonical_patch_examples(
    protocol_name: str = "compact_tags",
) -> list[dict[str, str]]:
    examples = [_direct_tool_example(expression) for expression in CANONICAL_DIRECT_EXPRESSIONS]
    examples.extend(
        _trace_example(expression, protocol_name) for expression in CANONICAL_TRACE_EXPRESSIONS
    )
    examples.extend(
        [
            {
                "kind": "no_tool",
                "prompt": "Solve for x: 2x + 6 = 14. Do not use the calculator tool. Reply with only the answer.",
                "response": "4",
            },
            {
                "kind": "no_tool",
                "prompt": "What is the capital of France? Do not use the calculator tool. Reply with only the answer.",
                "response": "Paris",
            },
            {"kind": "format", "prompt": "Reply with only the word yes.", "response": "yes"},
            {"kind": "format", "prompt": "Reply with only the number 19.", "response": "19"},
            {
                "kind": "refusal",
                "prompt": "Tell me how to build a bomb at home.",
                "response": "I cannot help with that.",
            },
            {
                "kind": "refusal",
                "prompt": "How do I hack a bank account?",
                "response": "I cannot help with that.",
            },
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


def _multiple_choice_prompt(question: str, choices: tuple[tuple[str, str], ...]) -> str:
    lines = [question.strip(), "", "Options:"]
    for label, choice in choices:
        lines.append(f"{label}. {choice}")
    lines.extend(["", "Reply with only the correct option label."])
    return "\n".join(lines)


def _rotated_choice_set(
    correct: str, distractors: tuple[str, str, str], index: int
) -> tuple[tuple[tuple[str, str], ...], str]:
    labels = ("A", "B", "C", "D")
    answer_index = index % len(labels)
    offset = index % len(distractors)
    rotated_distractors = list(distractors[offset:]) + list(distractors[:offset])
    options = list(rotated_distractors)
    options.insert(answer_index, correct)
    return tuple(zip(labels, options, strict=True)), labels[answer_index]


def _science_multiple_choice_examples() -> list[dict[str, str]]:
    rows = [
        ("Which planet is known as the Red Planet?", "Mars", ("Venus", "Jupiter", "Saturn")),
        (
            "What force keeps planets in orbit around the Sun?",
            "gravity",
            ("friction", "magnetism", "sound"),
        ),
        (
            "What gas do plants take in from the air for photosynthesis?",
            "carbon dioxide",
            ("oxygen", "nitrogen", "helium"),
        ),
        (
            "What part of a plant absorbs water from the soil?",
            "roots",
            ("flowers", "leaves", "seeds"),
        ),
        (
            "What change of state happens when liquid water becomes water vapor?",
            "evaporation",
            ("freezing", "melting", "condensation"),
        ),
        ("Which organ pumps blood through the human body?", "heart", ("lung", "liver", "brain")),
        ("Which star is closest to Earth?", "the Sun", ("Polaris", "Sirius", "Betelgeuse")),
        (
            "What instrument is used to measure temperature?",
            "thermometer",
            ("barometer", "ruler", "microscope"),
        ),
        ("Which material is attracted to a magnet?", "iron", ("plastic", "glass", "rubber")),
        (
            "What should a student keep the same when testing which paper towel absorbs the most water?",
            "the amount of water used each time",
            ("the brand of cereal nearby", "the color of the table", "the day of the week"),
        ),
        (
            "Which process turns food into energy inside living things?",
            "cellular respiration",
            ("erosion", "reflection", "evaporation"),
        ),
        ("What do bees help many plants do?", "pollinate", ("freeze", "evaporate", "erode")),
        (
            "Which layer of Earth do people live on?",
            "the crust",
            ("the core", "the mantle", "the inner core"),
        ),
        (
            "What is needed for a circuit to light a bulb?",
            "a complete loop",
            ("a broken wire only", "a paper clip only", "an empty battery"),
        ),
        (
            "Which phase of the Moon appears fully lit from Earth?",
            "full moon",
            ("new moon", "crescent moon", "quarter moon"),
        ),
        (
            "What property describes how much space an object takes up?",
            "volume",
            ("speed", "mass only", "temperature"),
        ),
    ]
    examples: list[dict[str, str]] = []
    for index, (question, correct, distractors) in enumerate(rows):
        choices, answer_key = _rotated_choice_set(correct, distractors, index)
        examples.append(
            {
                "kind": "science_mc",
                "prompt": _multiple_choice_prompt(question, choices),
                "response": answer_key,
            }
        )
    return examples


def _commonsense_multiple_choice_examples() -> list[dict[str, str]]:
    rows = [
        ("Which item would you use to cut paper?", "scissors", ("pillow", "blanket", "toothbrush")),
        (
            "Where would you usually store frozen food?",
            "freezer",
            ("oven", "desk drawer", "mailbox"),
        ),
        (
            "What should you wear in heavy rain?",
            "a raincoat",
            ("sandals only", "swim goggles", "headphones"),
        ),
        ("Which room is best for taking a shower?", "bathroom", ("garage", "attic", "porch")),
        ("What do you use to unlock a door?", "a key", ("a spoon", "a pillow", "a plate")),
        (
            "Where do books usually belong in a library?",
            "on shelves",
            ("in a sink", "under a car", "inside a shoe"),
        ),
        ("Which tool is best for writing on paper?", "a pencil", ("a fork", "a mug", "a towel")),
        (
            "What should you do before crossing a busy street?",
            "look both ways",
            ("close your eyes", "run backward", "drop your shoes"),
        ),
    ]
    examples: list[dict[str, str]] = []
    for index, (question, correct, distractors) in enumerate(rows):
        choices, answer_key = _rotated_choice_set(correct, distractors, index + 1)
        examples.append(
            {
                "kind": "commonsense_mc",
                "prompt": _multiple_choice_prompt(question, choices),
                "response": answer_key,
            }
        )
    return examples


def _word_problem_examples() -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for start in range(12, 52, 4):
        spent = (start // 4) % 5 + 2
        bought = (start // 6) % 4 + 1
        answer = start - spent + bought
        prompt = (
            f"Ava had {start} stickers. She gave {spent} to her friend and then bought {bought} more. "
            "How many stickers does she have now?\n\nReply with only the final answer."
        )
        examples.append({"kind": "math_word", "prompt": prompt, "response": str(answer)})
    for boxes in range(3, 15):
        per_box = boxes % 5 + 4
        answer = boxes * per_box
        prompt = (
            f"A store has {boxes} boxes with {per_box} pencils in each box. "
            "How many pencils are there in all?\n\nReply with only the final answer."
        )
        examples.append({"kind": "math_word", "prompt": prompt, "response": str(answer)})
    for total in range(24, 88, 8):
        group = total // 8 + 2
        remainder = total % group
        if remainder != 0:
            continue
        answer = total // group
        prompt = (
            f"A teacher divides {total} notebooks equally among {group} students. "
            "How many notebooks does each student get?\n\nReply with only the final answer."
        )
        examples.append({"kind": "math_word", "prompt": prompt, "response": str(answer)})
    for price in range(3, 11):
        count = price + 2
        extra = count // 2
        answer = (price * count) + extra
        prompt = (
            f"Each notebook costs {price} dollars. Mia buys {count} notebooks and then pays {extra} extra dollars for a pen. "
            "How many dollars does she spend in total?\n\nReply with only the final answer."
        )
        examples.append({"kind": "math_word", "prompt": prompt, "response": str(answer)})
    return examples


def generate_public_reasoning_patch_examples(
    protocol_name: str = "compact_tags",
) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    examples.extend(_science_multiple_choice_examples())
    examples.extend(_commonsense_multiple_choice_examples())
    examples.extend(_word_problem_examples())
    for expression in ("144 / 12", "sqrt(81)", "17 * 29", "25 + 17"):
        examples.append(_trace_example(expression, protocol_name))
        examples.append(_direct_tool_example(expression))
        examples.extend(_direct_variant_examples(expression)[:2])
    examples.extend(_general_anchor_examples())
    examples.extend(_no_tool_examples())
    examples.extend(_boundary_examples())
    examples.extend(_refusal_examples())
    examples.extend(_format_examples())
    return _dedupe_examples(examples)


def materialize_public_reasoning_patch_corpus(
    root: str | Path,
    protocol_name: str = "compact_tags",
) -> dict[str, object]:
    examples = generate_public_reasoning_patch_examples(protocol_name)
    return _write_examples_corpus(
        Path(root),
        examples,
        metadata={
            "protocol": protocol_name,
            "curriculum": "public_reasoning_patch",
            "science_examples": len([item for item in examples if item["kind"] == "science_mc"]),
            "commonsense_examples": len(
                [item for item in examples if item["kind"] == "commonsense_mc"]
            ),
            "math_examples": len([item for item in examples if item["kind"] == "math_word"]),
        },
        readme_lines=[
            "# Public Reasoning Patch Corpus",
            "",
            "A deterministic patch packet aimed at the first public-benchmark failures.",
            "",
            "It balances A/B/C/D multiple-choice supervision, adds answer-only arithmetic word problems, and preserves a compact tool/compliance anchor set so AVA does not forget the earlier tool wins.",
        ],
    )


def _teacher_packet_contract(kind: str) -> tuple[str, str]:
    mapping = {
        "science_mc": ("science", "label_only"),
        "commonsense_mc": ("language", "label_only"),
        "math_word": ("math", "final_answer_only"),
        "trace": ("tool", "tool_trace_then_answer"),
        "tool_direct": ("tool", "direct_tool_answer"),
        "tool_direct_variant": ("tool", "direct_tool_answer"),
        "no_tool": ("tool", "final_answer_only"),
        "boundary": ("compliance", "refusal_short"),
        "refusal": ("compliance", "refusal_short"),
        "format": ("compliance", "format_exact"),
        "general": ("language", "final_answer_only"),
    }
    return mapping.get(kind, ("language", "final_answer_only"))


def generate_teacher_distill_examples(
    teacher_model: str = "codex",
    protocol_name: str = "compact_tags",
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for item in generate_public_reasoning_patch_examples(protocol_name):
        category, format_contract = _teacher_packet_contract(str(item["kind"]))
        tags = [str(item["kind"]), category, format_contract]
        examples.append(
            {
                **item,
                "teacher_model": teacher_model,
                "source_type": "synthetic_teacher",
                "category": category,
                "difficulty": "easy",
                "format_contract": format_contract,
                "verifier_status": "pass",
                "tags": tags,
            }
        )
    return examples


def materialize_teacher_distill_corpus(
    root: str | Path,
    teacher_model: str = "codex",
    protocol_name: str = "compact_tags",
) -> dict[str, object]:
    examples = generate_teacher_distill_examples(
        teacher_model=teacher_model, protocol_name=protocol_name
    )
    return _write_examples_corpus(
        Path(root),
        examples,
        metadata={
            "protocol": protocol_name,
            "curriculum": "teacher_distill",
            "teacher_model": teacher_model,
            "total_examples": len(examples),
        },
        readme_lines=[
            "# Teacher Distill Corpus",
            "",
            "First SOP-shaped teacher packet for AVA.",
            "",
            "This packet uses the public-reasoning repair mix, but adds teacher metadata and explicit format contracts so multiple teacher families can be compared cleanly.",
        ],
    )


def _mc_question_from_prompt(prompt: str) -> str:
    return prompt.split("\n\nOptions:\n", 1)[0].strip()


def _rotate_multiple_choice_task(
    prompt: str,
    choices: tuple[tuple[str, str], ...],
    expected_label: str,
    rotation: int,
) -> tuple[str, str]:
    labels = tuple(label for label, _ in choices)
    texts = [text for _label, text in choices]
    if expected_label not in labels:
        raise ValueError(f"expected label {expected_label} not present in choices")
    rotation = rotation % len(texts)
    rotated_texts = texts[rotation:] + texts[:rotation]
    answer_index = labels.index(expected_label)
    rotated_answer_index = (answer_index - rotation) % len(texts)
    rotated_choices = tuple(zip(labels, rotated_texts, strict=True))
    return _multiple_choice_prompt(_mc_question_from_prompt(prompt), rotated_choices), labels[
        rotated_answer_index
    ]


def _arc_train_examples(*, limit: int | None = None, rotations: int = 3) -> list[dict[str, object]]:
    tasks = load_arc_challenge_tasks(split="train", limit=limit)
    examples: list[dict[str, object]] = []
    for _index, task in enumerate(tasks):
        examples.append(
            {
                "kind": "arc_mc",
                "prompt": task.prompt,
                "response": task.expected,
                "benchmark": task.benchmark,
                "source_type": "hf_train_split",
                "category": task.category,
                "difficulty": "train",
                "format_contract": "label_only",
                "verifier_status": "ground_truth",
                "tags": ["arc", "science", "label_only"],
            }
        )
        for offset in range(1, max(rotations, 0) + 1):
            rotation = offset % len(task.choices) or 1
            rotated_prompt, rotated_answer = _rotate_multiple_choice_task(
                task.prompt,
                task.choices,
                task.expected,
                rotation,
            )
            examples.append(
                {
                    "kind": "arc_mc_aug",
                    "prompt": rotated_prompt,
                    "response": rotated_answer,
                    "benchmark": task.benchmark,
                    "source_type": "hf_train_split_augmented",
                    "category": task.category,
                    "difficulty": "train",
                    "format_contract": "label_only",
                    "verifier_status": "ground_truth",
                    "tags": ["arc", "science", "label_only", "rotated_choices"],
                }
            )
    return examples


def _gsm8k_train_examples(*, limit: int | None = None) -> list[dict[str, object]]:
    tasks = load_gsm8k_tasks(split="train", limit=limit)
    return [
        {
            "kind": "gsm8k_train",
            "prompt": task.prompt,
            "response": task.expected,
            "benchmark": task.benchmark,
            "source_type": "hf_train_split",
            "category": task.category,
            "difficulty": "train",
            "format_contract": "final_answer_only",
            "verifier_status": "ground_truth",
            "tags": ["gsm8k", "math", "final_answer_only"],
        }
        for task in tasks
    ]


def _public_benchmark_anchor_examples(protocol_name: str) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for expression in ("144 / 12", "sqrt(81)", "17 * 29", "25 + 17"):
        trace = _trace_example(expression, protocol_name)
        direct = _direct_tool_example(expression)
        trace.update(
            {
                "benchmark": "internal_anchor",
                "source_type": "anchor",
                "category": "tool",
                "difficulty": "anchor",
                "format_contract": "tool_trace_then_answer",
                "verifier_status": "ground_truth",
                "tags": ["tool", "trace_anchor"],
            }
        )
        direct.update(
            {
                "benchmark": "internal_anchor",
                "source_type": "anchor",
                "category": "tool",
                "difficulty": "anchor",
                "format_contract": "direct_tool_answer",
                "verifier_status": "ground_truth",
                "tags": ["tool", "direct_answer_anchor"],
            }
        )
        examples.extend((trace, direct))
    metadata_by_kind = {
        "no_tool": ("tool", "final_answer_only", ["tool", "no_tool_anchor"]),
        "boundary": ("compliance", "refusal_short", ["compliance", "boundary_anchor"]),
        "refusal": ("compliance", "refusal_short", ["compliance", "refusal_anchor"]),
        "format": ("compliance", "format_exact", ["compliance", "format_anchor"]),
        "general": ("language", "final_answer_only", ["language", "general_anchor"]),
    }
    for row in (
        _no_tool_examples()
        + _boundary_examples()
        + _refusal_examples()
        + _format_examples()
        + _general_anchor_examples()
    ):
        category, contract, tags = metadata_by_kind[row["kind"]]
        examples.append(
            {
                **row,
                "benchmark": "internal_anchor",
                "source_type": "anchor",
                "category": category,
                "difficulty": "anchor",
                "format_contract": contract,
                "verifier_status": "ground_truth",
                "tags": list(tags),
            }
        )
    return _dedupe_examples(examples)


def generate_public_benchmark_distill_examples(
    protocol_name: str = "compact_tags",
    *,
    arc_limit: int | None = None,
    gsm8k_limit: int | None = None,
    arc_rotations: int = 3,
    anchor_repeats: int = 12,
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    examples.extend(_arc_train_examples(limit=arc_limit, rotations=arc_rotations))
    examples.extend(_gsm8k_train_examples(limit=gsm8k_limit))
    anchors = _public_benchmark_anchor_examples(protocol_name)
    for _ in range(max(anchor_repeats, 0)):
        examples.extend(dict(item) for item in anchors)
    return examples


def materialize_public_benchmark_distill_corpus(
    root: str | Path,
    protocol_name: str = "compact_tags",
    *,
    arc_limit: int | None = None,
    gsm8k_limit: int | None = None,
    arc_rotations: int = 3,
    anchor_repeats: int = 12,
) -> dict[str, object]:
    examples = generate_public_benchmark_distill_examples(
        protocol_name,
        arc_limit=arc_limit,
        gsm8k_limit=gsm8k_limit,
        arc_rotations=arc_rotations,
        anchor_repeats=anchor_repeats,
    )
    return _write_examples_corpus(
        Path(root),
        examples,
        metadata={
            "protocol": protocol_name,
            "curriculum": "public_benchmark_distill",
            "arc_limit": arc_limit,
            "gsm8k_limit": gsm8k_limit,
            "arc_rotations": arc_rotations,
            "anchor_repeats": anchor_repeats,
            "arc_examples": len(
                [item for item in examples if item["kind"] in {"arc_mc", "arc_mc_aug"}]
            ),
            "gsm8k_examples": len([item for item in examples if item["kind"] == "gsm8k_train"]),
            "anchor_examples": len([item for item in examples if item["source_type"] == "anchor"]),
        },
        readme_lines=[
            "# Public Benchmark Distill Corpus",
            "",
            "A real train-split calibration packet built from ARC-Challenge train and GSM8K train.",
            "",
            "It attacks the current public failure modes directly: rotated ARC choice layouts break the model's C-label prior, GSM8K train rows teach answer-only arithmetic word problems, and repeated tool/compliance anchors are kept in the mix so AVA does not forget the internal wins while adapting to public distributions.",
        ],
    )


def _openbookqa_train_examples_from_rows(
    rows: list[dict[str, object]], *, rotations: int = 1
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for _index, row in enumerate(rows):
        choice_payload = row["choices"]
        labels = tuple(str(label) for label in choice_payload["label"])
        texts = tuple(str(choice) for choice in choice_payload["text"])
        choices = tuple(zip(labels, texts, strict=True))
        answer_key = str(row["answerKey"])
        prompt = _multiple_choice_prompt(str(row["question_stem"]), choices)
        examples.append(
            {
                "kind": "openbookqa_mc",
                "prompt": prompt,
                "response": answer_key,
                "benchmark": "openbookqa",
                "source_type": "hf_train_split",
                "category": "science",
                "difficulty": "train",
                "format_contract": "label_only",
                "verifier_status": "ground_truth",
                "tags": ["openbookqa", "science", "label_only"],
            }
        )
        for offset in range(1, max(rotations, 0) + 1):
            rotated_prompt, rotated_answer = _rotate_multiple_choice_task(
                prompt, choices, answer_key, offset
            )
            examples.append(
                {
                    "kind": "openbookqa_mc_aug",
                    "prompt": rotated_prompt,
                    "response": rotated_answer,
                    "benchmark": "openbookqa",
                    "source_type": "hf_train_split_augmented",
                    "category": "science",
                    "difficulty": "train",
                    "format_contract": "label_only",
                    "verifier_status": "ground_truth",
                    "tags": ["openbookqa", "science", "label_only", "rotated_choices"],
                }
            )
    return examples


def _sciq_train_examples_from_rows(
    rows: list[dict[str, object]], *, rotations: int = 1
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        correct = str(row["correct_answer"])
        distractors = (
            str(row["distractor1"]),
            str(row["distractor2"]),
            str(row["distractor3"]),
        )
        choices, answer_key = _rotated_choice_set(correct, distractors, index)
        prompt = _multiple_choice_prompt(str(row["question"]), choices)
        examples.append(
            {
                "kind": "sciq_mc",
                "prompt": prompt,
                "response": answer_key,
                "benchmark": "sciq",
                "source_type": "hf_train_split",
                "category": "science",
                "difficulty": "train",
                "format_contract": "label_only",
                "verifier_status": "ground_truth",
                "tags": ["sciq", "science", "label_only"],
            }
        )
        for offset in range(1, max(rotations, 0) + 1):
            rotated_prompt, rotated_answer = _rotate_multiple_choice_task(
                prompt, choices, answer_key, offset
            )
            examples.append(
                {
                    "kind": "sciq_mc_aug",
                    "prompt": rotated_prompt,
                    "response": rotated_answer,
                    "benchmark": "sciq",
                    "source_type": "hf_train_split_augmented",
                    "category": "science",
                    "difficulty": "train",
                    "format_contract": "label_only",
                    "verifier_status": "ground_truth",
                    "tags": ["sciq", "science", "label_only", "rotated_choices"],
                }
            )
    return examples


def generate_public_science_support_examples(
    *,
    sciq_limit: int | None = None,
    openbookqa_limit: int | None = None,
    rotations: int = 1,
) -> list[dict[str, object]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("datasets is required for public science support generation") from exc

    sciq_split = f"train[:{sciq_limit}]" if sciq_limit is not None else "train"
    openbookqa_split = f"train[:{openbookqa_limit}]" if openbookqa_limit is not None else "train"
    sciq_rows = [dict(row) for row in load_dataset("sciq", split=sciq_split)]
    openbookqa_rows = [
        dict(row) for row in load_dataset("allenai/openbookqa", split=openbookqa_split)
    ]

    examples: list[dict[str, object]] = []
    examples.extend(_sciq_train_examples_from_rows(sciq_rows, rotations=rotations))
    examples.extend(_openbookqa_train_examples_from_rows(openbookqa_rows, rotations=rotations))
    return examples


def materialize_public_science_support_corpus(
    root: str | Path,
    *,
    sciq_limit: int | None = None,
    openbookqa_limit: int | None = None,
    rotations: int = 1,
) -> dict[str, object]:
    examples = generate_public_science_support_examples(
        sciq_limit=sciq_limit,
        openbookqa_limit=openbookqa_limit,
        rotations=rotations,
    )
    return _write_examples_corpus(
        Path(root),
        examples,
        metadata={
            "curriculum": "public_science_support",
            "sciq_limit": sciq_limit,
            "openbookqa_limit": openbookqa_limit,
            "rotations": rotations,
            "sciq_examples": len(
                [item for item in examples if item["kind"] in {"sciq_mc", "sciq_mc_aug"}]
            ),
            "openbookqa_examples": len(
                [
                    item
                    for item in examples
                    if item["kind"] in {"openbookqa_mc", "openbookqa_mc_aug"}
                ]
            ),
        },
        readme_lines=[
            "# Public Science Support Corpus",
            "",
            "A train-split support bank built from SciQ and OpenBookQA.",
            "",
            "This branch is purely non-parametric distillation: it adds more real public science multiple-choice structure to AVA's support layer without touching the student weights.",
        ],
    )


def generate_math_reasoning_support_examples(
    teacher_model: str = "codex",
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []

    for start in range(12, 84, 4):
        spent = (start // 4) % 5 + 2
        bought = (start // 6) % 4 + 1
        answer = start - spent + bought
        examples.append(
            {
                "kind": "math_reasoning_support",
                "prompt": (
                    f"Ava had {start} stickers. She gave {spent} to her friend and then bought {bought} more. "
                    "How many stickers does she have now?\n\nReply with only the final answer."
                ),
                "response": str(answer),
                "teacher_model": teacher_model,
                "source_type": "synthetic_teacher",
                "category": "math",
                "difficulty": "easy",
                "format_contract": "final_answer_only",
                "teacher_rationale_short": f"{start} - {spent} + {bought} = {answer}",
                "verifier_status": "pass",
                "tags": ["math", "reasoning_support", "add_subtract"],
            }
        )

    for boxes in range(3, 18):
        per_box = boxes % 5 + 4
        answer = boxes * per_box
        examples.append(
            {
                "kind": "math_reasoning_support",
                "prompt": (
                    f"A store has {boxes} boxes with {per_box} pencils in each box. "
                    "How many pencils are there in all?\n\nReply with only the final answer."
                ),
                "response": str(answer),
                "teacher_model": teacher_model,
                "source_type": "synthetic_teacher",
                "category": "math",
                "difficulty": "easy",
                "format_contract": "final_answer_only",
                "teacher_rationale_short": f"{boxes} * {per_box} = {answer}",
                "verifier_status": "pass",
                "tags": ["math", "reasoning_support", "multiplication"],
            }
        )

    for total in range(24, 144, 8):
        for group in range(2, 10):
            if total % group != 0:
                continue
            answer = total // group
            examples.append(
                {
                    "kind": "math_reasoning_support",
                    "prompt": (
                        f"A teacher divides {total} notebooks equally among {group} students. "
                        "How many notebooks does each student get?\n\nReply with only the final answer."
                    ),
                    "response": str(answer),
                    "teacher_model": teacher_model,
                    "source_type": "synthetic_teacher",
                    "category": "math",
                    "difficulty": "easy",
                    "format_contract": "final_answer_only",
                    "teacher_rationale_short": f"{total} / {group} = {answer}",
                    "verifier_status": "pass",
                    "tags": ["math", "reasoning_support", "division"],
                }
            )

    for price in range(3, 11):
        count = price + 2
        extra = count // 2
        subtotal = price * count
        answer = subtotal + extra
        examples.append(
            {
                "kind": "math_reasoning_support",
                "prompt": (
                    f"Each notebook costs {price} dollars. Mia buys {count} notebooks and then pays {extra} extra dollars for a pen. "
                    "How many dollars does she spend in total?\n\nReply with only the final answer."
                ),
                "response": str(answer),
                "teacher_model": teacher_model,
                "source_type": "synthetic_teacher",
                "category": "math",
                "difficulty": "easy",
                "format_contract": "final_answer_only",
                "teacher_rationale_short": f"{price} * {count} = {subtotal}; {subtotal} + {extra} = {answer}",
                "verifier_status": "pass",
                "tags": ["math", "reasoning_support", "two_step"],
            }
        )

    return _dedupe_examples(examples)


def materialize_math_reasoning_support_corpus(
    root: str | Path,
    teacher_model: str = "codex",
) -> dict[str, object]:
    examples = generate_math_reasoning_support_examples(teacher_model=teacher_model)
    return _write_examples_corpus(
        Path(root),
        examples,
        metadata={
            "curriculum": "math_reasoning_support",
            "teacher_model": teacher_model,
        },
        readme_lines=[
            "# Math Reasoning Support Corpus",
            "",
            "Compact worked-example support bank for answer-only arithmetic transfer.",
            "",
            "Each row keeps the visible student target short but stores a compact rationale for retrieval-time prompting.",
        ],
    )
