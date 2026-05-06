"""Agentic GSM8K eval over llama-server.

Multi-round tool-use loop:
  generate -> parse <tool_call> -> execute calc/python -> inject result -> generate
Up to MAX_ROUNDS rounds. Compare to greedy gsm8k for tool-use lift.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[2]))

from client import LlamaClient
from extractors import extract_numeric, numeric_match
from harness.engine import CalculatorTool, PythonTool

TOOL_SYSTEM = (
    "You are AVA, an AI assistant with tool-use capabilities. "
    "When solving math problems, use the calculator tool for precise computation.\n\n"
    "Available tools:\n"
    "- calculator: Evaluate a mathematical expression. "
    "Supports +, -, *, /, **, sqrt, abs, round.\n"
    "- python: Execute Python code. Use print() to show results.\n\n"
    'To use a tool: <tool_call>{"name": "tool_name", "arguments": {"arg": "value"}}</tool_call>\n'
    "Wait for the tool result before giving your final answer.\n\n"
    "Always end with: The answer is <number>."
)

MAX_ROUNDS = 3
TOOLS = {"calculator": CalculatorTool(), "python": PythonTool()}


def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for m in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        inner = m.group(1).strip()
        try:
            calls.append(json.loads(inner))
            continue
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            end = inner.rfind("}")
            calls.append(json.loads(inner[: end + 1]))
        except Exception:
            continue
    return calls


def _execute(call: dict[str, Any]) -> str:
    name = call.get("name", "")
    args = call.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    tool = TOOLS.get(name)
    if tool is None:
        return f"Unknown tool: {name}"
    try:
        return tool.execute(**args)
    except Exception as e:
        return f"Error: {e}"


async def run_agentic_task(client: LlamaClient, question: str,
                            expected: str) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": TOOL_SYSTEM},
        {"role": "user", "content":
         f"{question}\n\nSolve step by step using tools when helpful."},
    ]
    full_response = ""
    used_tools = False
    for _round in range(MAX_ROUNDS):
        out = await client.chat_generate(
            messages, max_tokens=512, temperature=0.0,
        )
        text = out["text"]
        full_response += text + "\n"
        calls = _parse_tool_calls(text)
        if not calls:
            break
        used_tools = True
        messages.append({"role": "assistant", "content": text})
        for c in calls:
            result = _execute(c)
            messages.append({"role": "user",
                             "content": f"Tool result: {result}"})
    extracted = extract_numeric(full_response)
    matched = numeric_match(extracted, expected)
    return {
        "extracted": extracted,
        "matched": matched,
        "used_tools": used_tools,
        "response": full_response[:600],
    }
