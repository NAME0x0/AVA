"""Answer extraction + matching for generative benchmarks."""
from __future__ import annotations

import re


_NUM_PATTERNS = [
    r"(?:the answer is|final answer is|final answer:|answer is|answer:)\s*\$?([-\d,]+\.?\d*)",
    r"####\s*\$?([-\d,]+\.?\d*)",
    r"\\boxed\{([^}]*)\}",
    r"(?:therefore|thus|so|hence)[,:]?\s+\$?([-\d,]+\.?\d*)",
    r"=\s*\$?([-\d,]+\.?\d*)\s*$",
    r"\$?([-\d,]+\.?\d*)\s*(?:dollars|miles|hours|people|items|pieces|years|cents)?\s*\.?\s*$",
]


def extract_numeric(text: str) -> str | None:
    text = text.strip()
    for pat in _NUM_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            val = m.group(1).strip().replace(",", "").rstrip(".")
            if val and val != "-":
                return val
    nums = re.findall(r"-?\d+\.?\d*", text)
    if nums:
        return nums[-1].replace(",", "").rstrip(".")
    return None


def numeric_match(predicted: str | None, expected: str) -> bool:
    if not predicted:
        return False
    try:
        return abs(float(predicted) - float(expected)) < 1e-2
    except (ValueError, TypeError):
        return predicted.strip() == expected.strip()


_BOX_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_boxed(text: str) -> str | None:
    matches = _BOX_RE.findall(text)
    if matches:
        return matches[-1].strip()
    m = re.search(r"(?:the answer is|final answer:?)\s*([^\n.]+)", text,
                  re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")
    return None


def normalize_math(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\\!", "", s)
    s = re.sub(r"\\,", "", s)
    s = re.sub(r"\\\\", r"\\\\", s)
    s = re.sub(r"\s+", "", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.rstrip(".")
    return s


def math_match(predicted: str | None, expected: str) -> bool:
    if not predicted:
        return False
    p = normalize_math(predicted)
    e = normalize_math(expected)
    if p == e:
        return True
    try:
        return abs(float(p) - float(e)) < 1e-3
    except (ValueError, TypeError):
        return False


def extract_python_code(text: str) -> str:
    import json as _json
    # tool_call JSON form: <tool_call>{"name": "python", "arguments": {"code": "..."}}</tool_call>
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL):
        inner = m.group(1)
        try:
            obj = _json.loads(inner)
        except _json.JSONDecodeError:
            try:
                end = inner.rfind("}")
                obj = _json.loads(inner[: end + 1])
            except Exception:
                continue
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = _json.loads(args)
            except Exception:
                args = {}
        code = args.get("code") or args.get("source") or args.get("python")
        if code:
            return code
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[0]
    return text
