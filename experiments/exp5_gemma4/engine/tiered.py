"""Routing helpers for fast-path / deep-path local Gemma 4 serving."""
from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_DEEP_KEYWORDS = (
    "analyze",
    "architecture",
    "compare",
    "debug",
    "derive",
    "design",
    "explain why",
    "implement",
    "optimize",
    "prove",
    "reason",
    "refactor",
    "step by step",
    "tradeoff",
)

DEFAULT_LONG_CONTEXT_HINTS = (
    "conversation so far",
    "document below",
    "full transcript",
    "long context",
    "paper below",
    "use the context",
)

DEFAULT_CODE_HINTS = (
    "```",
    "Traceback",
    "TypeError",
    "ValueError",
    "class ",
    "def ",
    "import ",
)


@dataclass(slots=True)
class RoutingConfig:
    """Heuristics for the local fast/deep router."""

    fast_model_id: str = "google/gemma-4-E2B-it"
    deep_model_id: str = "google/gemma-4-E4B-it"
    max_fast_chars: int = 320
    max_fast_words: int = 72
    max_fast_lines: int = 6
    deep_keywords: tuple[str, ...] = DEFAULT_DEEP_KEYWORDS
    long_context_hints: tuple[str, ...] = DEFAULT_LONG_CONTEXT_HINTS
    code_hints: tuple[str, ...] = DEFAULT_CODE_HINTS
    force_fast_prefixes: tuple[str, ...] = ("fast:", "quick:", "e2b:")
    force_deep_prefixes: tuple[str, ...] = ("deep:", "slow:", "e4b:")
    force_reason_prefixes: tuple[str, ...] = ("reason:", "think:", "reason+:", "e4b-think:")


@dataclass(slots=True)
class PromptFeatures:
    chars: int
    words: int
    lines: int
    keyword_hits: tuple[str, ...] = field(default_factory=tuple)
    long_context_hits: tuple[str, ...] = field(default_factory=tuple)
    code_hits: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class RouteDecision:
    tier: str
    reason: str
    cleaned_prompt: str
    features: PromptFeatures
    thinking_override: bool | None = None


def analyze_prompt(prompt: str, config: RoutingConfig | None = None) -> PromptFeatures:
    """Extract lightweight prompt features for local routing decisions."""
    cfg = config or RoutingConfig()
    stripped = prompt.strip()
    lowered = stripped.lower()
    words = [word for word in stripped.replace("\n", " ").split(" ") if word]

    keyword_hits = tuple(keyword for keyword in cfg.deep_keywords if keyword in lowered)
    long_context_hits = tuple(hint for hint in cfg.long_context_hints if hint in lowered)
    code_hits = tuple(hint for hint in cfg.code_hints if hint in prompt)

    return PromptFeatures(
        chars=len(stripped),
        words=len(words),
        lines=max(1, stripped.count("\n") + 1) if stripped else 0,
        keyword_hits=keyword_hits,
        long_context_hits=long_context_hits,
        code_hits=code_hits,
    )


def _strip_prefix(prompt: str, prefixes: tuple[str, ...]) -> str:
    stripped = prompt.strip()
    lowered = stripped.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return stripped[len(prefix):].lstrip()
    return stripped


def route_prompt(prompt: str, config: RoutingConfig | None = None) -> RouteDecision:
    """Route a prompt to the fast or deep local model."""
    cfg = config or RoutingConfig()
    stripped = prompt.strip()
    lowered = stripped.lower()

    if any(lowered.startswith(prefix) for prefix in cfg.force_fast_prefixes):
        cleaned = _strip_prefix(stripped, cfg.force_fast_prefixes)
        features = analyze_prompt(cleaned, cfg)
        return RouteDecision("fast", "explicit fast override", cleaned, features, thinking_override=False)

    if any(lowered.startswith(prefix) for prefix in cfg.force_reason_prefixes):
        cleaned = _strip_prefix(stripped, cfg.force_reason_prefixes)
        features = analyze_prompt(cleaned, cfg)
        return RouteDecision("deep", "explicit reasoning override", cleaned, features, thinking_override=True)

    if any(lowered.startswith(prefix) for prefix in cfg.force_deep_prefixes):
        cleaned = _strip_prefix(stripped, cfg.force_deep_prefixes)
        features = analyze_prompt(cleaned, cfg)
        return RouteDecision("deep", "explicit deep override", cleaned, features, thinking_override=False)

    features = analyze_prompt(stripped, cfg)
    reasons: list[str] = []

    if features.chars > cfg.max_fast_chars:
        reasons.append(f"prompt chars {features.chars}>{cfg.max_fast_chars}")
    if features.words > cfg.max_fast_words:
        reasons.append(f"prompt words {features.words}>{cfg.max_fast_words}")
    if features.lines > cfg.max_fast_lines:
        reasons.append(f"prompt lines {features.lines}>{cfg.max_fast_lines}")
    if features.keyword_hits:
        reasons.append(f"deep keywords: {', '.join(features.keyword_hits[:3])}")
    if features.long_context_hits:
        reasons.append(f"context hints: {', '.join(features.long_context_hits[:2])}")
    if features.code_hits:
        reasons.append(f"code hints: {', '.join(features.code_hits[:2])}")

    if reasons:
        return RouteDecision("deep", "; ".join(reasons), stripped, features)

    return RouteDecision("fast", "short prompt within fast-path limits", stripped, features)
