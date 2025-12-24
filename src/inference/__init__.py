"""
AVA Inference System

Implements test-time compute including extended reasoning (thinking)
and self-reflection capabilities.
"""

from .reflection import ReflectionEngine, ReflectionResult
from .thinking import ThinkingEngine, ThinkingResult

__all__ = [
    "ThinkingEngine",
    "ThinkingResult",
    "ReflectionEngine",
    "ReflectionResult",
]
