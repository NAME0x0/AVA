"""
AVA Tool System

Implements tools with safety levels that unlock based on developmental stage.
Tools progress from baby-safe (Level 0) to mature (Level 5).
"""

from .base_tools import (
    # Level 1 - Toddler
    calculator_tool,
    current_time_tool,
    # Level 0 - Baby Safe
    echo_tool,
    simple_math_tool,
    word_count_tool,
    # Level 2 - Child
    # web_search_tool,  # To be implemented
)
from .progression import ToolProgressionManager
from .registry import ToolAccessDecision, ToolDefinition, ToolRegistry, ToolSafetyLevel

__all__ = [
    "ToolRegistry",
    "ToolSafetyLevel",
    "ToolDefinition",
    "ToolAccessDecision",
    "ToolProgressionManager",
    "echo_tool",
    "current_time_tool",
    "simple_math_tool",
    "calculator_tool",
    "word_count_tool",
]
