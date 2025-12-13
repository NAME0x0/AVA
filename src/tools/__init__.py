"""
AVA Tool System

Implements tools with safety levels that unlock based on developmental stage.
Tools progress from baby-safe (Level 0) to mature (Level 5).
"""

from .registry import ToolRegistry, ToolSafetyLevel, ToolDefinition, ToolAccessDecision
from .progression import ToolProgressionManager
from .base_tools import (
    # Level 0 - Baby Safe
    echo_tool,
    current_time_tool,
    simple_math_tool,
    # Level 1 - Toddler
    calculator_tool,
    word_count_tool,
    # Level 2 - Child
    # web_search_tool,  # To be implemented
)

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
