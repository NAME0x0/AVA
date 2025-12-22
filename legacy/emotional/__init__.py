"""
AVA Emotional System

Implements emotional states that influence AVA's behavior, learning rate,
response generation, and tool selection. Emotions include hope, fear, joy,
surprise, and ambition.
"""

from .models import (
    EmotionType,
    EmotionVector,
    EmotionalState,
    EmotionalTrigger,
    TriggerType,
)
from .engine import EmotionalEngine
from .modulation import (
    get_learning_rate_modifier,
    get_response_modulation,
    get_tool_selection_bias,
)

__all__ = [
    "EmotionType",
    "EmotionVector",
    "EmotionalState",
    "EmotionalTrigger",
    "TriggerType",
    "EmotionalEngine",
    "get_learning_rate_modifier",
    "get_response_modulation",
    "get_tool_selection_bias",
]
