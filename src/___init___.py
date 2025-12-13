"""
AVA - Developmental AI

An AI that learns and matures like a human child, starting with
limited articulation and progressively improving through interaction.

Main Components:
- DevelopmentalAgent: The main agent class
- DevelopmentTracker: Tracks developmental stage and maturation
- EmotionalEngine: Processes emotions that influence behavior
- MemoryManager: Manages episodic and semantic memories
- ToolRegistry: Manages tools with developmental gating
- ThinkingEngine: Test-time compute for extended reasoning
- ContinualLearner: Collects samples for ongoing learning
"""

__version__ = "2.0.0"

from .agent import DevelopmentalAgent, InteractionResult

__all__ = [
    "DevelopmentalAgent",
    "InteractionResult",
]
