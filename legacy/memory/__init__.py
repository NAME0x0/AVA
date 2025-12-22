"""
AVA Memory System

Implements persistent memory for AVA including:
- Episodic memory: Event-based memories of interactions
- Semantic memory: Factual knowledge storage
- Memory consolidation: Decay and strengthening over time

Memory is emotionally tagged and influences future interactions.
"""

from .models import (
    MemoryType,
    MemoryItem,
    EpisodicMemory,
    SemanticMemory,
)
from .episodic import EpisodicMemoryStore
from .semantic import SemanticMemoryStore
from .consolidation import MemoryConsolidator
from .manager import MemoryManager

__all__ = [
    "MemoryType",
    "MemoryItem",
    "EpisodicMemory",
    "SemanticMemory",
    "EpisodicMemoryStore",
    "SemanticMemoryStore",
    "MemoryConsolidator",
    "MemoryManager",
]
