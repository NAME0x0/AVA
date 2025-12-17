"""
AVA Hippocampus Module - The Memory Formation System

This module implements the biological analog of the hippocampus:
- Short-term memory consolidation
- Episodic event storage with surprise tagging
- Test-time learning via Titans Neural Memory
- Replay buffer for offline consolidation

The hippocampus serves as the bridge between:
1. The Conscious Stream (immediate processing)
2. The Subconscious Dreamer (offline consolidation)
"""

from .titans import TitansSidecar, TitansSidecarConfig
from .episodic_buffer import EpisodicBuffer, Episode, BufferConfig

__all__ = [
    "TitansSidecar",
    "TitansSidecarConfig",
    "EpisodicBuffer",
    "Episode",
    "BufferConfig",
]
