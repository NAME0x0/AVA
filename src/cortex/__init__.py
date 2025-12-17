"""
AVA Cortex - The Synthetic Cognitive Architecture

This module implements the "Bicameral Mind" architecture:

1. CONSCIOUS SYSTEM (Online):
   - ConsciousStream: Real-time processing with Titans Neural Memory
   - Executive: High-level decision making and orchestration
   - Processes inputs, retrieves from memory, reasons with tools

2. SUBCONSCIOUS SYSTEM (Offline):
   - Dreamer: Background consolidation and optimization
   - Integrates with Nested Learning for Fast/Slow weight updates
   - "Dreams" by training on high-quality interactions

The architecture separates:
- Millisecond-scale: Neural Memory updates (surprise-driven)
- Second-scale: Inference and tool use
- Minute-scale: Fast weight adaptation (session learning)
- Hour-scale: Slow weight consolidation (permanent learning)

Reference Papers:
- Titans: Learning to Memorize at Test Time (2025)
- Nested Learning (Google, 2025)
- Toolformer: Language Models Teach Themselves (2023)
"""

from .stream import ConsciousStream, StreamConfig
from .dreaming import Dreamer, DreamerConfig
from .executive import Executive, ExecutiveConfig
from .entropix import Entropix, EntropixConfig, CognitiveState, CognitiveStateLabel

__all__ = [
    "ConsciousStream",
    "StreamConfig",
    "Dreamer",
    "DreamerConfig",
    "Executive",
    "ExecutiveConfig",
    "Entropix",
    "EntropixConfig",
    "CognitiveState",
    "CognitiveStateLabel",
]
