"""
AVA Source Package
==================

Core modules:
- ava: Main AVA interface (new, clean API)
- core: Cortex-Medulla architecture  
- tools: Tool system
- memory: Memory management
- cortex: Cognitive processing
- hippocampus: Neural memory
"""

# New clean API
from .ava import AVA, AVAConfig, Response

__version__ = "3.1.0"
__all__ = ["AVA", "AVAConfig", "Response"]
