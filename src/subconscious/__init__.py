"""
AVA Subconscious Module - Offline Consolidation System

This module implements the "sleeping" brain:
- The Nightmare Engine: QLoRA fine-tuning during idle periods
- Sleep cycle management
- Experience replay from the episodic buffer
- Knowledge distillation from high-quality interactions

The subconscious operates when AVA is "asleep":
1. Sample high-surprise episodes from the episodic buffer
2. Generate augmented training data (CoT, tools)
3. Run QLoRA fine-tuning to consolidate knowledge
4. Update slow weights in the nested learning system
"""

from .nightmare import NightmareEngine, NightmareConfig, SleepPhase

__all__ = [
    "NightmareEngine",
    "NightmareConfig",
    "SleepPhase",
]
