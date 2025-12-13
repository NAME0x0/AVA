"""
AVA Learning System

Implements continual learning for AVA including:
- Online learning from interactions
- Periodic QLoRA fine-tuning cycles
- Nested learning contexts for meta-learning
"""

from .continual import ContinualLearner, LearningSample, LearningEvent
from .fine_tuning import FineTuningScheduler, FineTuningConfig, FineTuningResult
from .nested import NestedLearningContext, LearningScope

__all__ = [
    "ContinualLearner",
    "LearningSample",
    "LearningEvent",
    "FineTuningScheduler",
    "FineTuningConfig",
    "FineTuningResult",
    "NestedLearningContext",
    "LearningScope",
]
