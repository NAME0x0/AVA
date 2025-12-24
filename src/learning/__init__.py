"""
AVA Learning System

Implements continual learning for AVA including:
- Online learning from interactions
- Periodic QLoRA fine-tuning cycles
- Nested learning contexts for meta-learning
- Fast/Slow weight separation (Nested Learning paradigm)
- QLoRA training wrapper for consumer hardware
- Distillation pipelines for self-improvement

Reference Papers:
- "Nested Learning: A New ML Paradigm for Continual Learning" (Google, 2025)
- "QLoRA: Efficient Finetuning of Quantized LLMs" (arXiv:2305.14314)
- "Distilling Step-by-Step!" (ACL Findings, 2023)
"""

from .continual import ContinualLearner, LearningEvent, LearningSample
from .fine_tuning import (
    DistillationConfig,
    DistillationPipeline,
    DistillationSample,
    FineTuningConfig,
    FineTuningResult,
    FineTuningScheduler,
)
from .nested import (
    FastSlowWeightManager,
    LearningScope,
    NestedLearningContext,
    create_nested_learning_system,
)
from .qlora import QLoRAConfig, QLoRATrainer, create_qlora_trainer

__all__ = [
    # Continual Learning
    "ContinualLearner",
    "LearningSample",
    "LearningEvent",
    # Fine-tuning
    "FineTuningScheduler",
    "FineTuningConfig",
    "FineTuningResult",
    # Distillation
    "DistillationPipeline",
    "DistillationConfig",
    "DistillationSample",
    # Nested Learning
    "NestedLearningContext",
    "LearningScope",
    "FastSlowWeightManager",
    "create_nested_learning_system",
    # QLoRA
    "QLoRATrainer",
    "QLoRAConfig",
    "create_qlora_trainer",
]

