"""Distillation loss + teacher serving for AVA v3 stage-2 QAT.

Loss (per token):
    L = 0.5 * CE(student, label) +
        0.3 * KL(softmax(teacher_logits / T) || softmax(student_logits / T)) * T**2 +
        0.2 * MiniLM_attn_distill(student_attn_maps, teacher_attn_maps)

Teacher: Qwen 3.6 35B-A3B served via the Exp5 streaming int4 loader
(experiments/exp5_gemma4/engine/streaming.py). Logits are pre-cached to disk
before training when token IDs are deterministic, otherwise streamed live.

P0 stub. Real implementation lands at P4.

References:
- BitDistiller / BitNet Distillation: arxiv.org/abs/2510.13998
- MiniLM attention distillation: arxiv.org/abs/2002.10957
- DistilLLM techniques 2025: (cite TBD)
"""
from __future__ import annotations

from typing import Any


def distill_loss_step(
    student_logits: Any,
    teacher_logits: Any,
    labels: Any,
    *,
    student_attn: Any | None = None,
    teacher_attn: Any | None = None,
    temperature: float = 2.0,
    ce_weight: float = 0.5,
    kl_weight: float = 0.3,
    attn_weight: float = 0.2,
) -> Any:
    raise NotImplementedError("P4 task")
