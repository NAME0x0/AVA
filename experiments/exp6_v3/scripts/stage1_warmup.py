"""Stage 1: BF16 warmup pretraining of the AVA v3 student.

P0 stub. Real implementation at P3.

- Loads student per configs/v3_student_arch.yaml
- All weights BF16; LoRA on routed experts; full FT on router + shared expert
- 2-3 B tokens of high-quality web + code + math
- Gradient checkpointing + CPU optim (DeepSpeed ZeRO-2)
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError("P3 task")


if __name__ == "__main__":
    main()
