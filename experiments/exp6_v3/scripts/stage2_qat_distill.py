"""Stage 2: Ternary QAT + Qwen 3.6 35B-A3B teacher distillation.

P0 stub. Real implementation at P4.

Pipeline:
- Quantize routed expert weights to ternary (BF16 master kept for gradient).
- ParetoQ unified QAT scheduler.
- Loss: 0.5 * CE + 0.3 * KL(teacher || student) + 0.2 * MiniLM-attn-distill.
- Teacher served via experiments/exp5_gemma4/engine/streaming.py (int4 stream).
- 5-10 B tokens; ZeRO-2 CPU optim, micro-batch 1, grad-accum 32, seq 4096.
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError("P4 task")


if __name__ == "__main__":
    main()
