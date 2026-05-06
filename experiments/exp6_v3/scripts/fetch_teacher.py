"""Download Qwen 3.6 teacher weights for AVA v3 distillation.

P0 stub. Real implementation at P1.

Targets:
  - Qwen/Qwen3.6-35B-A3B (primary teacher, MoE 256 experts, 3B active)
  - Qwen/Qwen3.6-27B (fallback dense teacher)

Both via huggingface_hub.snapshot_download to D:/AVA/experiments/exp6_v3/models/.
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError("P1 task — implement HF snapshot_download")


if __name__ == "__main__":
    main()
