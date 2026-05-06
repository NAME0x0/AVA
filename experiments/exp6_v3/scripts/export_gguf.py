"""Export trained student to GGUF Q1_0 / TQ1_0 / Q2_K for llama.cpp serving.

P0 stub. Real implementation at P9.

- Reuses scripts/convert_to_gguf.py with --base-model and --quantization flags.
- Targets: TQ1_0 (1.58-bit ternary, smallest), Q2_K (compatibility), Q4_K_M (sanity).
- Generates Modelfile with the Qwen 3.6 chat template.
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError("P9 task")


if __name__ == "__main__":
    main()
