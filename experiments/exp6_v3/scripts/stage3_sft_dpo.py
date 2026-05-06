"""Stage 3: SFT + DPO at ternary (continued QAT).

P0 stub. Real implementation at P5.

- SFT on configs/v3_corpus_mix.yaml SFT mix (~755K examples)
- DPO on configs/v3_corpus_mix.yaml DPO mix (~75K pairs)
- Tool-discrimination SFT on corpora/tool_discrimination_v1.jsonl (300 examples)
- Reuses the trl SFTTrainer + DPOTrainer, with ternary QAT linear layers patched in.
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError("P5 task")


if __name__ == "__main__":
    main()
