"""Build the AVA v3 SFT + DPO mix from configs/v3_corpus_mix.yaml.

P0 stub. Real implementation at P3 / P5.

Pipeline:
1. Load each entry, sample N rows according to `sample` field.
2. Apply per-source cleaning + length filters.
3. Render to AVA chat-template format.
4. Shuffle (seed=42), shard, write to corpora/v3_sft_v1.jsonl + v3_dpo_v1.jsonl.
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError("P3 task — implement corpus builder")


if __name__ == "__main__":
    main()
