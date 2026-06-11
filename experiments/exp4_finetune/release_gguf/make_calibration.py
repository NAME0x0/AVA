"""Build an imatrix calibration file for AVA v2.

Samples prompt/response pairs from the v2 post-train mix plus the general SFT
corpus, renders them in the Qwen chat-template token format the deployed model
actually sees, and writes a single UTF-8 text file for llama-imatrix.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

CORPORA = Path(r"D:\AVA\corpora")
OUT = Path(r"D:\AVA\gguf_build\calibration.txt")

SOURCES = [
    (CORPORA / "ava_v2_posttrain_mix_v1" / "examples.jsonl", 2500),
    (CORPORA / "ava_v2_posttrain_assistant_v1" / "examples.jsonl", 800),
    (CORPORA / "ava_v2_posttrain_generalist_v1" / "examples.jsonl", 800),
    (CORPORA / "general_sft" / "examples.jsonl", 500),
    (CORPORA / "gsm8k_reasoning_support_v1" / "examples.jsonl", 500),
    (CORPORA / "gsm8k_train_reasoning_support_v1" / "examples.jsonl", 500),
    (CORPORA / "arc_combined_support_v1" / "examples.jsonl", 400),
    (CORPORA / "arc_train_support_v1" / "examples.jsonl", 400),
]

CHAT = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n"


def main() -> None:
    rng = random.Random(20260611)
    chunks: list[str] = []
    for path, n in SOURCES:
        if not path.exists():
            print(f"skip missing {path}")
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        rng.shuffle(lines)
        taken = 0
        for line in lines:
            if taken >= n:
                break
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt, response = d.get("prompt", ""), d.get("response", "")
            if not prompt or not response:
                continue
            chunks.append(CHAT.format(prompt=prompt, response=response))
            taken += 1
        print(f"{path.parent.name}: {taken} examples")
    rng.shuffle(chunks)
    OUT.write_text("".join(chunks), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
