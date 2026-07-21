---
base_model: NAME0x0/AVA-v2
license: apache-2.0
language:
  - en
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - ollama
  - lm-studio
  - qlora
  - low-resource
  - reasoning
quantized_by: NAME0x0
---

<p align="center">
  <img src="https://raw.githubusercontent.com/NAME0x0/AVA/main/AVA_logo.png" alt="AVA logo" width="160" />
</p>

# AVA v2 — GGUF

Ready-to-run GGUF builds of [AVA v2](https://huggingface.co/NAME0x0/AVA-v2), a
2B reasoning model fine-tuned entirely on a single 4 GB laptop GPU. **82.0%
ARC-Challenge, 92.0% ARC-Easy, 59.2% MMLU** on a 17-benchmark, 52,027-instance
full evaluation ([report](https://github.com/NAME0x0/AVA/blob/main/experiments/exp4_finetune/eval_v2/RESULTS_REPORT_V2_FULL.md)).

Works with llama.cpp, Ollama, LM Studio, Jan, KoboldCpp — no Python, no GPU
required.

## Files

All sub-8-bit quants are built with an **importance matrix** calibrated on the
model's own training distribution (reasoning, math, science, instruction
following) — the same idea behind Google's Gemma QAT releases: keep the small
quants as close to reference quality as possible.

Measured quality cost vs the Q8_0 reference (perplexity on a held-out slice
of the training distribution, 512-token context — lower is better):

| File | Size | RAM needed | PPL | vs Q8_0 | Use when |
|---|---|---|---|---|---|
| AVA-v2-IQ4_XS.gguf | 1.11 GB | ~1.6 GB | 2.5347 | +2.0% | Tightest fit — old laptops, SBCs |
| AVA-v2-Q4_0.gguf | 1.12 GB | ~1.6 GB | 2.5244 | +1.6% | ARM/AVX-optimized CPU inference |
| **AVA-v2-Q4_K_M.gguf** | 1.19 GB | ~1.7 GB | 2.4907 | **+0.25%** | **Recommended default** |
| AVA-v2-Q5_K_M.gguf | 1.31 GB | ~1.8 GB | — | — | Better quality, still small |
| AVA-v2-Q8_0.gguf | 1.87 GB | ~2.4 GB | 2.4844 | reference | Matches the published eval |

## Quick start

### Ollama

```
ollama run hf.co/NAME0x0/AVA-v2-GGUF:Q4_K_M
```

### llama.cpp

```
llama-cli -m AVA-v2-Q4_K_M.gguf -ngl 99 --temp 0.7 \
  -p "Explain why ice floats on water."
```

### LM Studio / Jan

Search for `NAME0x0/AVA-v2-GGUF` in the model browser and download a file.

## Chat format

Qwen3.5 ChatML-style template (embedded in the GGUF — runtimes apply it
automatically):

```
<|im_start|>user
{your prompt}<|im_end|>
<|im_start|>assistant
```

## Benchmarks (Q8_0, full sets, 95% Wilson CI)

| Benchmark | n | Accuracy |
|---|---|---|
| ARC-Easy | 2,376 | **92.0%** |
| ARC-Challenge | 1,172 | **82.0%** |
| PIQA | 1,838 | 75.9% |
| BoolQ | 3,270 | 75.0% |
| MMLU (5-shot) | 14,042 | **59.2%** |
| GSM8K (greedy / k=5) | 1,319 / 200 | 35.3% / 44.0% |

Full 17-benchmark table and protocol: [RESULTS_REPORT_V2_FULL.md](https://github.com/NAME0x0/AVA/blob/main/experiments/exp4_finetune/eval_v2/RESULTS_REPORT_V2_FULL.md).

At 2B parameters, AVA v2's ARC-Challenge (82.0%) sits ahead of Llama 3.2
3B-Instruct (78.6%) and within two points of Phi-4-mini 3.8B (83.7%) — models
trained with cluster-scale compute. AVA v2 was trained in 100 minutes on one
4 GB laptop GPU.

## Provenance

- **Adapter + training details**: [NAME0x0/AVA-v2](https://huggingface.co/NAME0x0/AVA-v2)
- **Base model**: [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) (Apache 2.0)
- **Everything reproducible**: [github.com/NAME0x0/AVA](https://github.com/NAME0x0/AVA) — corpus builders, training configs, eval harness, and this quantization pipeline are all in the repo.

## Citation

```
@misc{ava-v2-2026,
  title={AVA v2: QLoRA Fine-tuning Under Extreme VRAM Constraints},
  author={Muhammad Afsah Mumtaz},
  year={2026},
  url={https://github.com/NAME0x0/AVA}
}
```
