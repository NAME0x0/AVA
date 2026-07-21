---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
library_name: peft
pipeline_tag: text-generation
language:
- en
tags:
- lora
- qlora
- 4bit
- low-resource
- reasoning
---

# AVA-v2

AVA-v2 is a QLoRA adapter for [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), trained and evaluated entirely on a single NVIDIA RTX A2000 Laptop GPU with 4 GB of VRAM. Training peaks at 1.81 GB of VRAM, runs for about 100 minutes, updates 0.58% of the parameters, and produces a 42 MB adapter.

The base is a vision-language model; only its text transformer is adapted. The vision encoder is unchanged and is not used.

## Training

| Setting | Value |
|---|---|
| Base model | Qwen/Qwen3.5-2B (text transformer) |
| Method | QLoRA (4-bit NF4 base + LoRA) |
| Rank / alpha / dropout | 16 / 32 / 0 |
| Target modules | q, k, v, o, gate, up, down projections |
| Trainable parameters | 10,911,744 (0.58%) |
| Sequence length | 384 |
| Batch x grad. accum. | 1 x 8 |
| Epochs / steps | 1 / 2,593 |
| Learning rate | 1.5e-4 |
| Training examples | 20,741 |
| Peak VRAM | 1.81 GB |
| Wall time | 100.5 min |
| Final loss | 0.4145 |

Training data comes from public train splits (GSM8K, ARC-Challenge, SciQ, OpenBookQA) plus a small set of hand-written examples. No validation or test split of any evaluated benchmark is used.

## Evaluation

Evaluated across 17 benchmark configurations comprising 52,027 evaluation instances, on the merged Q8_0 GGUF, with 95% Wilson confidence intervals.

| Benchmark | n | Accuracy |
|---|---|---|
| ARC-Easy | 2,376 | 92.0% |
| ARC-Challenge | 1,172 | 82.0% |
| MMLU (5-shot) | 14,042 | 59.2% |
| PIQA | 1,838 | 75.9% |
| BoolQ | 3,270 | 75.0% |
| HellaSwag | 10,042 | 56.8% |
| WinoGrande | 1,267 | 56.4% |
| TruthfulQA-MC1 | 817 | 47.5% |
| GSM8K (greedy / k=5) | 1,319 / 200 | 35.3% / 44.0% |
| MMLU-Pro | 12,032 | 30.9% |
| MGSM (en/es/fr) | 750 | 34.4% |
| MBPP+ | 378 | 35.7% |
| IFEval (strict) | 541 | 31.6% |
| HumanEval+ | 164 | 19.5% |
| MATH-500 | 500 | 18.8% |

Strong on science and general knowledge; weak on advanced mathematics, advanced computer science, and strict instruction following.

## Intended use and limitations

AVA-v2 is a compact general-purpose reasoning model for local, low-resource use. It is adapted on the ARC-Challenge and GSM8K training splits, so its scores on those benchmarks reflect in-distribution adaptation rather than zero-shot generalization. The base is a capable recent model, and AVA-v2 does not claim to surpass it in general capability; a matched full-set evaluation of the base is pending.

## Deployment

GGUF builds for CPU and low-VRAM inference: [NAME0x0/AVA-v2-GGUF](https://huggingface.co/NAME0x0/AVA-v2-GGUF). Merged half-precision weights: [NAME0x0/AVA-v2-merged](https://huggingface.co/NAME0x0/AVA-v2-merged). The recommended Q4_K_M build is 1.19 GB and raises perplexity 0.25% over Q8_0.

## Citation

```bibtex
@misc{mumtaz2026avav2,
  title  = {AVA-v2: Reasoning on a 4 GB Laptop GPU},
  author = {Muhammad Afsah Mumtaz},
  year   = {2026},
  url    = {https://github.com/NAME0x0/AVA}
}
```

Code, corpus builders, training configuration, and the full evaluation report: [github.com/NAME0x0/AVA](https://github.com/NAME0x0/AVA).
