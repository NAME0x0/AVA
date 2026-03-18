# AVA Experiment 4: Training & Evaluation Report
Generated: 2026-03-17 20:02:22

## Training Summary

### Qwen3.5-2B-AVA-v1
- **Corpus**: ava_exp4_finetune_fast.jsonl
- **Examples**: 5237
- **Final loss**: 1.0185
- **Time**: 251.0 minutes
- **LoRA rank**: 16
- **Learning rate**: 0.0002
- **GPU memory**: 1.81 GB

## Benchmark Results

### Qwen3.5-2B-AVA-v1
- **arc_challenge**: 66/100 = 66.0% (50s)
- **gsm8k**: 20/50 = 40.0% (1586s)

### Qwen3.5-2B
- **arc_challenge**: 66/100 = 66.0% (73s)
- **gsm8k**: 14/50 = 28.0% (2209s)

## Pipeline Status

- **Phase**: v2_launched
- **V1 ARC**: 66.0%
- **V1 GSM8K**: 40.0%
- **Base ARC**: 66.0%
- **Base GSM8K**: 28.0%
- **V2 proceed**: Yes

## Summary Comparison

| Benchmark            |  Scratch 14M |    Qwen Base |      AVA SFT |    AVA+Tools |
|----------------------|--------------|--------------|--------------|--------------|
| ARC-Challenge        |          24% |        66.0% |        66.0% |          N/A |
| GSM8K                |         0-2% |        28.0% |        40.0% |      pending |
