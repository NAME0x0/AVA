---
base_model: Qwen/Qwen3.5-2B
license: apache-2.0
language:
  - en
pipeline_tag: text-generation
tags:
  - merged
  - qlora
  - low-resource
  - reasoning
---

<p align="center">
  <img src="https://raw.githubusercontent.com/NAME0x0/AVA/main/AVA_logo.png" alt="AVA logo" width="160" />
</p>

# AVA v2 (merged weights)

Standalone bf16 weights of [AVA v2](https://huggingface.co/NAME0x0/AVA-v2) —
the QLoRA adapter pre-merged into [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B).
Load directly with `transformers`, no PEFT required:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "NAME0x0/AVA-v2-merged", device_map="auto", dtype="bfloat16"
)
tokenizer = AutoTokenizer.from_pretrained("NAME0x0/AVA-v2-merged")
```

- **Benchmarks, training details, limitations**: see the [adapter card](https://huggingface.co/NAME0x0/AVA-v2) — 82.0% ARC-Challenge, 92.0% ARC-Easy, 59.2% MMLU at 2B params, trained on a single 4 GB laptop GPU.
- **No Python / CPU-only**: use the [GGUF builds](https://huggingface.co/NAME0x0/AVA-v2-GGUF) (Ollama, llama.cpp, LM Studio).
- **Reproduce everything**: [github.com/NAME0x0/AVA](https://github.com/NAME0x0/AVA).
