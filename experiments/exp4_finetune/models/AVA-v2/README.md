---
library_name: peft
base_model: Qwen/Qwen3.5-2B
license: apache-2.0
tags:
  - qlora
  - 4bit
  - low-resource
  - arc-challenge
  - gsm8k
  - science
  - math
  - reasoning
datasets:
  - custom
language:
  - en
pipeline_tag: text-generation
---

# AVA v2

AVA v2 is a QLoRA fine-tune of [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) that achieves **79% on ARC-Challenge** and **48% on GSM8K** while training and running inference in under 2 GB of VRAM.

Trained entirely on a single NVIDIA RTX A2000 Laptop GPU (4 GB VRAM). The adapter is 42 MB.

## Results

| Benchmark | Qwen3.5-2B Base | AVA v2 | Improvement |
|---|---|---|---|
| ARC-Challenge (100) | 66.0% | **79.0%** | +13.0pp |
| GSM8K (50) | 28.0% | **48.0%** | +20.0pp |

### Comparison to Other Small Models

| Model | Params | ARC-C | GSM8K |
|---|---|---|---|
| Gemma 2 2B | 2.0B | 55.7% | 24.3% |
| SmolLM2-1.7B-Instruct | 1.7B | ~52% | 48.2% |
| Llama 3.2 1B-Instruct | 1.0B | 59.4% | 44.4% |
| Llama 3.2 3B-Instruct | 3.0B | 78.6% | 77.7% |
| **AVA v2** | **2.0B** | **79.0%** | **48.0%** |

AVA v2's ARC-Challenge score at 2B parameters exceeds Llama 3.2 3B-Instruct (78.6% at 3B).

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-2B",
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")

model = PeftModel.from_pretrained(model, "NAME0x0/AVA-v2")
model = model.merge_and_unload()

messages = [{"role": "user", "content": "Explain why ice floats on water."}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Training Details

- **Method**: QLoRA (4-bit NF4 + LoRA rank 16)
- **Base model**: Qwen3.5-2B
- **Training data**: 20,741 prompt-response pairs (math, science, reasoning, instruction following)
- **Hardware**: NVIDIA RTX A2000 Laptop (4 GB VRAM)
- **Training time**: 100.5 minutes
- **Final loss**: 0.4145
- **Peak VRAM**: 1.81 GB
- **Trainable params**: 10,911,744 / 1,892,736,832 (0.58%)
- **Optimizer**: paged_adamw_8bit
- **LR schedule**: cosine, peak 1.5e-4
- **Batch size**: 1 (gradient accumulation 8, effective batch 8)
- **Max sequence length**: 384 tokens
- **Epochs**: 1

## Limitations

- Evaluation was run on 100 ARC-Challenge and 50 GSM8K items (not full test sets)
- Evaluation protocols (shot count, prompting) differ across model comparison sources
- The model inherits Qwen3.5-2B's base capabilities and limitations
- Max training sequence length was 384 tokens due to VRAM constraints

## Citation

```
@misc{ava-v2-2026,
  title={AVA v2: QLoRA Fine-tuning Under Extreme VRAM Constraints},
  author={Afsah},
  year={2026},
  url={https://github.com/NAME0x0/AVA}
}
```
