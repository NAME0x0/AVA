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
  - mmlu
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

AVA v2 is a 42 MB QLoRA adapter for [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), trained and evaluated entirely on a single NVIDIA RTX A2000 Laptop GPU with 4 GB VRAM. It targets strong general-purpose reasoning at the 2B scale on consumer hardware.

On a 17-benchmark / 16,872-task full evaluation at Q8_0 GGUF, AVA v2 reaches:

- **82.0%** ARC-Challenge (1,172 questions)
- **92.0%** ARC-Easy (2,376 questions)
- **75.9%** PIQA · **75.0%** BoolQ
- **59.2%** MMLU 5-shot (14,042 questions)
- **35.3%** GSM8K greedy / **44.0%** with k=5 self-consistency
- **30.9%** MMLU-Pro · **18.8%** MATH-500
- **35.7%** MBPP+ · **19.5%** HumanEval+

Training peaked at **1.81 GB VRAM** and finished in **100 minutes**. Inference fits in under 2 GB of VRAM.

Full report: [AVA repo / RESULTS_REPORT_V2_FULL.md](https://github.com/NAME0x0/AVA/blob/main/experiments/exp4_finetune/eval_v2/RESULTS_REPORT_V2_FULL.md).

## Full Benchmark Results

All scores Q8_0 GGUF on llama-server (Flash Attention, Q8 KV cache). 95% Wilson confidence intervals.

| Benchmark | n | Accuracy | 95% CI |
|---|---|---|---|
| ARC-Easy | 2,376 | **92.0%** | [90.8, 93.0] |
| ARC-Challenge | 1,172 | **82.0%** | [79.7, 84.1] |
| PIQA | 1,838 | **75.9%** | [73.9, 77.8] |
| BoolQ | 3,270 | **75.0%** | [73.5, 76.5] |
| MMLU (5-shot) | 14,042 | **59.2%** | [58.4, 60.1] |
| HellaSwag | 10,042 | **56.8%** | [55.8, 57.8] |
| WinoGrande XL | 1,267 | **56.4%** | [53.7, 59.1] |
| TruthfulQA-MC1 | 817 | **47.5%** | [44.1, 50.9] |
| GSM8K self-cons (k=5) | 200 | **44.0%** | [37.3, 50.9] |
| MBPP+ | 378 | **35.7%** | [31.0, 40.7] |
| Agentic GSM8K (calc/python) | 1,319 | **35.4%** | [32.9, 38.0] |
| GSM8K (greedy) | 1,319 | **35.3%** | [32.8, 38.0] |
| MGSM (en/es/fr) | 750 | **34.4%** | [31.1, 37.9] |
| IFEval (strict) | 541 | **31.6%** | [27.8, 35.6] |
| MMLU-Pro | 12,032 | **30.9%** | [30.1, 31.8] |
| HumanEval+ | 164 | **19.5%** | [14.2, 26.3] |
| MATH-500 | 500 | **18.8%** | [15.6, 22.5] |

## Comparison to Other Small Models

Reported scores from official model cards / technical reports. Evaluation protocols differ (shot count, prompting). AVA v2 numbers from the full eval above; AVA v2 GSM8K shown as `greedy / k=5 self-cons`.

| Model | Params | ARC-C | MMLU | HellaSwag | GSM8K |
|---|---|---|---|---|---|
| TinyLlama 1.1B-Chat | 1.1B | 30.1 | 25.3 | 60.3 | 2.0 |
| Llama 3.2 1B-Instruct | 1.0B | 59.4 | 49.3 | 60.8 | 44.4 |
| Qwen2.5 1.5B-Instruct | 1.5B | 54.7 | 60.9 | 67.9 | 68.5 |
| SmolLM2 1.7B-Instruct | 1.7B | 52.0 | 50.4 | 68.9 | 48.2 |
| Gemma 2 2B-Instruct | 2.0B | 55.7 | 51.3 | 73.0 | 24.3 |
| Qwen3.5 2B Base | 2.0B | 66.0 | — | — | 28.0 |
| **AVA v2 (this model)** | **2.0B** | **82.0** | **59.2** | 56.8 | **35.3 / 44.0** |
| Qwen2.5 3B-Instruct | 3.0B | ~70 | 65.6 | 73.6 | 79.1 |
| Llama 3.2 3B-Instruct | 3.0B | 78.6 | 63.4 | 69.8 | 77.7 |
| Phi-4-mini 3.8B-Instruct | 3.8B | 83.7 | 67.3 | 76.2 | 88.6 |
| Phi-3.5-mini-Instruct | 3.8B | 84.6 | 69.0 | 69.4 | 86.2 |
| Mistral 7B-Instruct v0.2 | 7.0B | 55.5 | 60.1 | 81.3 | 52.2 |

Where AVA v2 stands at 2B:

- **ARC-Challenge** (science reasoning): 82.0% on the full 1,172-question set, ahead of Llama 3.2 3B-Instruct (78.6%) and competitive with Phi-4-mini 3.8B (83.7%) and Phi-3.5-mini 3.8B (84.6%).
- **MMLU** (general knowledge): 59.2% — close to Mistral 7B (60.1%) and roughly +8pp over Gemma 2 2B (51.3%).
- **GSM8K** (math): the main weak area. Greedy 35.3% trails Qwen2.5 1.5B (68.5%) and the Phi/Llama 3B-class models. With k=5 self-consistency, 44.0% lands near Llama 3.2 1B-IT (44.4%).
- **HellaSwag** (commonsense narrative): 56.8% — below most peers. The fine-tune corpus emphasized science and instruction-following, not narrative completion.

The headline win is reasoning-on-tiny-hardware: a 42 MB adapter trained on 4 GB VRAM gets 82% ARC-C, where most peers above used cluster-scale training compute.

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
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

For a no-Python path, GGUF builds (Q4_K_M, Q8_0) are at [`NAME0x0/AVA-v2-GGUF`](https://huggingface.co/NAME0x0/AVA-v2-GGUF) and run on Ollama, llama.cpp, LM Studio, etc.

## Training Details

- **Method**: QLoRA (4-bit NF4 base + LoRA rank 16, alpha 32 on all attention + MLP projections)
- **Base model**: Qwen3.5-2B (1.89B parameters)
- **Training data**: 20,741 prompt-response pairs (math, science, reasoning, instruction-following, tool-use)
- **Hardware**: NVIDIA RTX A2000 Laptop, 4 GB VRAM, single GPU
- **Training time**: 100.5 minutes (2,593 steps)
- **Final epoch loss**: 0.4145
- **Peak VRAM**: 1.81 GB
- **Trainable parameters**: 10,911,744 / 1,892,736,832 (0.58%)
- **Optimizer**: paged_adamw_8bit
- **LR schedule**: cosine, peak 1.5e-4
- **Effective batch size**: 8 (per-device 1 × grad-accum 8)
- **Max sequence length**: 384 tokens
- **Epochs**: 1
- **Attention backend**: SDPA (Triton-compiled)

## Limitations

- **Math is weak.** GSM8K 35.3% greedy / 44.0% k=5; MATH-500 18.8%. Self-consistency is the cheapest reasoning lever before re-training.
- **Tool-use is mostly latent.** Agentic GSM8K invoked the calculator on only 0.6% of problems despite tool examples in the SFT corpus. The model defaults to direct chain-of-thought.
- **Multilingual transfer is partial.** MGSM en 42.8% → es 32.0% → fr 28.4%.
- **Max training sequence length was 384 tokens.** Long-form reasoning chains beyond that range were not seen during training.
- **MMLU 5-shot context overflow.** 2.7% of MMLU items errored on a 8K context cap (long sub-categories like `professional_law`); they are counted as failures. Accuracy on completed items was 60.8%.
- **MCQ scoring is letter-argmax.** AVA's eval uses 1-token argmax over candidate label tokens via `/completion n_probs=60`, which differs slightly from lm-evaluation-harness's logprob-of-continuation scoring. Numbers are directionally comparable to leaderboards but not numerically identical.

## More documentation

- **Reproduce**: [docs/REPRODUCE.md](https://github.com/NAME0x0/AVA/blob/main/docs/REPRODUCE.md)
- **Windows setup (Triton / FLA / BnB)**: [public gist](https://gist.github.com/NAME0x0/8fe9084e606d3e7ae17d4f1da6a96667)
- **Full eval report**: [RESULTS_REPORT_V2_FULL.md](https://github.com/NAME0x0/AVA/blob/main/experiments/exp4_finetune/eval_v2/RESULTS_REPORT_V2_FULL.md)
- **Cross-model comparison**: [docs/COMPARE.md](https://github.com/NAME0x0/AVA/blob/main/docs/COMPARE.md)
- **Experiment progression (v1 → v2 → v3)**: [docs/EXPERIMENTS.md](https://github.com/NAME0x0/AVA/blob/main/docs/EXPERIMENTS.md)
- **Roadmap**: [docs/ROADMAP.md](https://github.com/NAME0x0/AVA/blob/main/docs/ROADMAP.md)

## Citation

```
@misc{ava-v2-2026,
  title={AVA v2: QLoRA Fine-tuning Under Extreme VRAM Constraints},
  author={Muhammad Afsah Mumtaz},
  year={2026},
  url={https://github.com/NAME0x0/AVA}
}
```
