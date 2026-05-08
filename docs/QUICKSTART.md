# Quickstart

Three ways to run AVA v2. Pick one.

## 1. Ollama (easiest, no Python)

Works on CPU, Apple Silicon, AMD, NVIDIA. Needs ~2 GB RAM/VRAM.

```bash
# Download a GGUF from HuggingFace
# https://huggingface.co/NAME0x0/AVA-v2-GGUF

ollama create ava-v2 -f Modelfile
ollama run ava-v2
```

`Modelfile` lives at the repo root.

## 2. Python (LoRA adapter on the base model)

Needs CUDA + ≥4 GB VRAM.

```bash
git clone https://github.com/NAME0x0/AVA.git
cd AVA
pip install -e ".[bench]"
pip install peft

python scripts/chat.py
# or one-shot:
python scripts/chat.py --prompt "Why does ice float on water?"
```

Auto-downloads the adapter from HuggingFace.

## 3. Inline Python API

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
m = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-2B", quantization_config=bnb,
    device_map="auto", dtype=torch.bfloat16, attn_implementation="sdpa",
)
m = PeftModel.from_pretrained(m, "NAME0x0/AVA-v2").merge_and_unload()

msgs = [{"role": "user", "content": "Why does ice float on water?"}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").to(m.device)
out = m.generate(**ids, max_new_tokens=512, temperature=0.7, do_sample=True)
print(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Requirements

| Path | OS | Hardware | Python |
|---|---|---|---|
| Ollama GGUF | any | 2 GB free | not required |
| Python adapter | Linux / macOS / Windows | NVIDIA GPU, 4 GB+ VRAM | 3.10+ |

Windows users hitting Triton/FLA build errors: see [WINDOWS_SETUP.md](WINDOWS_SETUP.md).

## What now

- See [RESULTS.md](RESULTS.md) for the full 17-benchmark eval.
- See [COMPARE.md](COMPARE.md) for cross-model comparison.
- See [REPRODUCE.md](REPRODUCE.md) to retrain from scratch.
