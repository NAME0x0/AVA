# Reproduce AVA v2

End-to-end recipe to retrain AVA v2 from scratch. Linux is the easier path; Windows works but needs extra setup — see [WINDOWS_SETUP.md](WINDOWS_SETUP.md).

## Hardware

- NVIDIA GPU, ≥4 GB VRAM (Ampere or newer recommended)
- 32 GB system RAM helpful, not strictly required
- ~50 GB free disk for base model + corpus + checkpoints

Tested rig: RTX A2000 Laptop, Windows 11, Python 3.13, PyTorch 2.10.0+cu130.

## 1. Install

```bash
git clone https://github.com/NAME0x0/AVA.git
cd AVA

# Core deps
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install transformers==5.3.0 peft==0.18.1 bitsandbytes==0.49.2 datasets accelerate

# Repo as editable package
pip install -e ".[dev,bench,train]"
```

Optional but recommended:

```bash
# Triton for SDPA kernel acceleration
pip install triton                          # Linux
pip install triton-windows==3.6.0.post26    # Windows — see WINDOWS_SETUP.md

# Flash-Linear-Attention (Qwen3.5 fast path)
pip install flash-linear-attention==0.4.2
```

> **Important**: When FLA is installed, you must set `attn_implementation="sdpa"` in `from_pretrained()`. FLA's weight restructuring is incompatible with BitsAndBytes 4-bit quantized weights.

## 2. Download base model

```bash
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-2B')"
```

## 3. Prepare corpus

The v2 training corpus (`ava_exp4_finetune_v2_augmented.jsonl`) is 20,741 prompt-response pairs across math, science, reasoning, instruction-following, and tool use. JSONL format:

```json
{"prompt": "What causes tides on Earth?", "response": "Tides are primarily caused by..."}
```

To rebuild from scratch:

```bash
python experiments/exp4_finetune/scripts/build_finetune_corpus_v2.py
python experiments/exp4_finetune/scripts/build_v2_augmented.py
```

See [DATA.md](DATA.md) for the corpus mixture rationale.

## 4. Train

```bash
python -u experiments/exp4_finetune/scripts/finetune_v2_full.py > training.log 2>&1
```

Expected: ~2,593 steps, ~100 minutes on the reference laptop GPU. Checkpoints save every 200 steps. Final epoch loss target: ~0.41.

Key config (already in the script):

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1.5e-4,
    lr_scheduler_type="cosine",
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    eval_strategy="no",  # eval OOMs on 4 GB VRAM (248K vocab)
)
```

## 5. Evaluate

```bash
python -u experiments/exp4_finetune/scripts/benchmark_full.py \
    --adapter experiments/exp4_finetune/models/AVA-v2 \
    --arc-limit 100 --gsm8k-limit 50
```

For the full 17-benchmark eval, see `experiments/exp4_finetune/eval_v2/scripts/`.

## 6. Build GGUF for Ollama / llama.cpp

```bash
# Merge adapter into base model
python scripts/convert_to_gguf.py --merge-only

# Full pipeline (needs llama.cpp)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build --config Release -j$(nproc) && cd ..
python scripts/convert_to_gguf.py --llama-cpp ./llama.cpp --quants Q4_K_M Q8_0

# Use with Ollama
ollama create ava-v2 -f Modelfile
ollama run ava-v2
```

The GGUF build also runs in CI — trigger from Actions or publish a Release.

## Software stack reference

| Component | Version | Purpose |
|---|---|---|
| Python | 3.13 | Runtime |
| PyTorch | 2.10.0+cu130 | Tensor compute |
| Transformers | 5.3.0 | Model loading, Trainer |
| PEFT | 0.18.1 | LoRA adapter management |
| BitsAndBytes | 0.49.2 | 4-bit NF4 quantization |
| Triton (Win) | 3.6.0.post26 | GPU kernel compilation |
| Flash-Linear-Attention | 0.4.2 | Qwen3.5 attention backend |
| causal-conv1d | 1.5.0.post8 | FLA dependency (Windows fork) |

## Troubleshooting

- **Triton kernel compile errors on Windows** → [WINDOWS_SETUP.md](WINDOWS_SETUP.md)
- **OOM during model loading** → don't use Unsloth on Windows (it materializes full weights pre-quant). Use vanilla HF Trainer like the script does.
- **OOM during eval** → eval inline is disabled by design. Run the eval as a separate post-training step.
- **Process dies silently** → use `python -u` for unbuffered logs. Windows + redirected stdout buffers everything by default.
