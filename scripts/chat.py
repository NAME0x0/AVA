"""Chat with AVA v2 — interactive inference on a single GPU.

Usage:
    # From HuggingFace (downloads automatically):
    python scripts/chat.py

    # From a local adapter directory:
    python scripts/chat.py --adapter ./experiments/exp4_finetune/models/AVA-v2

    # Single-shot mode (no interactive loop):
    python scripts/chat.py --prompt "Explain why ice floats on water."

    # Adjust generation parameters:
    python scripts/chat.py --max-tokens 1024 --temperature 0.6
"""
from __future__ import annotations

import argparse
import os
import sys

# Triton CC auto-detect (Windows)
if sys.platform == "win32" and not os.environ.get("CC"):
    _candidates = [
        r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36014\bin\Hostx64\x64\cl.exe",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe",
    ]
    for _cc in _candidates:
        if os.path.exists(_cc):
            os.environ["CC"] = _cc
            break

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HF_REPO = "NAME0x0/AVA-v2"
BASE_MODEL = "Qwen/Qwen3.5-2B"


def load_model(adapter: str) -> tuple:
    """Load AVA v2 in 4-bit and merge the LoRA adapter."""
    print(f"Loading base model: {BASE_MODEL}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print(f"Loading adapter: {adapter}")
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    model.eval()

    vram_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"Ready. VRAM used: {vram_gb:.2f} GB\n")
    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate a response for a single user message."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
        )

    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return response.strip()


def interactive_loop(model, tokenizer, *, max_new_tokens: int, temperature: float) -> None:
    """Run an interactive chat session."""
    print("AVA v2 — Interactive Chat")
    print("Type your message and press Enter. Type 'quit' or Ctrl+C to exit.\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        response = generate(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )
        print(f"\nAVA: {response}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with AVA v2")
    parser.add_argument(
        "--adapter", default=HF_REPO,
        help=f"HuggingFace repo ID or local adapter path (default: {HF_REPO})",
    )
    parser.add_argument(
        "--prompt", default=None,
        help="Single prompt (non-interactive mode)",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    model, tokenizer = load_model(args.adapter)

    if args.prompt:
        response = generate(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens, temperature=args.temperature,
        )
        print(response)
    else:
        interactive_loop(
            model, tokenizer,
            max_new_tokens=args.max_tokens, temperature=args.temperature,
        )


if __name__ == "__main__":
    main()
