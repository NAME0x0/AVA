"""Quick test: verify GPU placement works with manual layer movement."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp5_gemma4.engine.loader import load_model


def main():
    model_id = "google/gemma-4-E4B-it"

    print("=== GPU Placement Test ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load with no quantization (bf16 to CPU, then move layers to GPU)
    model, processor, meta = load_model(
        model_id=model_id,
        quantization="none",
        gpu_memory_gb=3.5,
        cpu_memory_gb=28.0,
    )

    print(f"\nLoad metadata: {meta}")
    print(f"GPU allocated after load: {torch.cuda.memory_allocated() / 1e6:.0f} MB")

    # Verify layer placement
    text_model = model.model.language_model
    for i, layer in enumerate(text_model.layers):
        # Check first param device
        device = next(layer.parameters()).device
        if i < 5 or i >= len(text_model.layers) - 2:
            print(f"  Layer {i}: {device}")
        elif i == 5:
            print(f"  ...")

    # Test inference
    print("\n=== Inference Test ===")
    messages = [{"role": "user", "content": "Explain quantum computing in 3 sentences."}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt")
    # Inputs start on CPU — the hooks will handle device transfers
    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    print(f"Input device: {inputs['input_ids'].device}")
    print(f"Input shape: {inputs['input_ids'].shape}")

    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=100, temperature=0.0, do_sample=False,
        )
    elapsed = time.perf_counter() - t0

    response = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Response: {response}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")
    print(f"Tokens generated: {output.shape[1] - inputs['input_ids'].shape[1]}")

    tok_per_s = (output.shape[1] - inputs["input_ids"].shape[1]) / elapsed
    print(f"Speed: {tok_per_s:.1f} tok/s")


if __name__ == "__main__":
    main()
