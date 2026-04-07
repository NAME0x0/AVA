"""End-to-end 26B MoE inference via streaming int4 quantization.

Safety: monitors RSS throughout and aborts if approaching 30 GB.
Expected memory profile:
  - Loading peak: ~17.7 GB (1 bf16 layer buffer + quantized layers)
  - Resident after load: ~15.2 GB (all int4 + embeddings in bf16)
  - Inference overhead: ~1-2 GB (activations + KV cache for short prompt)
  - Total expected: ~17 GB << 32 GB available
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import psutil
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Safety threshold: abort if RSS exceeds this (GB)
RAM_ABORT_THRESHOLD_GB = 29.0


def get_rss_gb() -> float:
    return psutil.Process().memory_info().rss / 1e9


def check_ram(stage: str) -> None:
    rss = get_rss_gb()
    print(f"  [RAM] {stage}: {rss:.1f} GB")
    if rss > RAM_ABORT_THRESHOLD_GB:
        print(f"  ABORT: RSS {rss:.1f} GB exceeds {RAM_ABORT_THRESHOLD_GB} GB threshold")
        sys.exit(1)


def main():
    print("=" * 60)
    print("26B-A4B End-to-End Streaming Int4 Inference Test")
    print("=" * 60)

    check_ram("before import")

    from experiments.exp5_gemma4.engine.streaming import streaming_load

    check_ram("after import")

    # --Step 1: Streaming load with int4 quantization --
    print("\n--Loading model (streaming int4) --")
    t0 = time.perf_counter()

    model, processor, metadata = streaming_load(
        "google/gemma-4-26B-A4B-it",
        gpu_memory_gb=3.5,
        cpu_memory_gb=28.0,
        group_size=128,
        dtype="bfloat16",
    )

    load_time = time.perf_counter() - t0
    check_ram("after load")

    print(f"\nLoad summary:")
    for k, v in metadata.items():
        print(f"  {k}: {v}")

    # --Step 2: Verify model structure --
    print("\n--Verifying model structure --")

    # Count QuantizedLinear and QuantizedMoEExperts modules
    from experiments.exp5_gemma4.engine.streaming import QuantizedLinear, QuantizedMoEExperts

    n_qlinear = sum(1 for m in model.modules() if isinstance(m, QuantizedLinear))
    n_qexperts = sum(1 for m in model.modules() if isinstance(m, QuantizedMoEExperts))
    print(f"  QuantizedLinear modules: {n_qlinear}")
    print(f"  QuantizedMoEExperts modules: {n_qexperts}")

    # Check no meta tensors remain
    meta_params = [
        name for name, p in model.named_parameters()
        if p.device.type == "meta"
    ]
    meta_buffers = [
        name for name, b in model.named_buffers()
        if b.device.type == "meta"
    ]
    if meta_params:
        print(f"  WARNING: {len(meta_params)} parameters still on meta: {meta_params[:5]}")
    if meta_buffers:
        print(f"  WARNING: {len(meta_buffers)} buffers still on meta: {meta_buffers[:5]}")
    if not meta_params and not meta_buffers:
        print("  All parameters and buffers materialized (no meta tensors)")

    check_ram("after verification")

    # --Step 3: Run inference --
    print("\n--Running inference --")

    prompt = "What is the capital of France?"
    print(f"  Prompt: {prompt}")

    # Tokenize
    messages = [{"role": "user", "content": prompt}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)
    print(f"  Input tokens: {input_ids.shape[1]}")

    check_ram("before generate")

    # Generate with conservative settings
    t1 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=32,
            do_sample=False,
            temperature=1.0,
        )
    gen_time = time.perf_counter() - t1

    check_ram("after generate")

    # Decode
    new_tokens = output_ids[0, input_ids.shape[1]:]
    response = processor.decode(new_tokens, skip_special_tokens=True)
    n_tokens = len(new_tokens)
    tok_per_sec = n_tokens / gen_time if gen_time > 0 else 0

    print(f"\n  Response: {response}")
    print(f"  Tokens generated: {n_tokens}")
    print(f"  Generation time: {gen_time:.1f}s")
    print(f"  Speed: {tok_per_sec:.2f} tok/s")

    # --Step 4: Validate output --
    print("\n--Validation --")
    response_lower = response.lower()
    if "paris" in response_lower:
        print("  PASS: Response contains 'Paris'")
    else:
        print(f"  WARN: Response does not contain 'Paris': {response[:100]}")

    # --Summary --
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Model: google/gemma-4-26B-A4B-it")
    print(f"  bf16 size: {metadata.get('bf16_gb', '?')} GB")
    print(f"  Quantized size: {metadata.get('q4_gb', '?')} GB")
    print(f"  Compression: {metadata.get('compression_ratio', '?')}x")
    print(f"  Load time: {load_time:.1f}s")
    print(f"  Peak RAM: {get_rss_gb():.1f} GB")
    print(f"  QuantizedLinear: {n_qlinear}")
    print(f"  QuantizedMoEExperts: {n_qexperts}")
    print(f"  Inference speed: {tok_per_sec:.2f} tok/s")
    print(f"  Response: {response[:80]}")
    print(f"  Correct: {'paris' in response_lower}")

    # Cleanup
    del model, processor
    gc.collect()
    check_ram("after cleanup")


if __name__ == "__main__":
    main()
