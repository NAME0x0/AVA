"""Test save + reload cycle for 26B quantized model.

Phase 1: streaming_load + save_quantized (slow, ~271s + save time)
Phase 2: load_quantized (fast reload, should be << 60s)
Phase 3: inference on reloaded model (verify correctness)
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

SAVE_DIR = Path("D:/AVA/quantized_cache/google--gemma-4-26B-A4B-it")
RAM_ABORT_GB = 29.0


def get_rss_gb() -> float:
    return psutil.Process().memory_info().rss / 1e9


def check_ram(stage: str) -> None:
    rss = get_rss_gb()
    print(f"  [RAM] {stage}: {rss:.1f} GB")
    if rss > RAM_ABORT_GB:
        print(f"  ABORT: RSS {rss:.1f} GB exceeds threshold")
        sys.exit(1)


def phase1_save():
    """Load 26B via streaming, save quantized weights to disk."""
    from experiments.exp5_gemma4.engine.streaming import streaming_load, save_quantized

    print("=" * 60)
    print("Phase 1: Streaming load + save")
    print("=" * 60)

    check_ram("before load")

    model, processor, metadata = streaming_load(
        "google/gemma-4-26B-A4B-it",
        gpu_memory_gb=3.5,
        cpu_memory_gb=28.0,
        group_size=128,
        dtype="bfloat16",
    )

    check_ram("after load")

    save_quantized(model, SAVE_DIR, metadata, processor=processor)

    check_ram("after save")

    # Cleanup to free RAM for phase 2
    del model, processor
    gc.collect()
    check_ram("after cleanup")


def phase2_reload_and_test():
    """Reload from disk and run inference."""
    from experiments.exp5_gemma4.engine.streaming import load_quantized

    print("\n" + "=" * 60)
    print("Phase 2: Fast reload from disk")
    print("=" * 60)

    check_ram("before reload")

    t0 = time.perf_counter()
    model, processor, metadata = load_quantized(
        SAVE_DIR,
        dtype="bfloat16",
        gpu_memory_gb=3.5,
    )
    reload_time = time.perf_counter() - t0

    check_ram("after reload")

    print(f"\n  Reload time: {reload_time:.1f}s")

    # Quick structure check
    from experiments.exp5_gemma4.engine.streaming import QuantizedLinear, QuantizedMoEExperts
    n_ql = sum(1 for m in model.modules() if isinstance(m, QuantizedLinear))
    n_qe = sum(1 for m in model.modules() if isinstance(m, QuantizedMoEExperts))
    print(f"  QuantizedLinear: {n_ql}")
    print(f"  QuantizedMoEExperts: {n_qe}")

    meta_params = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    meta_bufs = [n for n, b in model.named_buffers() if b.device.type == "meta"]
    if meta_params:
        print(f"  WARNING: {len(meta_params)} meta params: {meta_params[:3]}")
    if meta_bufs:
        print(f"  WARNING: {len(meta_bufs)} meta buffers: {meta_bufs[:3]}")
    if not meta_params and not meta_bufs:
        print("  All tensors materialized")

    # Inference
    print("\n-- Inference --")
    prompt = "What is the capital of France?"
    messages = [{"role": "user", "content": prompt}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt")

    check_ram("before generate")

    t1 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=32,
            do_sample=False,
        )
    gen_time = time.perf_counter() - t1

    check_ram("after generate")

    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    response = processor.decode(new_tokens, skip_special_tokens=True)

    safe_response = response.encode("ascii", "backslashreplace").decode("ascii")
    print(f"  Response: {safe_response}")
    print(f"  Generation: {gen_time:.1f}s, {len(new_tokens) / gen_time:.2f} tok/s")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Reload time: {reload_time:.1f}s (vs 271s streaming load)")
    print(f"  Speedup: {271.0 / reload_time:.1f}x")
    print(f"  QuantizedLinear: {n_ql}, QuantizedMoEExperts: {n_qe}")
    print(f"  Response: {safe_response[:80]}")
    print(f"  Correct: {'paris' in response.lower()}")

    del model, processor
    gc.collect()


def main():
    # If cache already exists, skip phase 1
    if (SAVE_DIR / "quantized_config.json").exists():
        print(f"Cache exists at {SAVE_DIR}, skipping phase 1")
    else:
        phase1_save()

    phase2_reload_and_test()


if __name__ == "__main__":
    main()
