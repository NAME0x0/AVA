"""Smoke-test exact assisted decoding on the cached 26B runtime.

Loads the cached 26B verifier plus a smaller Gemma 4 draft model, then
compares plain greedy decoding against exact assisted decoding on the same
prompt. This keeps the current streamed verifier path intact while giving us a
real local measurement for the draft/verifier approach.
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


def timed_generate(model, *, input_ids, attention_mask, max_new_tokens: int, **kwargs):
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            **kwargs,
        )
    return output_ids, time.perf_counter() - t0


def main() -> None:
    from experiments.exp5_gemma4.engine.speculative import (
        AssistedDecodingConfig,
        load_assistant_model,
    )
    from experiments.exp5_gemma4.engine.streaming import load_quantized

    print("=" * 60)
    print("Exact Assisted Decoding Smoke Test")
    print("=" * 60)

    if not (SAVE_DIR / "quantized_config.json").exists():
        raise FileNotFoundError(f"Missing quantized cache at {SAVE_DIR}")

    assistant_cfg = AssistedDecodingConfig(
        assistant_model_id="google/gemma-4-E4B-it",
        assistant_quantization="quanto-int4",
        assistant_gpu_memory_gb=0.0,
        assistant_cpu_memory_gb=10.0,
        num_assistant_tokens=8,
        num_assistant_tokens_schedule="heuristic_transient",
        assistant_confidence_threshold=0.4,
    )
    assistant_model, _, assistant_meta, assisted_generate_kwargs = load_assistant_model(assistant_cfg)
    print(
        "Assistant loaded: "
        f"{assistant_meta['model_id']} / {assistant_meta['quantization']} / "
        f"input_device={assistant_meta['assistant_input_device']}"
    )
    check_ram("after assistant load")

    verifier, processor, verifier_meta = load_quantized(
        SAVE_DIR,
        dtype="bfloat16",
        gpu_memory_gb=3.5,
    )
    print(f"Verifier reload complete: {verifier_meta.get('model_id', 'cached-26b')}")
    check_ram("after verifier reload")

    prompt = "What is the capital of France?"
    messages = [{"role": "user", "content": prompt}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    print("\n-- Exact Assisted Decode --")
    assisted_ids, assisted_time = timed_generate(
        verifier,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=32,
        **assisted_generate_kwargs,
    )
    check_ram("after assisted decode")

    assisted_new_tokens = assisted_ids[0, input_ids.shape[1]:]
    assisted_new_tokens = assisted_new_tokens.cpu().clone()
    assisted_response = processor.decode(assisted_new_tokens, skip_special_tokens=True)
    del assisted_ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n-- Plain Greedy Decode --")
    plain_ids, plain_time = timed_generate(
        verifier,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=32,
    )
    check_ram("after plain decode")

    plain_new_tokens = plain_ids[0, input_ids.shape[1]:]
    plain_new_tokens = plain_new_tokens.cpu().clone()
    plain_response = processor.decode(plain_new_tokens, skip_special_tokens=True)
    del plain_ids
    outputs_match = torch.equal(plain_new_tokens, assisted_new_tokens)
    speedup = plain_time / assisted_time if assisted_time > 0 else 0.0

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Plain decode:    {plain_time:.1f}s, {len(plain_new_tokens) / plain_time:.2f} tok/s")
    print(f"  Assisted decode: {assisted_time:.1f}s, {len(assisted_new_tokens) / assisted_time:.2f} tok/s")
    print(f"  Speedup:         {speedup:.2f}x")
    print(f"  Outputs match:   {outputs_match}")
    print(f"  Plain response:  {plain_response.encode('ascii', 'backslashreplace').decode('ascii')}")
    print(f"  Assisted reply:  {assisted_response.encode('ascii', 'backslashreplace').decode('ascii')}")

    del assistant_model, verifier, processor
    gc.collect()
    check_ram("after cleanup")


if __name__ == "__main__":
    main()
