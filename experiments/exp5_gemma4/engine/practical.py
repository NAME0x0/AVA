"""Shared helpers for practical local Gemma 4 runtime branches."""
from __future__ import annotations

import gc
import time
from typing import Any

import torch

from .patches import patch_model_for_turboquant, patch_model_for_yarn
from .turboquant import TurboQuantV3
from .yarn import YarnContextExtender


PRACTICAL_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in three short sentences.",
    "Write a Python function that returns the Fibonacci sequence up to n.",
]


def apply_practical_optimizations(
    model: Any,
    arch_info: dict[str, Any],
    *,
    enable_turboquant: bool,
    enable_yarn: bool,
    tq_key_bits: int,
    tq_value_bits: int,
    tq_protected_layers: int,
    yarn_target_context: int,
) -> dict[str, Any]:
    """Apply the exact optimizations useful for local Gemma 4 branches."""
    meta: dict[str, Any] = {}

    if enable_turboquant:
        global_layers = arch_info.get("global_layer_indices", [])
        tq = TurboQuantV3(
            key_bits=tq_key_bits,
            value_bits=tq_value_bits,
            protected_layers=tq_protected_layers,
            total_global_layers=len(global_layers),
        )
        tq.global_layer_indices = global_layers
        patch_model_for_turboquant(model, tq)
        meta["turboquant"] = {
            "key_bits": tq_key_bits,
            "value_bits": tq_value_bits,
            "protected_layers": tq_protected_layers,
            "global_layers_patched": len(global_layers),
            **tq.estimate_compression_ratio(
                arch_info.get("global_head_dim") or arch_info["head_dim"]
            ),
        }

    if enable_yarn:
        yarn = YarnContextExtender(
            target_context=yarn_target_context,
            original_context=arch_info.get("max_position_embeddings", 131_072),
            head_dim=arch_info.get("global_head_dim") or arch_info["head_dim"],
        )
        patch_model_for_yarn(model, yarn)
        meta["yarn"] = yarn.summary()

    return meta


def build_chat_inputs(
    processor: Any,
    prompt: str,
    *,
    enable_thinking: bool,
) -> tuple[dict[str, torch.Tensor], int]:
    """Build chat-template inputs for Gemma 4."""
    messages = [{"role": "user", "content": prompt}]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    inputs = processor(text=text, return_tensors="pt")
    return inputs, inputs["input_ids"].shape[1]


def benchmark_chat_speed(
    model: Any,
    processor: Any,
    prompts: list[str],
    *,
    max_new_tokens: int,
    enable_thinking: bool,
    cache_implementation: str | None = "static",
) -> dict[str, float]:
    """Measure practical chat generation throughput."""
    total_input_tokens = 0
    total_output_tokens = 0
    total_time = 0.0

    for prompt in prompts:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        inputs, input_len = build_chat_inputs(
            processor,
            prompt,
            enable_thinking=enable_thinking,
        )
        t0 = time.perf_counter()
        generate_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "use_cache": True,
        }
        if cache_implementation is not None:
            generate_kwargs["cache_implementation"] = cache_implementation
        with torch.inference_mode():
            output = model.generate(**generate_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        total_input_tokens += input_len
        total_output_tokens += output.shape[1] - input_len
        total_time += elapsed

    total_tokens = total_input_tokens + total_output_tokens
    return {
        "total_tokens": total_tokens,
        "output_tokens": total_output_tokens,
        "total_time_s": total_time,
        "tok_per_s": total_tokens / total_time if total_time > 0 else 0.0,
        "decode_tok_per_s": total_output_tokens / total_time if total_time > 0 else 0.0,
        "avg_latency_s": total_time / len(prompts) if prompts else 0.0,
    }


def decode_chat_response(
    model: Any,
    processor: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    enable_thinking: bool,
    cache_implementation: str | None = "static",
) -> str:
    """Run a single deterministic generation and decode the reply text."""
    result = generate_chat_response(
        model,
        processor,
        prompt,
        max_new_tokens=max_new_tokens,
        enable_thinking=enable_thinking,
        cache_implementation=cache_implementation,
    )
    return result["response"]


def generate_chat_response(
    model: Any,
    processor: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    enable_thinking: bool,
    cache_implementation: str | None = "static",
) -> dict[str, Any]:
    """Run a single deterministic generation and return text plus timing."""
    inputs, input_len = build_chat_inputs(
        processor,
        prompt,
        enable_thinking=enable_thinking,
    )
    t0 = time.perf_counter()
    generate_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs.get("attention_mask"),
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
    }
    if cache_implementation is not None:
        generate_kwargs["cache_implementation"] = cache_implementation
    with torch.inference_mode():
        output = model.generate(**generate_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    response = processor.decode(output[0][input_len:], skip_special_tokens=True)
    return {
        "response": response,
        "input_tokens": input_len,
        "output_tokens": output.shape[1] - input_len,
        "elapsed_s": elapsed,
    }
