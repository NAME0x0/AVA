"""Benchmarking utilities for Gemma 4 compression experiments.

Measures:
  - VRAM usage (peak allocated, reserved)
  - Tokens per second (prefill + decode)
  - Perplexity at various context lengths
  - KV cache memory with and without TurboQuant
  - Expert cache hit rates for MoE offloading
"""
from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""

    name: str
    value: float
    unit: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    results: list[BenchmarkResult] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def add(self, name: str, value: float, unit: str, **metadata: Any) -> None:
        self.results.append(BenchmarkResult(name, value, unit, metadata))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "config": self.config,
            "timestamp": self.timestamp,
            "results": [
                {"name": r.name, "value": r.value, "unit": r.unit, **r.metadata}
                for r in self.results
            ],
        }
        path.write_text(json.dumps(data, indent=2))
        print(f"Results saved to {path}")

    def print_summary(self) -> None:
        print("\n=== Benchmark Results ===")
        for r in self.results:
            print(f"  {r.name}: {r.value:.2f} {r.unit}")
        print()


def measure_vram() -> dict[str, float]:
    """Snapshot current VRAM usage."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0, "reserved_mb": 0, "max_allocated_mb": 0}
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1e6,
        "reserved_mb": torch.cuda.memory_reserved() / 1e6,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1e6,
    }


def reset_vram_stats() -> None:
    """Reset peak VRAM tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()


def get_model_input_device(model: Any) -> torch.device:
    """Choose the correct input device for a loaded model.

    Manually split models keep embeddings on CPU and rely on boundary hooks
    to move activations into GPU-resident decoder layers. Those models must
    receive CPU inputs even though `model.device` may report CUDA.
    """
    if hasattr(model, "_manual_gpu_layers"):
        return torch.device("cpu")
    return getattr(model, "device", torch.device("cpu"))


def benchmark_inference_speed(
    model: Any,
    processor: Any,
    prompts: list[str],
    max_new_tokens: int = 128,
    warmup_runs: int = 1,
    assistant_model: Any | None = None,
    generate_kwargs: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Measure tokens/second for generation.

    Returns:
        dict with prefill_tok_s, decode_tok_s, total_tok_s, latency_s
    """
    input_device = get_model_input_device(model)
    extra_generate_kwargs = dict(generate_kwargs or {})
    if assistant_model is not None:
        extra_generate_kwargs["assistant_model"] = assistant_model

    # Warmup
    for _ in range(warmup_runs):
        if assistant_model is not None and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
        inputs = processor(text=prompts[0], return_tensors="pt").to(input_device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=8, **extra_generate_kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_input_tokens = 0
    total_output_tokens = 0
    total_time = 0.0

    for prompt in prompts:
        if assistant_model is not None and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
        inputs = processor(text=prompt, return_tensors="pt").to(input_device)
        input_len = inputs["input_ids"].shape[1]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **extra_generate_kwargs,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        output_len = output.shape[1] - input_len
        total_input_tokens += input_len
        total_output_tokens += output_len
        total_time += elapsed

    total_tokens = total_input_tokens + total_output_tokens
    return {
        "total_tokens": total_tokens,
        "output_tokens": total_output_tokens,
        "total_time_s": total_time,
        "tok_per_s": total_tokens / total_time if total_time > 0 else 0,
        "decode_tok_per_s": total_output_tokens / total_time if total_time > 0 else 0,
        "avg_latency_s": total_time / len(prompts),
    }


def benchmark_perplexity(
    model: Any,
    processor: Any,
    texts: list[str],
    max_length: int | None = None,
    stride: int = 512,
) -> dict[str, float]:
    """Compute perplexity over a set of texts using sliding window.

    Uses a strided approach to handle texts longer than model context.
    """
    import math

    total_nll = 0.0
    total_tokens = 0
    input_device = get_model_input_device(model)

    for text in texts:
        encodings = processor(text, return_tensors="pt")
        input_ids = encodings["input_ids"].to(input_device)
        seq_len = input_ids.shape[1]

        if max_length:
            seq_len = min(seq_len, max_length)
            input_ids = input_ids[:, :seq_len]

        prev_end = 0
        for begin in range(0, seq_len, stride):
            end = min(begin + stride, seq_len)
            target_len = end - prev_end

            chunk_ids = input_ids[:, begin:end]
            target_ids = chunk_ids.clone()
            # Mask out tokens we've already scored
            target_ids[:, :-target_len] = -100

            with torch.no_grad():
                outputs = model(chunk_ids, labels=target_ids)
                nll = outputs.loss.item() * target_len

            total_nll += nll
            total_tokens += target_len
            prev_end = end

            if end >= seq_len:
                break

    avg_nll = total_nll / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(avg_nll)

    return {
        "perplexity": perplexity,
        "avg_nll": avg_nll,
        "total_tokens": total_tokens,
    }


def run_memory_projection(arch_info: dict[str, Any]) -> None:
    """Print KV cache memory projections at various context lengths."""
    from .loader import estimate_kv_cache_memory

    contexts = [8_192, 32_768, 131_072, 262_144, 524_288, 1_048_576]
    print("\n=== KV Cache Memory Projection ===")
    print(f"{'Context':>10}  {'BF16 (MB)':>12}  {'TQ K4/V2 (MB)':>14}  {'Savings':>8}")
    print("-" * 52)

    for ctx in contexts:
        bf16 = estimate_kv_cache_memory(arch_info, ctx)
        tq = estimate_kv_cache_memory(arch_info, ctx, turboquant_key_bits=4, turboquant_value_bits=2)
        savings = 1 - tq["total_compressed_mb"] / bf16["total_bf16_mb"] if bf16["total_bf16_mb"] > 0 else 0
        print(
            f"{ctx:>10,}  {bf16['total_bf16_mb']:>12,.1f}  "
            f"{tq['total_compressed_mb']:>14,.1f}  {savings:>7.0%}"
        )


# Standard benchmark prompts
SPEED_PROMPTS = [
    "Explain the theory of general relativity in simple terms.",
    "Write a Python function to find the longest common subsequence of two strings.",
    "What are the main differences between classical and quantum computing?",
]

QUALITY_PROMPTS = [
    ("What is 17 * 23?", "391"),
    ("What planet is known as the Red Planet?", "Mars"),
    ("In Python, what does len([1,2,3]) return?", "3"),
    ("What is the chemical formula for water?", "H2O"),
    ("What is the square root of 144?", "12"),
]
