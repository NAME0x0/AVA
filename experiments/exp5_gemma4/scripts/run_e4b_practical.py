"""Practical local Gemma 4 E4B runner for the 4 GB VRAM / 32 GB RAM target.

This script treats E4B as the main interactive branch rather than a draft
model. It prefers:
  - bf16 weights with manual CPU/GPU layer placement
  - optional TurboQuant / YaRN patches
  - chat-template generation with thinking disabled by default for latency
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp5_gemma4.engine.benchmark import BenchmarkSuite, measure_vram, reset_vram_stats
from experiments.exp5_gemma4.engine.loader import inspect_model_architecture, load_model
from experiments.exp5_gemma4.engine.patches import get_model_memory_footprint
from experiments.exp5_gemma4.engine.practical import (
    PRACTICAL_PROMPTS,
    apply_practical_optimizations,
    benchmark_chat_speed,
    decode_chat_response,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Practical Gemma 4 E4B runner")
    parser.add_argument("--model", default="google/gemma-4-E4B-it", help="Hugging Face model ID")
    parser.add_argument(
        "--quantization",
        default="none",
        choices=["none", "quanto-int4", "bnb-nf4", "gptq", "awq", "prequantized"],
        help="E4B practical branch defaults to bf16 + manual GPU placement",
    )
    parser.add_argument("--gpu-memory-gb", type=float, default=3.5)
    parser.add_argument("--cpu-memory-gb", type=float, default=28.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--cache-implementation", default="static", help="KV cache backend for generation")
    parser.add_argument("--quick", action="store_true", help="Use one short prompt")
    parser.add_argument("--enable-thinking", action="store_true", help="Keep Gemma thinking enabled")
    parser.add_argument("--no-turboquant", action="store_true", help="Disable TurboQuant")
    parser.add_argument("--no-yarn", action="store_true", help="Disable YaRN")
    parser.add_argument("--tq-key-bits", type=int, default=4)
    parser.add_argument("--tq-value-bits", type=int, default=2)
    parser.add_argument("--yarn-context", type=int, default=524_288, help="Practical context target")
    args = parser.parse_args()

    suite = BenchmarkSuite(
        config={
            "model_id": args.model,
            "quantization": args.quantization,
            "quick": args.quick,
            "enable_thinking": args.enable_thinking,
            "cache_implementation": args.cache_implementation,
            "optimizations": {
                "turboquant": not args.no_turboquant,
                "yarn": not args.no_yarn,
            },
        },
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    print("\n=== Loading E4B Practical Model ===")
    reset_vram_stats()
    model, processor, load_meta = load_model(
        model_id=args.model,
        quantization=args.quantization,
        gpu_memory_gb=args.gpu_memory_gb,
        cpu_memory_gb=args.cpu_memory_gb,
    )
    suite.add("load_time", load_meta["load_time_s"], "seconds")
    suite.add("gpu_memory_after_load", load_meta.get("gpu_allocated_mb", 0.0), "MB")
    suite.config["load_metadata"] = load_meta

    print("\n=== Architecture Inspection ===")
    arch_info = inspect_model_architecture(model)
    print(json.dumps(arch_info, indent=2))
    suite.config["architecture"] = arch_info

    print("\n=== Applying Practical Optimizations ===")
    opt_meta = apply_practical_optimizations(
        model,
        arch_info,
        enable_turboquant=not args.no_turboquant,
        enable_yarn=not args.no_yarn,
        tq_key_bits=args.tq_key_bits,
        tq_value_bits=args.tq_value_bits,
        tq_protected_layers=1,
        yarn_target_context=args.yarn_context,
    )
    suite.config["optimizations_meta"] = opt_meta

    print("\n=== Memory Footprint ===")
    footprint = get_model_memory_footprint(model)
    print(f"  Total params: {footprint['total_params'] / 1e9:.2f}B")
    print(f"  GPU: {footprint['gpu_gb']:.2f} GB")
    print(f"  CPU: {footprint['cpu_gb']:.2f} GB")
    suite.config["memory_footprint"] = footprint

    prompts = PRACTICAL_PROMPTS[:1] if args.quick else PRACTICAL_PROMPTS
    print("\n=== Practical Chat Speed ===")
    speed = benchmark_chat_speed(
        model,
        processor,
        prompts,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=args.enable_thinking,
        cache_implementation=args.cache_implementation,
    )
    suite.add("decode_tok_per_s", speed["decode_tok_per_s"], "tok/s")
    suite.add("total_tok_per_s", speed["tok_per_s"], "tok/s")
    suite.add("avg_latency", speed["avg_latency_s"], "seconds")
    print(f"  Decode:  {speed['decode_tok_per_s']:.2f} tok/s")
    print(f"  Total:   {speed['tok_per_s']:.2f} tok/s")
    print(f"  Latency: {speed['avg_latency_s']:.2f}s per prompt")

    print("\n=== Sanity Prompt ===")
    sanity_prompt = "What is the capital of France?"
    response = decode_chat_response(
        model,
        processor,
        sanity_prompt,
        max_new_tokens=32,
        enable_thinking=args.enable_thinking,
        cache_implementation=args.cache_implementation,
    )
    print(f"  Response: {response.encode('ascii', 'backslashreplace').decode('ascii')}")

    vram = measure_vram()
    suite.add("peak_vram", vram.get("max_allocated_mb", 0.0), "MB")
    print(f"  Peak VRAM: {vram.get('max_allocated_mb', 0.0):.0f} MB")

    suite.print_summary()
    results_dir = PROJECT_ROOT / "experiments" / "exp5_gemma4" / "results"
    model_tag = args.model.split("/")[-1].lower()
    suffix = "thinking" if args.enable_thinking else "practical"
    suite.save(results_dir / f"e4b_{suffix}_{model_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")


if __name__ == "__main__":
    main()
