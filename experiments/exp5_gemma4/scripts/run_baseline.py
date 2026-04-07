"""Baseline: Load Gemma 4 26B MoE with CPU/GPU offloading and benchmark.

This script establishes the baseline performance metrics before applying
our custom optimizations (TurboQuant, YaRN, MoE-aware offloading).

Usage:
    # Full baseline on 26B with streamed int4 + cached reload:
    python experiments/exp5_gemma4/scripts/run_baseline.py

    # Quick test with a single prompt:
    python experiments/exp5_gemma4/scripts/run_baseline.py --quick

    # Dry run (no model download, just memory projections):
    python experiments/exp5_gemma4/scripts/run_baseline.py --dry-run

    # Use a smaller model for local testing:
    python experiments/exp5_gemma4/scripts/run_baseline.py --model google/gemma-4-E2B-it
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp5_gemma4.engine.benchmark import (
    QUALITY_PROMPTS,
    SPEED_PROMPTS,
    BenchmarkSuite,
    benchmark_inference_speed,
    get_model_input_device,
    measure_vram,
    reset_vram_stats,
    run_memory_projection,
)
from experiments.exp5_gemma4.engine.loader import (
    estimate_kv_cache_memory,
    inspect_model_architecture,
    load_model,
)
from experiments.exp5_gemma4.engine.turboquant import TurboQuantV3
from experiments.exp5_gemma4.engine.yarn import YarnContextExtender


def dry_run(model_id: str) -> None:
    """Print memory projections without loading the model."""
    print(f"=== Dry Run: {model_id} ===\n")

    # Use known architecture info for the 26B MoE
    if "26B" in model_id:
        arch_info = {
            "num_layers": 30,
            "hidden_size": 2816,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "num_global_key_value_heads": 2,
            "head_dim": 256,
            "global_head_dim": 512,
            "attention_k_eq_v": True,
            "num_experts": 128,
            "top_k_experts": 8,
            "moe_intermediate_size": 704,
            "sliding_window": 1024,
            "max_position_embeddings": 262_144,
            "global_layer_indices": [5, 11, 17, 23, 29],
            "sliding_layer_indices": [i for i in range(30) if i not in [5, 11, 17, 23, 29]],
            "vocab_size": 262_144,
        }
    else:
        print("Dry run only supports 26B MoE architecture info. Loading config...")
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        text_config = getattr(config, "text_config", config)
        layer_types = getattr(text_config, "layer_types", [])
        arch_info = {
            "num_layers": text_config.num_hidden_layers,
            "hidden_size": text_config.hidden_size,
            "num_attention_heads": text_config.num_attention_heads,
            "num_key_value_heads": text_config.num_key_value_heads,
            "num_global_key_value_heads": getattr(text_config, "num_global_key_value_heads", text_config.num_key_value_heads),
            "head_dim": text_config.head_dim,
            "global_head_dim": getattr(text_config, "global_head_dim", text_config.head_dim),
            "attention_k_eq_v": getattr(text_config, "attention_k_eq_v", False),
            "num_experts": getattr(text_config, "num_experts", 0),
            "top_k_experts": getattr(text_config, "top_k_experts", 0),
            "moe_intermediate_size": getattr(text_config, "moe_intermediate_size", 0),
            "sliding_window": text_config.sliding_window,
            "max_position_embeddings": text_config.max_position_embeddings,
            "global_layer_indices": [i for i, lt in enumerate(layer_types) if lt == "full_attention"],
            "sliding_layer_indices": [i for i, lt in enumerate(layer_types) if lt == "sliding_attention"],
            "vocab_size": text_config.vocab_size,
        }

    print("Architecture:")
    print(json.dumps(arch_info, indent=2))

    # KV cache projections
    run_memory_projection(arch_info)

    # TurboQuant compression estimates
    tq = TurboQuantV3(key_bits=4, value_bits=2)
    print("\nTurboQuant V3 Compression Estimates:")
    ratios = tq.estimate_compression_ratio(
        arch_info.get("global_head_dim") or arch_info["head_dim"]
    )
    for k, v in ratios.items():
        print(f"  {k}: {v:.2f}")

    # YaRN summary
    yarn = YarnContextExtender(target_context=1_048_576)
    print("\nYaRN Context Extension:")
    for k, v in yarn.summary().items():
        print(f"  {k}: {v}")

    # Expert memory estimates
    if arch_info["num_experts"] > 0:
        expert_params = 3 * arch_info["hidden_size"] * arch_info["moe_intermediate_size"]
        expert_bytes_bf16 = expert_params * 2
        expert_bytes_q4 = expert_params // 2  # 4-bit = 0.5 bytes/param
        total_expert_memory_q4 = expert_bytes_q4 * arch_info["num_experts"] * arch_info["num_layers"]

        print(f"\nMoE Expert Memory:")
        print(f"  Params per expert: {expert_params:,}")
        print(f"  Size per expert (BF16): {expert_bytes_bf16 / 1e6:.1f} MB")
        print(f"  Size per expert (Q4): {expert_bytes_q4 / 1e6:.1f} MB")
        print(f"  Total experts: {arch_info['num_experts']} per layer x {arch_info['num_layers']} layers")
        print(f"  Total expert memory (Q4): {total_expert_memory_q4 / 1e9:.1f} GB")
        print(f"  Active per token: {arch_info['top_k_experts']} experts")
        print(f"  Active memory (Q4): {expert_bytes_q4 * arch_info['top_k_experts'] / 1e6:.1f} MB per layer")


def run_full_baseline(
    model_id: str, quick: bool = False, quantization: str = "streaming-int4",
) -> None:
    """Load model and run full benchmark suite."""
    suite = BenchmarkSuite(
        config={"model_id": model_id, "quick": quick, "quantization": quantization},
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # 1. Load model
    print("\n=== Loading Model ===")
    reset_vram_stats()
    model, processor, load_meta = load_model(
        model_id=model_id,
        quantization=quantization,
        gpu_memory_gb=3.5,
        cpu_memory_gb=28.0,
    )
    suite.add("load_time", load_meta["load_time_s"], "seconds")
    suite.add("gpu_memory_after_load", load_meta.get("gpu_allocated_mb", 0), "MB")
    suite.config["load_metadata"] = load_meta

    # 2. Architecture inspection
    print("\n=== Architecture Inspection ===")
    arch_info = inspect_model_architecture(model)
    print(json.dumps(arch_info, indent=2))
    suite.config["architecture"] = arch_info

    # 3. KV cache projections
    run_memory_projection(arch_info)

    # 4. Inference speed
    print("\n=== Inference Speed ===")
    prompts = SPEED_PROMPTS[:1] if quick else SPEED_PROMPTS
    max_tokens = 32 if quick else 128
    speed = benchmark_inference_speed(
        model, processor, prompts,
        max_new_tokens=max_tokens,
        warmup_runs=0 if quick else 1,
    )
    suite.add("decode_tok_per_s", speed["decode_tok_per_s"], "tok/s")
    suite.add("total_tok_per_s", speed["tok_per_s"], "tok/s")
    suite.add("avg_latency", speed["avg_latency_s"], "seconds")
    print(f"  Decode: {speed['decode_tok_per_s']:.1f} tok/s")
    print(f"  Total:  {speed['tok_per_s']:.1f} tok/s")
    print(f"  Latency: {speed['avg_latency_s']:.2f}s per prompt")

    # 5. Quality spot-check
    if not quick:
        print("\n=== Quality Spot-Check ===")
        correct = 0
        input_device = get_model_input_device(model)
        for prompt, expected in QUALITY_PROMPTS:
            inputs = processor(text=prompt, return_tensors="pt").to(input_device)
            import torch
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=32, temperature=0.0, do_sample=False)
            response = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            is_correct = expected.lower() in response.lower()
            correct += int(is_correct)
            status = "PASS" if is_correct else "FAIL"
            print(f"  [{status}] Q: {prompt[:50]}  A: {response.strip()[:50]}  Expected: {expected}")
        suite.add("quality_accuracy", correct / len(QUALITY_PROMPTS) * 100, "%")

    # 6. Peak VRAM
    vram = measure_vram()
    suite.add("peak_vram", vram.get("max_allocated_mb", 0), "MB")
    print(f"\n  Peak VRAM: {vram.get('max_allocated_mb', 0):.0f} MB")

    # Save results
    suite.print_summary()
    results_dir = PROJECT_ROOT / "experiments" / "exp5_gemma4" / "results"
    model_tag = model_id.split("/")[-1].lower()
    suite.save(results_dir / f"baseline_{model_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma 4 baseline benchmark")
    parser.add_argument(
        "--model", default="google/gemma-4-26B-A4B-it",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--quantization", default="streaming-int4",
        choices=["streaming-int4", "quanto-int4", "bnb-nf4", "gptq", "awq", "prequantized", "none"],
        help="Quantization method (default: streaming-int4 for 26B-local feasibility)",
    )
    parser.add_argument("--quick", action="store_true", help="Quick test (1 prompt, fewer tokens)")
    parser.add_argument("--dry-run", action="store_true", help="Memory projections only, no model download")
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args.model)
    else:
        run_full_baseline(args.model, quick=args.quick, quantization=args.quantization)


if __name__ == "__main__":
    main()
