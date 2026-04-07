"""Optimized pipeline: Gemma 4 + TurboQuant + YaRN + MoE offloading.

Loads Gemma 4 26B MoE (or E4B for dev), applies all optimizations, and
benchmarks against the baseline to measure improvements.

Usage:
    # Quick test with E4B (dev model):
    python experiments/exp5_gemma4/scripts/run_optimized.py --model google/gemma-4-E4B-it --quick

    # Full optimized run with 26B MoE (streamed int4 + local cache):
    python experiments/exp5_gemma4/scripts/run_optimized.py --quantization streaming-int4

    # Compare against baseline:
    python experiments/exp5_gemma4/scripts/run_optimized.py --compare results/baseline_*.json
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

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
from experiments.exp5_gemma4.engine.patches import (
    get_model_memory_footprint,
    patch_model_for_turboquant,
    patch_model_for_yarn,
)
from experiments.exp5_gemma4.engine.turboquant import TurboQuantV3
from experiments.exp5_gemma4.engine.yarn import YarnContextExtender
from experiments.exp5_gemma4.engine.speculative import (
    AssistedDecodingConfig,
    load_assistant_model,
)


def apply_optimizations(
    model,
    arch_info: dict,
    enable_turboquant: bool = True,
    enable_yarn: bool = True,
    tq_key_bits: int = 4,
    tq_value_bits: int = 2,
    tq_protected_layers: int = 1,
    yarn_target_context: int = 1_048_576,
) -> dict:
    """Apply TurboQuant and YaRN patches to a loaded model.

    Returns metadata about what was applied.
    """
    meta = {}

    if enable_turboquant:
        print("\n--- Applying TurboQuant V3 ---")
        global_layers = arch_info.get("global_layer_indices", [5, 11, 17, 23, 29])
        tq = TurboQuantV3(
            key_bits=tq_key_bits,
            value_bits=tq_value_bits,
            protected_layers=tq_protected_layers,
            total_global_layers=len(global_layers),
        )
        tq.global_layer_indices = global_layers
        handles = patch_model_for_turboquant(model, tq)

        head_dim = arch_info.get("global_head_dim") or arch_info["head_dim"]
        ratios = tq.estimate_compression_ratio(head_dim)
        meta["turboquant"] = {
            "key_bits": tq_key_bits,
            "value_bits": tq_value_bits,
            "protected_layers": tq_protected_layers,
            "global_layers_patched": len(global_layers),
            **ratios,
        }

        # KV cache with TurboQuant compression
        kv_1m = estimate_kv_cache_memory(
            arch_info, seq_len=1_048_576,
            turboquant_key_bits=tq_key_bits,
            turboquant_value_bits=tq_value_bits,
        )
        meta["kv_cache_1m_compressed_mb"] = kv_1m["total_compressed_mb"]
        meta["kv_cache_1m_bf16_mb"] = kv_1m["total_bf16_mb"]
        print(f"  KV cache at 1M tokens: {kv_1m['total_bf16_mb']:.0f} MB (bf16)"
              f" ->{kv_1m['total_compressed_mb']:.0f} MB (compressed)")

    if enable_yarn:
        print("\n--- Applying YaRN Context Extension ---")
        yarn = YarnContextExtender(
            target_context=yarn_target_context,
            original_context=arch_info.get("max_position_embeddings", 262_144),
            head_dim=arch_info.get("global_head_dim") or arch_info["head_dim"],
        )
        patch_model_for_yarn(model, yarn)
        meta["yarn"] = yarn.summary()

    return meta


def run_optimized_benchmark(
    model_id: str,
    quantization: str = "streaming-int4",
    quick: bool = False,
    enable_turboquant: bool = True,
    enable_yarn: bool = True,
    tq_key_bits: int = 4,
    tq_value_bits: int = 2,
    yarn_target_context: int = 1_048_576,
    compare_file: str | None = None,
    assisted_decoding: AssistedDecodingConfig | None = None,
) -> None:
    """Load model, apply optimizations, and benchmark."""
    suite = BenchmarkSuite(
        config={
            "model_id": model_id,
            "quantization": quantization,
            "quick": quick,
            "optimizations": {
                "turboquant": enable_turboquant,
                "yarn": enable_yarn,
            },
        },
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    assistant_model = None
    assisted_generate_kwargs: dict[str, object] = {}
    if assisted_decoding is not None:
        print("\n=== Loading Assistant Model ===")
        assistant_model, _, assistant_meta, assisted_generate_kwargs = load_assistant_model(assisted_decoding)
        suite.config["assistant_metadata"] = assistant_meta

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

    # 3. Apply optimizations
    print("\n=== Applying Optimizations ===")
    opt_meta = apply_optimizations(
        model, arch_info,
        enable_turboquant=enable_turboquant,
        enable_yarn=enable_yarn,
        tq_key_bits=tq_key_bits,
        tq_value_bits=tq_value_bits,
        yarn_target_context=yarn_target_context,
    )
    suite.config["optimizations_meta"] = opt_meta

    vram_after_patches = measure_vram()
    suite.add("gpu_memory_after_patches", vram_after_patches.get("allocated_mb", 0), "MB")
    print(f"\n  GPU memory after patches: {vram_after_patches.get('allocated_mb', 0):.0f} MB")

    # 4. Memory footprint comparison
    print("\n=== Memory Footprint ===")
    footprint = get_model_memory_footprint(model)
    print(f"  Total params: {footprint['total_params'] / 1e9:.2f}B")
    print(f"  GPU: {footprint['gpu_gb']:.2f} GB")
    print(f"  CPU: {footprint['cpu_gb']:.2f} GB")
    suite.config["memory_footprint"] = footprint

    # 5. KV cache projections (with and without TurboQuant)
    print("\n=== KV Cache Projections ===")
    run_memory_projection(arch_info)

    if enable_turboquant:
        print("\n  With TurboQuant compression:")
        for ctx_len in [8192, 32768, 131072, 524288, 1048576]:
            kv = estimate_kv_cache_memory(
                arch_info, seq_len=ctx_len,
                turboquant_key_bits=tq_key_bits,
                turboquant_value_bits=tq_value_bits,
            )
            print(f"    {ctx_len:>10,} tokens: {kv['total_compressed_mb']:>8.1f} MB "
                  f"(vs {kv['total_bf16_mb']:>8.1f} MB bf16, "
                  f"{kv['compression_ratio']:.1f}x compression)")

    # 6. Inference speed
    print("\n=== Inference Speed ===")
    prompts = SPEED_PROMPTS[:1] if quick else SPEED_PROMPTS
    max_tokens = 32 if quick else 128
    speed = benchmark_inference_speed(
        model, processor, prompts,
        max_new_tokens=max_tokens,
        warmup_runs=0 if quick else 1,
        assistant_model=assistant_model,
        generate_kwargs=assisted_generate_kwargs,
    )
    suite.add("decode_tok_per_s", speed["decode_tok_per_s"], "tok/s")
    suite.add("total_tok_per_s", speed["tok_per_s"], "tok/s")
    suite.add("avg_latency", speed["avg_latency_s"], "seconds")
    print(f"  Decode: {speed['decode_tok_per_s']:.1f} tok/s")
    print(f"  Total:  {speed['tok_per_s']:.1f} tok/s")
    print(f"  Latency: {speed['avg_latency_s']:.2f}s per prompt")

    # 7. Quality spot-check
    if not quick:
        print("\n=== Quality Spot-Check ===")
        correct = 0
        input_device = get_model_input_device(model)
        for prompt, expected in QUALITY_PROMPTS:
            if assisted_generate_kwargs and torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
            inputs = processor(text=prompt, return_tensors="pt").to(input_device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    temperature=0.0,
                    do_sample=False,
                    **assisted_generate_kwargs,
                )
            response = processor.decode(
                output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            )
            is_correct = expected.lower() in response.lower()
            correct += int(is_correct)
            status = "PASS" if is_correct else "FAIL"
            print(f"  [{status}] Q: {prompt[:50]}  A: {response.strip()[:50]}  Expected: {expected}")
        suite.add("quality_accuracy", correct / len(QUALITY_PROMPTS) * 100, "%")

    # 8. Peak VRAM
    vram = measure_vram()
    suite.add("peak_vram", vram.get("max_allocated_mb", 0), "MB")
    print(f"\n  Peak VRAM: {vram.get('max_allocated_mb', 0):.0f} MB")

    # 9. Save results
    suite.print_summary()
    results_dir = PROJECT_ROOT / "experiments" / "exp5_gemma4" / "results"
    model_tag = model_id.split("/")[-1].lower()
    suite.save(results_dir / f"optimized_{model_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    # 10. Compare with baseline if provided
    if compare_file:
        _compare_results(compare_file, suite)


def _compare_results(baseline_path: str, optimized: BenchmarkSuite) -> None:
    """Compare optimized results against a baseline."""
    baseline_file = Path(baseline_path)
    if not baseline_file.exists():
        print(f"\nBaseline file not found: {baseline_path}")
        return

    with open(baseline_file) as f:
        baseline_data = json.load(f)

    print("\n" + "=" * 60)
    print("=== Baseline vs Optimized Comparison ===")
    print("=" * 60)

    baseline_results = {r["name"]: r for r in baseline_data.get("results", [])}
    optimized_results = {r.name: r for r in optimized.results}

    for metric in ["decode_tok_per_s", "total_tok_per_s", "avg_latency",
                    "peak_vram", "gpu_memory_after_load", "quality_accuracy"]:
        base = baseline_results.get(metric, {})
        opt = optimized_results.get(metric)
        if base and opt:
            base_val = base.get("value", 0)
            opt_val = opt.value
            if base_val > 0:
                change = ((opt_val - base_val) / base_val) * 100
                direction = "+" if change > 0 else ""
                print(f"  {metric:30s}: {base_val:>10.1f} ->{opt_val:>10.1f} ({direction}{change:.1f}%)")
            else:
                print(f"  {metric:30s}: {base_val:>10.1f} ->{opt_val:>10.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma 4 optimized benchmark")
    parser.add_argument(
        "--model", default="google/gemma-4-26B-A4B-it",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--quantization", default="streaming-int4",
        choices=["streaming-int4", "quanto-int4", "bnb-nf4", "gptq", "awq", "prequantized", "none"],
        help="Quantization method",
    )
    parser.add_argument("--quick", action="store_true", help="Quick test")
    parser.add_argument("--no-turboquant", action="store_true", help="Disable TurboQuant")
    parser.add_argument("--no-yarn", action="store_true", help="Disable YaRN")
    parser.add_argument("--tq-key-bits", type=int, default=4, help="TurboQuant key bits")
    parser.add_argument("--tq-value-bits", type=int, default=2, help="TurboQuant value bits")
    parser.add_argument("--yarn-context", type=int, default=1_048_576, help="YaRN target context")
    parser.add_argument("--compare", type=str, default=None, help="Baseline JSON to compare against")
    parser.add_argument("--assistant-model", type=str, default=None, help="Optional draft model for exact assisted decoding")
    parser.add_argument(
        "--assistant-quantization",
        default="quanto-int4",
        choices=["streaming-int4", "quanto-int4", "bnb-nf4", "gptq", "awq", "prequantized", "none"],
        help="Draft-model quantization when assisted decoding is enabled",
    )
    parser.add_argument("--assistant-gpu-memory-gb", type=float, default=0.0, help="Draft-model GPU budget")
    parser.add_argument("--assistant-cpu-memory-gb", type=float, default=10.0, help="Draft-model CPU budget")
    parser.add_argument("--assistant-num-tokens", type=int, default=8, help="Initial speculative token budget")
    parser.add_argument(
        "--assistant-schedule",
        type=str,
        default="heuristic_transient",
        choices=["heuristic", "heuristic_transient", "constant"],
        help="Assistant token schedule",
    )
    parser.add_argument(
        "--assistant-confidence-threshold",
        type=float,
        default=0.4,
        help="Assistant confidence threshold",
    )
    args = parser.parse_args()

    assisted_decoding = None
    if args.assistant_model:
        assisted_decoding = AssistedDecodingConfig(
            assistant_model_id=args.assistant_model,
            assistant_quantization=args.assistant_quantization,
            assistant_gpu_memory_gb=args.assistant_gpu_memory_gb,
            assistant_cpu_memory_gb=args.assistant_cpu_memory_gb,
            num_assistant_tokens=args.assistant_num_tokens,
            num_assistant_tokens_schedule=args.assistant_schedule,
            assistant_confidence_threshold=args.assistant_confidence_threshold,
        )

    run_optimized_benchmark(
        model_id=args.model,
        quantization=args.quantization,
        quick=args.quick,
        enable_turboquant=not args.no_turboquant,
        enable_yarn=not args.no_yarn,
        tq_key_bits=args.tq_key_bits,
        tq_value_bits=args.tq_value_bits,
        yarn_target_context=args.yarn_context,
        compare_file=args.compare,
        assisted_decoding=assisted_decoding,
    )


if __name__ == "__main__":
    main()
