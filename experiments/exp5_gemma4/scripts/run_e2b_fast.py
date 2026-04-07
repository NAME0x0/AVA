"""Fast local Gemma 4 E2B runner for the 4 GB VRAM / 32 GB RAM target.

This is the practical low-latency branch intended to feel fast on local
hardware. It keeps the deep E4B branch separate, so speed-oriented defaults
can be more aggressive here.
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
from experiments.exp5_gemma4.engine.llamacpp import (
    LlamaCppServer,
    LlamaCppServerConfig,
    benchmark_llamacpp_chat_speed,
)
from experiments.exp5_gemma4.engine.loader import inspect_model_architecture, load_model
from experiments.exp5_gemma4.engine.patches import get_model_memory_footprint
from experiments.exp5_gemma4.engine.practical import (
    PRACTICAL_PROMPTS,
    apply_practical_optimizations,
    benchmark_chat_speed,
    decode_chat_response,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast Gemma 4 E2B runner")
    parser.add_argument("--model", default="google/gemma-4-E2B-it", help="Hugging Face model ID")
    parser.add_argument(
        "--backend",
        default="transformers",
        choices=["transformers", "llama.cpp"],
        help="Use the current Transformers runtime or a local llama.cpp server",
    )
    parser.add_argument(
        "--quantization",
        default="none",
        choices=["none", "quanto-int4", "bnb-nf4", "gptq", "awq", "prequantized"],
        help="E2B fast branch defaults to bf16 + manual GPU placement",
    )
    parser.add_argument("--gpu-memory-gb", type=float, default=3.5)
    parser.add_argument("--cpu-memory-gb", type=float, default=24.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--cache-implementation", default="static", help="KV cache backend for generation")
    parser.add_argument("--quick", action="store_true", help="Use one short prompt")
    parser.add_argument("--enable-thinking", action="store_true", help="Keep Gemma thinking enabled")
    parser.add_argument("--no-turboquant", action="store_true", help="Disable TurboQuant")
    parser.add_argument("--no-yarn", action="store_true", help="Disable YaRN")
    parser.add_argument("--tq-key-bits", type=int, default=4)
    parser.add_argument("--tq-value-bits", type=int, default=2)
    parser.add_argument(
        "--yarn-context",
        type=int,
        default=262_144,
        help="Fast-path context target; deep E4B path keeps the higher 512K budget",
    )
    parser.add_argument("--llama-server-exe", help="Path to llama-server executable")
    parser.add_argument("--llama-model", help="Local GGUF file for llama.cpp")
    parser.add_argument("--llama-hf-repo", help="GGUF Hugging Face repo, e.g. ggml-org/gemma-4-E2B-it-GGUF:Q4_K_M")
    parser.add_argument("--llama-hf-file", help="Specific GGUF filename inside --llama-hf-repo")
    parser.add_argument("--llama-hf-token", help="Optional Hugging Face token for llama.cpp downloads")
    parser.add_argument("--llama-quant", help="Override the default GGUF quant when auto-mapping the repo")
    parser.add_argument("--llama-port", type=int, default=0, help="llama.cpp server port (0 = auto)")
    parser.add_argument("--llama-gpu-layers", default="-1", help="llama.cpp --gpu-layers value")
    parser.add_argument("--llama-flash-attn", default="on", choices=["on", "off", "auto"])
    parser.add_argument("--llama-cache-type-k", default="f16")
    parser.add_argument("--llama-cache-type-v", default="f16")
    parser.add_argument("--llama-threads", type=int)
    parser.add_argument("--llama-threads-batch", type=int)
    parser.add_argument("--llama-chat-template", help="Optional llama.cpp built-in chat template override")
    parser.add_argument("--llama-offline", action="store_true", help="Force llama.cpp to stay offline and use cache only")
    args = parser.parse_args()

    suite = BenchmarkSuite(
        config={
            "model_id": args.model,
            "backend": args.backend,
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

    results_dir = PROJECT_ROOT / "experiments" / "exp5_gemma4" / "results"
    model_tag = args.model.split("/")[-1].lower()
    suffix = "thinking" if args.enable_thinking else "fast"

    if args.backend == "llama.cpp":
        print("\n=== Starting E2B Fast Model via llama.cpp ===")
        server_config = LlamaCppServerConfig(
            model_id=args.model,
            executable=args.llama_server_exe,
            model_path=args.llama_model,
            hf_repo=args.llama_hf_repo,
            hf_file=args.llama_hf_file,
            hf_token=args.llama_hf_token,
            gguf_quant=args.llama_quant,
            port=args.llama_port,
            ctx_size=args.yarn_context,
            gpu_layers=args.llama_gpu_layers,
            flash_attn=args.llama_flash_attn,
            cache_type_k=args.llama_cache_type_k,
            cache_type_v=args.llama_cache_type_v,
            reasoning="on" if args.enable_thinking else "off",
            reasoning_budget=-1 if args.enable_thinking else 0,
            reasoning_format="deepseek",
            chat_template=args.llama_chat_template,
            threads=args.llama_threads,
            threads_batch=args.llama_threads_batch,
            alias=model_tag,
            offline=args.llama_offline,
        )
        log_path = results_dir / "llamacpp_logs" / f"e2b_{suffix}_{model_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        with LlamaCppServer(server_config, log_path=log_path) as server:
            suite.add("load_time", server.startup_elapsed_s, "seconds")
            suite.config["llamacpp"] = {
                "command": server.command,
                "port": server.port,
                "log_path": str(log_path),
                "ctx_size": args.yarn_context,
                "gpu_layers": args.llama_gpu_layers,
                "cache_type_k": args.llama_cache_type_k,
                "cache_type_v": args.llama_cache_type_v,
                "flash_attn": args.llama_flash_attn,
                "chat_template": args.llama_chat_template,
            }
            prompts = PRACTICAL_PROMPTS[:1] if args.quick else PRACTICAL_PROMPTS
            print("\n=== Fast Chat Speed ===")
            speed = benchmark_llamacpp_chat_speed(
                server,
                prompts,
                max_new_tokens=args.max_new_tokens,
                enable_thinking=args.enable_thinking,
            )
            suite.add("decode_tok_per_s", speed["decode_tok_per_s"], "tok/s")
            suite.add("total_tok_per_s", speed["tok_per_s"], "tok/s")
            suite.add("avg_latency", speed["avg_latency_s"], "seconds")
            print(f"  Decode:  {speed['decode_tok_per_s']:.2f} tok/s")
            print(f"  Total:   {speed['tok_per_s']:.2f} tok/s")
            print(f"  Latency: {speed['avg_latency_s']:.2f}s per prompt")

            print("\n=== Sanity Prompt ===")
            sanity_prompt = "What is the capital of France?"
            response = server.generate_chat_response(
                sanity_prompt,
                max_new_tokens=32,
                enable_thinking=args.enable_thinking,
            )
            print(f"  Response: {response['response'].encode('ascii', 'backslashreplace').decode('ascii')}")

            suite.print_summary()
            suite.save(results_dir / f"e2b_{suffix}_{model_tag}_llamacpp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        return

    print("\n=== Loading E2B Fast Model ===")
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

    print("\n=== Applying Fast-Path Optimizations ===")
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
    print("\n=== Fast Chat Speed ===")
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
    suite.save(results_dir / f"e2b_{suffix}_{model_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")


if __name__ == "__main__":
    main()
