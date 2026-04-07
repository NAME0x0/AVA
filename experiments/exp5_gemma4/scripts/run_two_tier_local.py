"""Local two-tier Gemma 4 runtime: fast E2B by default, deep E4B on demand."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp5_gemma4.engine.benchmark import BenchmarkSuite, measure_vram, reset_vram_stats
from experiments.exp5_gemma4.engine.llamacpp import LlamaCppServer, LlamaCppServerConfig
from experiments.exp5_gemma4.engine.loader import (
    inspect_model_architecture,
    load_model,
    release_manual_gpu_residency,
    restore_manual_gpu_residency,
)
from experiments.exp5_gemma4.engine.patches import get_model_memory_footprint
from experiments.exp5_gemma4.engine.practical import (
    PRACTICAL_PROMPTS,
    apply_practical_optimizations,
    generate_chat_response,
)
from experiments.exp5_gemma4.engine.tiered import RoutingConfig, route_prompt


def _default_fast_llama_model() -> str | None:
    candidate = (
        PROJECT_ROOT
        / "experiments"
        / "exp5_gemma4"
        / "gguf"
        / "gemma-4-e2b-it-q8_0"
        / "gemma-4-e2b-it-Q8_0.gguf"
    )
    return str(candidate) if candidate.exists() else None


def _default_fast_llama_server() -> str | None:
    candidate = (
        PROJECT_ROOT
        / "experiments"
        / "exp5_gemma4"
        / "tools"
        / "llama.cpp-head"
        / "build"
        / "bin"
        / "llama-server.exe"
    )
    return str(candidate) if candidate.exists() else None


def load_branch(
    *,
    name: str,
    backend: str,
    model_id: str,
    quantization: str,
    gpu_memory_gb: float,
    cpu_memory_gb: float,
    enable_turboquant: bool,
    enable_yarn: bool,
    tq_key_bits: int,
    tq_value_bits: int,
    tq_protected_layers: int,
    yarn_context: int,
    enable_thinking: bool,
    cache_implementation: str,
    llama_server_exe: str | None = None,
    llama_model: str | None = None,
    llama_hf_repo: str | None = None,
    llama_hf_file: str | None = None,
    llama_hf_token: str | None = None,
    llama_quant: str | None = None,
    llama_port: int = 0,
    llama_gpu_layers: str = "-1",
    llama_flash_attn: str = "on",
    llama_cache_type_k: str = "f16",
    llama_cache_type_v: str = "f16",
    llama_threads: int | None = None,
    llama_threads_batch: int | None = None,
    llama_chat_template: str | None = None,
    llama_offline: bool = True,
) -> dict[str, Any]:
    """Load one branch of the local two-tier runtime."""
    print(f"\n=== Loading {name.title()} Branch ===")
    if backend == "llama.cpp":
        results_dir = PROJECT_ROOT / "experiments" / "exp5_gemma4" / "results"
        log_path = (
            results_dir
            / "llamacpp_logs"
            / f"{name}_{model_id.split('/')[-1].lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        server_config = LlamaCppServerConfig(
            model_id=model_id,
            executable=llama_server_exe,
            model_path=llama_model or _default_fast_llama_model(),
            hf_repo=llama_hf_repo,
            hf_file=llama_hf_file,
            hf_token=llama_hf_token,
            gguf_quant=llama_quant,
            port=llama_port,
            ctx_size=yarn_context,
            gpu_layers=llama_gpu_layers,
            flash_attn=llama_flash_attn,
            cache_type_k=llama_cache_type_k,
            cache_type_v=llama_cache_type_v,
            reasoning="on" if enable_thinking else "off",
            reasoning_budget=-1 if enable_thinking else 0,
            reasoning_format="deepseek",
            chat_template=llama_chat_template,
            threads=llama_threads,
            threads_batch=llama_threads_batch,
            alias=model_id.split("/")[-1].lower(),
            offline=llama_offline,
        )
        server = LlamaCppServer(server_config, log_path=log_path)
        load_meta = server.start()
        return {
            "name": name,
            "backend": backend,
            "model_id": model_id,
            "server": server,
            "gpu_memory_gb": gpu_memory_gb,
            "gpu_active": True,
            "load_meta": load_meta,
            "architecture": {"backend": "llama.cpp"},
            "optimizations": {
                "flash_attn": llama_flash_attn,
                "cache_type_k": llama_cache_type_k,
                "cache_type_v": llama_cache_type_v,
                "ctx_size": yarn_context,
            },
            "memory_footprint": {
                "backend": "llama.cpp",
                "gpu_layers": llama_gpu_layers,
                "cache_implementation": cache_implementation,
            },
            "load_args": {
                "name": name,
                "backend": backend,
                "model_id": model_id,
                "quantization": quantization,
                "gpu_memory_gb": gpu_memory_gb,
                "cpu_memory_gb": cpu_memory_gb,
                "enable_turboquant": enable_turboquant,
                "enable_yarn": enable_yarn,
                "tq_key_bits": tq_key_bits,
                "tq_value_bits": tq_value_bits,
                "tq_protected_layers": tq_protected_layers,
                "yarn_context": yarn_context,
                "enable_thinking": enable_thinking,
                "cache_implementation": cache_implementation,
                "llama_server_exe": llama_server_exe,
                "llama_model": llama_model,
                "llama_hf_repo": llama_hf_repo,
                "llama_hf_file": llama_hf_file,
                "llama_hf_token": llama_hf_token,
                "llama_quant": llama_quant,
                "llama_port": llama_port,
                "llama_gpu_layers": llama_gpu_layers,
                "llama_flash_attn": llama_flash_attn,
                "llama_cache_type_k": llama_cache_type_k,
                "llama_cache_type_v": llama_cache_type_v,
                "llama_threads": llama_threads,
                "llama_threads_batch": llama_threads_batch,
                "llama_chat_template": llama_chat_template,
                "llama_offline": llama_offline,
            },
        }

    model, processor, load_meta = load_model(
        model_id=model_id,
        quantization=quantization,
        gpu_memory_gb=gpu_memory_gb,
        cpu_memory_gb=cpu_memory_gb,
    )
    arch_info = inspect_model_architecture(model)
    print(json.dumps(arch_info, indent=2))
    opt_meta = apply_practical_optimizations(
        model,
        arch_info,
        enable_turboquant=enable_turboquant,
        enable_yarn=enable_yarn,
        tq_key_bits=tq_key_bits,
        tq_value_bits=tq_value_bits,
        tq_protected_layers=tq_protected_layers,
        yarn_target_context=yarn_context,
    )
    footprint = get_model_memory_footprint(model)
    return {
        "name": name,
        "backend": backend,
        "model_id": model_id,
        "model": model,
        "processor": processor,
        "gpu_memory_gb": gpu_memory_gb,
        "gpu_active": bool(getattr(model, "_manual_gpu_layers", 0)),
        "load_meta": load_meta,
        "architecture": arch_info,
        "optimizations": opt_meta,
        "memory_footprint": footprint,
        "load_args": {
            "name": name,
            "backend": backend,
            "model_id": model_id,
            "quantization": quantization,
            "gpu_memory_gb": gpu_memory_gb,
            "cpu_memory_gb": cpu_memory_gb,
            "enable_turboquant": enable_turboquant,
            "enable_yarn": enable_yarn,
            "tq_key_bits": tq_key_bits,
            "tq_value_bits": tq_value_bits,
            "tq_protected_layers": tq_protected_layers,
            "yarn_context": yarn_context,
            "enable_thinking": enable_thinking,
            "cache_implementation": cache_implementation,
            "llama_server_exe": llama_server_exe,
            "llama_model": llama_model,
            "llama_hf_repo": llama_hf_repo,
            "llama_hf_file": llama_hf_file,
            "llama_hf_token": llama_hf_token,
            "llama_quant": llama_quant,
            "llama_port": llama_port,
            "llama_gpu_layers": llama_gpu_layers,
            "llama_flash_attn": llama_flash_attn,
            "llama_cache_type_k": llama_cache_type_k,
            "llama_cache_type_v": llama_cache_type_v,
            "llama_threads": llama_threads,
            "llama_threads_batch": llama_threads_batch,
            "llama_chat_template": llama_chat_template,
            "llama_offline": llama_offline,
        },
    }


def load_branch_from_spec(branch: dict[str, Any]) -> dict[str, Any]:
    """Reload a previously unloaded branch from its saved spec."""
    if branch.get("backend") == "llama.cpp" and "server" in branch:
        return branch
    if "model" in branch and "processor" in branch:
        return branch
    reloaded = load_branch(**branch["load_args"])
    branch.update(reloaded)
    return branch


def unload_branch(branch: dict[str, Any]) -> None:
    """Free the inactive branch so the other model can fit on this machine."""
    if branch.get("backend") == "llama.cpp":
        server = branch.pop("server", None)
        if server is not None:
            server.stop()
        branch["gpu_active"] = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    if "model" not in branch:
        return

    release_manual_gpu_residency(branch["model"])
    branch["gpu_active"] = False
    branch.pop("model", None)
    branch.pop("processor", None)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def activate_loaded_branch(
    target_branch: dict[str, Any],
    current_branch: dict[str, Any] | None,
) -> tuple[dict[str, Any], float, bool]:
    """Ensure only one loaded branch owns GPU residency at a time."""
    if target_branch.get("backend") == "llama.cpp":
        return target_branch, 0.0, False
    if current_branch is target_branch and target_branch.get("gpu_active", False):
        return target_branch, 0.0, False

    t0 = time.perf_counter()
    switched = False

    if current_branch is not None and current_branch.get("gpu_active", False):
        print(f"  Releasing GPU residency from {current_branch['name']}")
        release_manual_gpu_residency(current_branch["model"])
        current_branch["gpu_active"] = False
        switched = True

    if not target_branch.get("gpu_active", False):
        print(f"  Activating GPU residency for {target_branch['name']}")
        restore_manual_gpu_residency(
            target_branch["model"],
            gpu_memory_gb=target_branch["gpu_memory_gb"],
        )
        target_branch["gpu_active"] = bool(getattr(target_branch["model"], "_manual_gpu_layers", 0))
        switched = True

    return target_branch, time.perf_counter() - t0, switched


def main() -> None:
    parser = argparse.ArgumentParser(description="Local two-tier Gemma 4 runtime")
    parser.add_argument("--fast-model", default="google/gemma-4-E2B-it")
    parser.add_argument("--deep-model", default="google/gemma-4-E4B-it")
    parser.add_argument("--fast-backend", default="llama.cpp", choices=["transformers", "llama.cpp"])
    parser.add_argument("--fast-quantization", default="none")
    parser.add_argument("--deep-quantization", default="none")
    parser.add_argument("--fast-gpu-memory-gb", type=float, default=3.5)
    parser.add_argument("--deep-gpu-memory-gb", type=float, default=3.5)
    parser.add_argument("--fast-cpu-memory-gb", type=float, default=24.0)
    parser.add_argument("--deep-cpu-memory-gb", type=float, default=28.0)
    parser.add_argument("--fast-yarn-context", type=int, default=262_144)
    parser.add_argument("--deep-yarn-context", type=int, default=524_288)
    parser.add_argument("--fast-cache-implementation", default="static")
    parser.add_argument("--deep-cache-implementation", default="static")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--quick", action="store_true", help="Use one short prompt")
    parser.add_argument("--prompt", action="append", default=[], help="Custom prompt; repeatable")
    parser.add_argument("--force-fast", action="store_true", help="Bypass routing and stay on E2B")
    parser.add_argument("--force-deep", action="store_true", help="Bypass routing and always use E4B")
    parser.add_argument("--route-only", action="store_true", help="Print route decisions without loading models")
    parser.add_argument("--warm-deep", action="store_true", help="Eagerly load the deep model at startup")
    parser.add_argument("--fast-thinking", action="store_true", help="Keep thinking enabled on E2B")
    parser.add_argument("--deep-thinking", action="store_true", help="Keep thinking enabled on E4B")
    parser.add_argument(
        "--llama-server-exe",
        default=_default_fast_llama_server(),
        help="Path to llama-server executable for the fast E2B branch",
    )
    parser.add_argument("--llama-model", default=_default_fast_llama_model(), help="Local GGUF file for fast E2B llama.cpp")
    parser.add_argument("--llama-hf-repo", help="GGUF Hugging Face repo override for the fast E2B llama.cpp branch")
    parser.add_argument("--llama-hf-file", help="Specific GGUF filename override for the fast E2B llama.cpp branch")
    parser.add_argument("--llama-hf-token", help="Optional Hugging Face token for llama.cpp downloads")
    parser.add_argument("--llama-quant", help="Override the default GGUF quant when auto-mapping the fast branch repo")
    parser.add_argument("--llama-port", type=int, default=0, help="llama.cpp server port for the fast branch")
    parser.add_argument("--llama-gpu-layers", default="-1", help="llama.cpp --gpu-layers value for the fast branch")
    parser.add_argument("--llama-flash-attn", default="on", choices=["on", "off", "auto"])
    parser.add_argument("--llama-cache-type-k", default="f16")
    parser.add_argument("--llama-cache-type-v", default="f16")
    parser.add_argument("--llama-threads", type=int)
    parser.add_argument("--llama-threads-batch", type=int)
    parser.add_argument("--llama-chat-template", help="Optional llama.cpp built-in chat template override for the fast branch")
    parser.add_argument("--llama-offline", action="store_true", help="Force llama.cpp to stay offline on the fast branch")
    parser.add_argument("--max-fast-chars", type=int, default=320)
    parser.add_argument("--max-fast-words", type=int, default=72)
    parser.add_argument("--max-fast-lines", type=int, default=6)
    args = parser.parse_args()

    prompts = args.prompt or (PRACTICAL_PROMPTS[:1] if args.quick else PRACTICAL_PROMPTS)
    routing_cfg = RoutingConfig(
        fast_model_id=args.fast_model,
        deep_model_id=args.deep_model,
        max_fast_chars=args.max_fast_chars,
        max_fast_words=args.max_fast_words,
        max_fast_lines=args.max_fast_lines,
    )

    if args.route_only:
        print("\n=== Route Decisions ===")
        for prompt in prompts:
            decision = route_prompt(prompt, routing_cfg)
            print(f"[{decision.tier.upper()}] {decision.reason}")
            if decision.thinking_override is not None:
                print(f"  Thinking: {'on' if decision.thinking_override else 'off'}")
            print(f"  Prompt: {decision.cleaned_prompt.encode('ascii', 'backslashreplace').decode('ascii')}")
        return

    if args.force_fast and args.force_deep:
        raise SystemExit("--force-fast and --force-deep cannot be combined")
    if args.warm_deep and not args.force_deep:
        print("Warm deep is disabled on this 4 GB VRAM path; deep will load lazily instead.")
        args.warm_deep = False

    suite = BenchmarkSuite(
        config={
            "fast_model_id": args.fast_model,
            "deep_model_id": args.deep_model,
            "fast_backend": args.fast_backend,
            "fast_quantization": args.fast_quantization,
            "deep_quantization": args.deep_quantization,
            "fast_cache_implementation": args.fast_cache_implementation,
            "deep_cache_implementation": args.deep_cache_implementation,
            "routing": {
                "max_fast_chars": args.max_fast_chars,
                "max_fast_words": args.max_fast_words,
                "max_fast_lines": args.max_fast_lines,
            },
        },
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    reset_vram_stats()
    fast_branch = load_branch(
        name="fast",
        backend=args.fast_backend,
        model_id=args.fast_model,
        quantization=args.fast_quantization,
        gpu_memory_gb=args.fast_gpu_memory_gb,
        cpu_memory_gb=args.fast_cpu_memory_gb,
        enable_turboquant=True,
        enable_yarn=True,
        tq_key_bits=4,
        tq_value_bits=2,
        tq_protected_layers=1,
        yarn_context=args.fast_yarn_context,
        enable_thinking=args.fast_thinking,
        cache_implementation=args.fast_cache_implementation,
        llama_server_exe=args.llama_server_exe,
        llama_model=args.llama_model,
        llama_hf_repo=args.llama_hf_repo,
        llama_hf_file=args.llama_hf_file,
        llama_hf_token=args.llama_hf_token,
        llama_quant=args.llama_quant,
        llama_port=args.llama_port,
        llama_gpu_layers=args.llama_gpu_layers,
        llama_flash_attn=args.llama_flash_attn,
        llama_cache_type_k=args.llama_cache_type_k,
        llama_cache_type_v=args.llama_cache_type_v,
        llama_threads=args.llama_threads,
        llama_threads_batch=args.llama_threads_batch,
        llama_chat_template=args.llama_chat_template,
        llama_offline=args.llama_offline or bool(args.llama_model),
    )
    suite.config["fast_branch"] = {
        "load_metadata": fast_branch["load_meta"],
        "architecture": fast_branch["architecture"],
        "optimizations": fast_branch["optimizations"],
        "memory_footprint": fast_branch["memory_footprint"],
    }

    deep_branch: dict[str, Any] = {
        "name": "deep",
        "model_id": args.deep_model,
        "gpu_memory_gb": args.deep_gpu_memory_gb,
        "gpu_active": False,
        "load_args": {
            "name": "deep",
            "backend": "transformers",
            "model_id": args.deep_model,
            "quantization": args.deep_quantization,
            "gpu_memory_gb": args.deep_gpu_memory_gb,
            "cpu_memory_gb": args.deep_cpu_memory_gb,
            "enable_turboquant": True,
            "enable_yarn": True,
            "tq_key_bits": 4,
            "tq_value_bits": 2,
            "tq_protected_layers": 1,
            "yarn_context": args.deep_yarn_context,
            "enable_thinking": args.deep_thinking,
            "cache_implementation": args.deep_cache_implementation,
            "llama_server_exe": None,
            "llama_model": None,
            "llama_hf_repo": None,
            "llama_hf_file": None,
            "llama_hf_token": None,
            "llama_quant": None,
            "llama_port": 0,
            "llama_gpu_layers": "-1",
            "llama_flash_attn": "on",
            "llama_cache_type_k": "f16",
            "llama_cache_type_v": "f16",
            "llama_threads": None,
            "llama_threads_batch": None,
            "llama_chat_template": None,
            "llama_offline": True,
        },
    }
    if args.warm_deep or args.force_deep:
        if fast_branch.get("gpu_active", False):
            print("  Releasing fast branch before deep warm-up")
            unload_branch(fast_branch)
        deep_branch = load_branch_from_spec(deep_branch)
        suite.config["deep_branch"] = {
            "load_metadata": deep_branch["load_meta"],
            "architecture": deep_branch["architecture"],
            "optimizations": deep_branch["optimizations"],
            "memory_footprint": deep_branch["memory_footprint"],
        }

    total_output_tokens = 0
    total_time_s = 0.0
    total_request_time_s = 0.0
    total_switch_overhead_s = 0.0
    route_counts = {"fast": 0, "deep": 0}
    branch_switches = 0
    active_branch: dict[str, Any] | None = deep_branch if args.force_deep else fast_branch

    print("\n=== Two-Tier Responses ===")
    for prompt in prompts:
        branch_management_s = 0.0
        if args.force_fast:
            decision = route_prompt("fast: " + prompt, routing_cfg)
        elif args.force_deep:
            decision = route_prompt("deep: " + prompt, routing_cfg)
        else:
            decision = route_prompt(prompt, routing_cfg)

        branch = fast_branch
        thinking = args.fast_thinking
        cache_implementation = args.fast_cache_implementation

        if decision.tier == "deep":
            branch = deep_branch
            thinking = args.deep_thinking
            cache_implementation = args.deep_cache_implementation

        if decision.thinking_override is not None:
            thinking = decision.thinking_override

        if active_branch is not None and active_branch is not branch:
            print(f"  Unloading inactive {active_branch['name']} branch before switch")
            t0 = time.perf_counter()
            unload_branch(active_branch)
            branch_management_s += time.perf_counter() - t0
            active_branch = None
            branch_switches += 1

        if "model" not in branch:
            print(f"  Loading {branch['name']} branch on demand")
            t0 = time.perf_counter()
            branch = load_branch_from_spec(branch)
            branch_management_s += time.perf_counter() - t0
            if branch["name"] == "fast":
                fast_branch = branch
            else:
                deep_branch = branch
                suite.config["deep_branch"] = {
                    "load_metadata": branch["load_meta"],
                    "architecture": branch["architecture"],
                    "optimizations": branch["optimizations"],
                    "memory_footprint": branch["memory_footprint"],
                }

        active_branch, switch_overhead_s, switched = activate_loaded_branch(branch, active_branch)
        branch_management_s += switch_overhead_s
        total_switch_overhead_s += branch_management_s
        branch_switches += int(switched)
        route_counts[decision.tier] += 1
        request_t0 = time.perf_counter()
        if branch.get("backend") == "llama.cpp":
            result = branch["server"].generate_chat_response(
                decision.cleaned_prompt,
                max_new_tokens=args.max_new_tokens,
                enable_thinking=thinking,
            )
        else:
            result = generate_chat_response(
                branch["model"],
                branch["processor"],
                decision.cleaned_prompt,
                max_new_tokens=args.max_new_tokens,
                enable_thinking=thinking,
                cache_implementation=cache_implementation,
            )
        total_output_tokens += result["output_tokens"]
        total_time_s += result["elapsed_s"]
        total_request_time_s += time.perf_counter() - request_t0 + branch_management_s

        print(f"[{decision.tier.upper()}] {decision.reason}")
        print(f"  Response: {result['response'].encode('ascii', 'backslashreplace').decode('ascii')}")

    suite.add("decode_tok_per_s", total_output_tokens / total_time_s if total_time_s > 0 else 0.0, "tok/s")
    suite.add("avg_latency", total_time_s / len(prompts) if prompts else 0.0, "seconds")
    suite.add("avg_request_latency", total_request_time_s / len(prompts) if prompts else 0.0, "seconds")
    suite.add("switch_overhead", total_switch_overhead_s, "seconds")
    suite.add("branch_switches", float(branch_switches), "count")
    suite.add("fast_routes", float(route_counts["fast"]), "count")
    suite.add("deep_routes", float(route_counts["deep"]), "count")

    vram = measure_vram()
    suite.add("peak_vram", vram.get("max_allocated_mb", 0.0), "MB")
    suite.print_summary()
    results_dir = PROJECT_ROOT / "experiments" / "exp5_gemma4" / "results"
    suite.save(results_dir / f"two_tier_local_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")


if __name__ == "__main__":
    main()
