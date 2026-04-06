"""Model loading with smart CPU/GPU memory management.

Loads Gemma 4 26B MoE with BitsAndBytes 4-bit quantization and
accelerate device_map to split between GPU (4 GB) and CPU (32 GB).

The loader also prepares the model for our custom optimizations:
  - Extracts MoE expert references for the offloader
  - Identifies global vs sliding attention layers
  - Configures the model for extended context via YaRN
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch


def get_memory_config(
    gpu_memory_gb: float = 3.5,
    cpu_memory_gb: float = 28.0,
) -> dict[str, str]:
    """Build max_memory dict for accelerate device_map."""
    mem = {"cpu": f"{cpu_memory_gb:.0f}GiB"}
    if torch.cuda.is_available():
        mem[0] = f"{gpu_memory_gb:.1f}GiB"  # accelerate expects int device keys
    return mem


def _load_with_quanto(
    model_id: str,
    torch_dtype: torch.dtype,
    max_memory: dict[str, str],
) -> tuple[Any, Any]:
    """Load model to CPU in bf16, quantize with quanto int4, then dispatch.

    This bypasses the limitation where both BnB and Quanto refuse
    on-the-fly quantization with CPU+GPU device_map.  Instead:
      1. Load bf16 weights to CPU (streamed, low memory)
      2. Quantize all linear layers to int4 in-place on CPU
      3. Use accelerate to dispatch layers across CPU and GPU
    """
    from transformers import AutoModelForCausalLM, AutoProcessor

    print("  Step 1/3: Loading bf16 weights to CPU (streamed)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(model_id)

    print("  Step 2/3: Quantizing to int4 with quanto...")
    from optimum.quanto import quantize, qint4, freeze
    quantize(model, weights=qint4)
    freeze(model)  # Freeze quantized weights (replaces dynamic quantization with static)

    print("  Step 3/3: Dispatching across CPU and GPU...")
    from accelerate import infer_auto_device_map, dispatch_model
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=model._no_split_modules if hasattr(model, "_no_split_modules") else [],
    )
    model = dispatch_model(model, device_map=device_map)

    return model, processor


def _load_prequantized(
    model_id: str,
    torch_dtype: torch.dtype,
    max_memory: dict[str, str],
    offload_folder: str | None = None,
) -> tuple[Any, Any]:
    """Load a pre-quantized model (GPTQ, AWQ, or auto-detected).

    Pre-quantized models have weights already compressed on disk, so they
    never materialize the full bf16 model in RAM.  This is the only way to
    load the 26B MoE on a 32 GB RAM machine (bf16 = 50 GB).

    Supports:
      - GPTQ models (quantization_config in config.json has quant_method="gptq")
      - AWQ models (quant_method="awq")
      - Any model with embedded quantization config (auto-detected by transformers)
    """
    from transformers import AutoModelForCausalLM, AutoProcessor

    print("  Loading pre-quantized model with device_map='auto'...")

    load_kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": model_id,
        "device_map": "auto",
        "max_memory": max_memory,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if offload_folder:
        load_kwargs["offload_folder"] = offload_folder
        Path(offload_folder).mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    processor = AutoProcessor.from_pretrained(model_id)

    return model, processor


def load_model(
    model_id: str = "google/gemma-4-26B-A4B-it",
    quantization: str = "quanto-int4",
    gpu_memory_gb: float = 3.5,
    cpu_memory_gb: float = 28.0,
    dtype: str = "bfloat16",
    offload_folder: str | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load Gemma 4 with quantization and CPU/GPU split.

    Args:
        quantization: "quanto-int4" (CPU+GPU compatible),
                      "bnb-nf4" (GPU-only, fails with CPU offload),
                      "gptq" (load pre-quantized GPTQ model),
                      "awq" (load pre-quantized AWQ model),
                      "prequantized" (auto-detect from model config),
                      "none" (bf16, needs offload_folder for large models)

    Returns:
        model: the loaded model
        processor: tokenizer/processor
        metadata: dict with loading stats (time, memory, device_map info)
    """
    from transformers import AutoModelForCausalLM, AutoProcessor

    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    max_memory = get_memory_config(gpu_memory_gb, cpu_memory_gb)

    print(f"Loading {model_id}")
    print(f"  Quantization: {quantization}")
    print(f"  Memory budget: {json.dumps({str(k): v for k, v in max_memory.items()})}")

    t0 = time.perf_counter()

    if quantization == "quanto-int4":
        # Two-step: load bf16 to CPU, quantize in-place, then dispatch
        model, processor = _load_with_quanto(model_id, torch_dtype, max_memory)
    elif quantization in ("gptq", "awq", "prequantized"):
        # Load pre-quantized model — weights are already compressed on disk,
        # so this never materializes the full bf16 model in RAM.
        model, processor = _load_prequantized(
            model_id, torch_dtype, max_memory, offload_folder,
        )
    elif quantization == "bnb-nf4":
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            max_memory=max_memory,
            dtype=torch_dtype,
            low_cpu_mem_usage=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            ),
        )
        processor = AutoProcessor.from_pretrained(model_id)
    else:
        # No quantization — load to CPU first, then dispatch across devices.
        # device_map="auto" doesn't always work for Gemma4ForConditionalGeneration,
        # so we load to CPU and manually dispatch like the quanto path.
        print("  Step 1/2: Loading bf16 weights to CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch_dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(model_id)

        print("  Step 2/2: Dispatching across CPU and GPU...")
        from accelerate import infer_auto_device_map, dispatch_model
        no_split = (
            model._no_split_modules
            if hasattr(model, "_no_split_modules")
            else []
        )
        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=no_split,
        )
        if offload_folder:
            Path(offload_folder).mkdir(parents=True, exist_ok=True)
        model = dispatch_model(
            model, device_map=device_map,
            offload_dir=offload_folder,
        )

    load_time = time.perf_counter() - t0

    # Gather device placement info
    device_stats = _analyze_device_placement(model)

    metadata = {
        "model_id": model_id,
        "load_time_s": round(load_time, 1),
        "quantization": quantization,
        **device_stats,
    }

    if torch.cuda.is_available():
        metadata["gpu_allocated_mb"] = round(
            torch.cuda.memory_allocated() / 1e6, 1
        )
        metadata["gpu_reserved_mb"] = round(
            torch.cuda.memory_reserved() / 1e6, 1
        )

    print(f"  Loaded in {load_time:.1f}s")
    print(f"  GPU allocated: {metadata.get('gpu_allocated_mb', 0):.0f} MB")
    print(f"  Layers on GPU: {device_stats.get('layers_on_gpu', '?')}")
    print(f"  Layers on CPU: {device_stats.get('layers_on_cpu', '?')}")

    return model, processor, metadata


def _analyze_device_placement(model: Any) -> dict[str, Any]:
    """Analyze which layers ended up on which device."""
    gpu_params = 0
    cpu_params = 0
    gpu_bytes = 0
    cpu_bytes = 0
    layers_on_gpu = 0
    layers_on_cpu = 0

    try:
        # Walk the model's hf_device_map if available
        device_map = getattr(model, "hf_device_map", {})
        for module_name, device in device_map.items():
            if "layer" in module_name.lower():
                if str(device).startswith("cuda") or device == 0:
                    layers_on_gpu += 1
                else:
                    layers_on_cpu += 1
    except Exception:
        pass

    for name, param in model.named_parameters():
        n = param.numel()
        b = param.nbytes
        if param.device.type == "cuda":
            gpu_params += n
            gpu_bytes += b
        else:
            cpu_params += n
            cpu_bytes += b

    return {
        "gpu_params_m": round(gpu_params / 1e6, 1),
        "cpu_params_m": round(cpu_params / 1e6, 1),
        "gpu_weight_mb": round(gpu_bytes / 1e6, 1),
        "cpu_weight_mb": round(cpu_bytes / 1e6, 1),
        "layers_on_gpu": layers_on_gpu,
        "layers_on_cpu": layers_on_cpu,
    }


def inspect_model_architecture(model: Any) -> dict[str, Any]:
    """Extract architectural details relevant to our optimizations.

    Identifies:
      - Global vs sliding attention layers
      - MoE expert structure (number, sizes)
      - KV head configuration per layer type
      - RoPE configuration per layer type
    """
    config = model.config
    text_config = getattr(config, "text_config", config)

    layer_types = getattr(text_config, "layer_types", [])
    global_indices = [i for i, lt in enumerate(layer_types) if lt == "full_attention"]
    sliding_indices = [i for i, lt in enumerate(layer_types) if lt == "sliding_attention"]

    return {
        "num_layers": text_config.num_hidden_layers,
        "hidden_size": text_config.hidden_size,
        "num_attention_heads": text_config.num_attention_heads,
        "num_key_value_heads": text_config.num_key_value_heads,
        "num_global_key_value_heads": getattr(text_config, "num_global_key_value_heads", None),
        "head_dim": text_config.head_dim,
        "global_head_dim": getattr(text_config, "global_head_dim", None),
        "attention_k_eq_v": getattr(text_config, "attention_k_eq_v", False),
        "num_experts": getattr(text_config, "num_experts", 0),
        "top_k_experts": getattr(text_config, "top_k_experts", 0),
        "moe_intermediate_size": getattr(text_config, "moe_intermediate_size", 0),
        "sliding_window": text_config.sliding_window,
        "max_position_embeddings": text_config.max_position_embeddings,
        "global_layer_indices": global_indices,
        "sliding_layer_indices": sliding_indices,
        "rope_config": {
            "full_attention": getattr(text_config, "rope_parameters", {}).get("full_attention", {}),
            "sliding_attention": getattr(text_config, "rope_parameters", {}).get("sliding_attention", {}),
        },
        "vocab_size": text_config.vocab_size,
    }


def estimate_kv_cache_memory(
    arch_info: dict[str, Any],
    seq_len: int,
    dtype_bytes: int = 2,  # BF16
    turboquant_key_bits: int | None = None,
    turboquant_value_bits: int | None = None,
) -> dict[str, float]:
    """Estimate KV cache memory at a given sequence length.

    Takes into account:
      - Sliding layers only cache `sliding_window` tokens
      - Global layers cache the full sequence
      - k_eq_v halves global cache if keys == values
      - TurboQuant compression on global layers
    """
    n_global = len(arch_info["global_layer_indices"])
    n_sliding = len(arch_info["sliding_layer_indices"])
    sw = arch_info["sliding_window"]

    # Sliding layers: limited to window size
    sliding_kv_heads = arch_info["num_key_value_heads"]
    sliding_head_dim = arch_info["head_dim"]
    sliding_tokens = min(seq_len, sw)
    # K + V per layer
    sliding_per_layer = 2 * sliding_kv_heads * sliding_head_dim * sliding_tokens * dtype_bytes
    sliding_total = sliding_per_layer * n_sliding

    # Global layers: full sequence
    global_kv_heads = arch_info.get("num_global_key_value_heads") or arch_info["num_key_value_heads"]
    global_head_dim = arch_info.get("global_head_dim") or arch_info["head_dim"]
    k_eq_v = arch_info.get("attention_k_eq_v", False)
    kv_multiplier = 1 if k_eq_v else 2  # If K==V, store only once

    global_per_layer_bf16 = kv_multiplier * global_kv_heads * global_head_dim * seq_len * dtype_bytes
    global_total_bf16 = global_per_layer_bf16 * n_global

    # With TurboQuant compression
    if turboquant_key_bits and turboquant_value_bits:
        if k_eq_v:
            # Only one tensor to compress; use key_bits
            effective_bits = turboquant_key_bits
        else:
            effective_bits = (turboquant_key_bits + turboquant_value_bits) / 2
        compression_ratio = (dtype_bytes * 8) / effective_bits
        global_total_compressed = global_total_bf16 / compression_ratio
    else:
        global_total_compressed = global_total_bf16
        compression_ratio = 1.0

    return {
        "sliding_cache_mb": sliding_total / 1e6,
        "global_cache_bf16_mb": global_total_bf16 / 1e6,
        "global_cache_compressed_mb": global_total_compressed / 1e6,
        "total_bf16_mb": (sliding_total + global_total_bf16) / 1e6,
        "total_compressed_mb": (sliding_total + global_total_compressed) / 1e6,
        "compression_ratio": compression_ratio,
        "seq_len": seq_len,
    }
