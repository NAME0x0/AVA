"""Model loading with smart CPU/GPU memory management.

Supports several loading paths for Gemma 4 experiments, including:
  - manual bf16 CPU->GPU layer placement for dense/smaller models
  - Quanto / BitsAndBytes / GPTQ / AWQ quantized paths
  - streaming int4 load + local cached reload for 26B-scale feasibility

The loader also prepares the model for custom optimizations by exposing:
  - layer placement and memory metadata
  - architecture details used by YaRN and TurboQuant
"""
from __future__ import annotations

import importlib
import json
import time
import types
from pathlib import Path
from typing import Any

import torch


def _find_text_model(model: Any) -> Any | None:
    """Locate the Gemma text stack that owns decoder layers."""
    for prefix_parts in [("model", "language_model"), ("model",)]:
        obj = model
        for part in prefix_parts:
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "layers"):
            return obj
    return None


def get_memory_config(
    gpu_memory_gb: float = 3.5,
    cpu_memory_gb: float = 28.0,
) -> dict[str, str]:
    """Build max_memory dict for accelerate device_map."""
    mem = {"cpu": f"{cpu_memory_gb:.0f}GiB"}
    if torch.cuda.is_available():
        mem[0] = f"{gpu_memory_gb:.1f}GiB"  # accelerate expects int device keys
    return mem


def resolve_model_source(model_id: str) -> str:
    """Prefer a cached Hugging Face snapshot path over a repo ID.

    This keeps local experiments deterministic on machines where outbound HF
    requests are blocked, while still accepting plain repo IDs in callers.
    """
    local_path = Path(model_id)
    if local_path.exists():
        return str(local_path)

    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    model_root = cache_root / f"models--{model_id.replace('/', '--')}"
    snapshots_dir = model_root / "snapshots"
    if not snapshots_dir.exists():
        return model_id

    ref_main = model_root / "refs" / "main"
    if ref_main.exists():
        snapshot_name = ref_main.read_text(encoding="utf-8").strip()
        snapshot_path = snapshots_dir / snapshot_name
        if snapshot_path.exists():
            return str(snapshot_path)

    snapshots = sorted(
        (path for path in snapshots_dir.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return str(snapshots[0]) if snapshots else model_id


def _module_nbytes(module: Any) -> int:
    """Estimate module residency including parameters and buffers."""
    total = 0
    seen: set[int] = set()
    for tensor in list(module.parameters()) + list(module.buffers()):
        ident = id(tensor)
        if ident in seen:
            continue
        seen.add(ident)
        total += tensor.nbytes
    return total


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

    print("  Step 3/3: Moving layers to GPU...")
    # Parse gpu_memory_gb from max_memory dict
    gpu_gb = 3.5
    for k, v in max_memory.items():
        if k != "cpu":
            gpu_gb = float(v.replace("GiB", "").replace("GB", ""))
            break
    _move_layers_to_gpu(model, gpu_gb)

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
                      "streaming-int4" (layer-by-layer load + int4, any model size),
                      "none" (bf16, needs offload_folder for large models)

    Returns:
        model: the loaded model
        processor: tokenizer/processor
        metadata: dict with loading stats (time, memory, device_map info)
    """
    from transformers import AutoModelForCausalLM, AutoProcessor

    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    max_memory = get_memory_config(gpu_memory_gb, cpu_memory_gb)
    resolved_model_id = resolve_model_source(model_id)

    print(f"Loading {model_id}")
    if resolved_model_id != model_id:
        print(f"  Resolved local snapshot: {resolved_model_id}")
    print(f"  Quantization: {quantization}")
    print(f"  Memory budget: {json.dumps({str(k): v for k, v in max_memory.items()})}")

    t0 = time.perf_counter()

    if quantization == "streaming-int4":
        # Streaming: layer-by-layer safetensors load + int4 quantization.
        # Auto-caches quantized weights to disk for fast subsequent loads.
        from .streaming import load_quantized, save_quantized, streaming_load

        # Check for cached quantized weights
        cache_dir = Path(offload_folder or "quantized_cache") / model_id.replace("/", "--")
        if (cache_dir / "quantized_config.json").exists():
            print(f"  Found cached quantized weights at {cache_dir}")
            model, processor, stream_meta = load_quantized(
                cache_dir,
                dtype=dtype,
                gpu_memory_gb=gpu_memory_gb,
            )
        else:
            model, processor, stream_meta = streaming_load(
                model_id=resolved_model_id,
                gpu_memory_gb=gpu_memory_gb,
                cpu_memory_gb=cpu_memory_gb,
                group_size=128,
                dtype=dtype,
            )
            # Cache for next time
            save_quantized(model, cache_dir, stream_meta, processor=processor)

        load_time = time.perf_counter() - t0
        metadata = {
            "model_id": model_id,
            "load_time_s": round(load_time, 1),
            "quantization": quantization,
            **stream_meta,
            **_analyze_device_placement(model),
        }
        if torch.cuda.is_available():
            metadata["gpu_allocated_mb"] = round(torch.cuda.memory_allocated() / 1e6, 1)
        print(f"  Loaded in {load_time:.1f}s")
        return model, processor, metadata

    elif quantization == "quanto-int4":
        # Two-step: load bf16 to CPU, quantize in-place, then dispatch
        model, processor = _load_with_quanto(resolved_model_id, torch_dtype, max_memory)
    elif quantization in ("gptq", "awq", "prequantized"):
        # Load pre-quantized model — weights are already compressed on disk,
        # so this never materializes the full bf16 model in RAM.
        model, processor = _load_prequantized(
            resolved_model_id, torch_dtype, max_memory, offload_folder,
        )
    elif quantization == "bnb-nf4":
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_id,
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
        processor = AutoProcessor.from_pretrained(resolved_model_id)
    else:
        # No quantization — load to CPU first, then manually move layers to GPU.
        # accelerate's dispatch_model uses meta tensors for CPU-offloaded params,
        # which breaks Gemma4's multimodal forward path (it accesses
        # embed_tokens.weight directly, bypassing accelerate hooks).
        # Instead, we keep all tensors real and manually place layers on GPU.
        print("  Step 1/2: Loading bf16 weights to CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_id,
            dtype=torch_dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(resolved_model_id)

        print("  Step 2/2: Moving layers to GPU...")
        _move_layers_to_gpu(model, gpu_memory_gb)

    load_time = time.perf_counter() - t0

    # Gather device placement info
    device_stats = _analyze_device_placement(model)

    metadata = {
        "model_id": model_id,
        "resolved_model_source": resolved_model_id,
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


def _move_layers_to_gpu(
    model: Any,
    gpu_memory_gb: float = 3.5,
    gpu_budget_bytes: int | None = None,
) -> None:
    """Move transformer layers to GPU without accelerate dispatch_model.

    accelerate's dispatch_model replaces CPU-mapped parameters with meta
    tensors and uses hooks to move them on-demand.  This breaks Gemma4's
    multimodal forward path which directly accesses embed_tokens.weight
    (bypassing hooks -> meta tensor error).

    Instead, we keep ALL tensors real and manually .to('cuda:0') specific
    decoder layers.  A pre-forward hook on the first GPU layer moves
    hidden_states from CPU to CUDA, and a post-forward hook on the last
    GPU layer moves outputs back to CPU for the remaining CPU layers.

    Everything except the decoder layers stays on CPU as real tensors,
    so embed_tokens.weight is always accessible.
    """
    if not torch.cuda.is_available():
        print("  No CUDA available, staying on CPU")
        return

    # Reserve headroom for KV cache, activations, and intermediate tensors
    if gpu_budget_bytes is None:
        gpu_budget_bytes = int((gpu_memory_gb - 1.5) * 1024**3)

    _remove_device_boundary_hooks(model)
    text_model = _find_text_model(model)
    if text_model is None:
        print("  Warning: Could not find text model layers, staying on CPU")
        return

    # Measure per-layer sizes and greedily assign to GPU
    layer_sizes = [_module_nbytes(layer) for layer in text_model.layers]
    gpu_used, gpu_layers = 0, 0
    for size in layer_sizes:
        if gpu_used + size <= gpu_budget_bytes:
            gpu_used += size
            gpu_layers += 1
        else:
            break

    if gpu_layers == 0:
        print("  Warning: No layers fit in GPU budget, staying on CPU")
        return

    # Move layers to GPU
    device = torch.device("cuda:0")
    for i in range(gpu_layers):
        text_model.layers[i].to(device)

    print(f"  Moved {gpu_layers}/{len(layer_sizes)} layers to GPU "
          f"({gpu_used / 1e6:.0f} MB)")
    print(f"  GPU budget: {gpu_budget_bytes / 1e6:.0f} MB, "
          f"remaining: {(gpu_budget_bytes - gpu_used) / 1e6:.0f} MB")

    # Install boundary hooks for automatic device transfers.
    # The decoder layer stack processes hidden_states sequentially:
    #   embed_tokens (CPU) -> layer_0 (GPU) -> ... -> layer_N-1 (GPU)
    #   -> layer_N (CPU) -> ... -> layer_41 (CPU) -> lm_head (CPU)
    #
    # We need:
    #   1. Pre-hook on layer 0: move inputs CPU -> CUDA
    #   2. Post-hook on last GPU layer: move outputs CUDA -> CPU
    if _install_block_forward(text_model, gpu_layers, device):
        handles = []
    else:
        handles = _install_device_boundary_hooks(text_model, gpu_layers, device)

    # Store placement info on model for _analyze_device_placement
    model._manual_gpu_layers = gpu_layers
    model._manual_total_layers = len(layer_sizes)
    model._manual_gpu_used_bytes = gpu_used
    model._manual_gpu_budget_gb = gpu_memory_gb
    model._device_boundary_hook_handles = handles


def _remove_device_boundary_hooks(model: Any) -> None:
    """Remove any previously installed CPU/GPU boundary hooks."""
    text_model = _find_text_model(model)
    if text_model is not None:
        _restore_block_forward(text_model)

    handles = getattr(model, "_device_boundary_hook_handles", [])
    for handle in handles:
        handle.remove()
    if hasattr(model, "_device_boundary_hook_handles"):
        model._device_boundary_hook_handles = []
    if text_model is not None and hasattr(text_model, "_boundary_transfer_cache"):
        text_model._boundary_transfer_cache = {}


def _restore_block_forward(text_model: Any) -> None:
    """Restore the original text-model forward if it was patched."""
    if hasattr(text_model, "_original_manual_split_forward"):
        text_model.forward = text_model._original_manual_split_forward
        delattr(text_model, "_original_manual_split_forward")
    for attr in ("_manual_split_gpu_layers", "_manual_split_device"):
        if hasattr(text_model, attr):
            delattr(text_model, attr)


def _install_block_forward(
    text_model: Any,
    gpu_layers: int,
    device: torch.device,
) -> bool:
    """Patch Gemma 4 text forward to handle the GPU block without per-layer hooks."""
    if text_model.__class__.__name__ != "Gemma4TextModel":
        return False
    if gpu_layers <= 0:
        return False

    module = importlib.import_module(text_model.__class__.__module__)
    dynamic_cache_cls = getattr(module, "DynamicCache", None)
    create_causal_mask = getattr(module, "create_causal_mask", None)
    create_sliding_window_causal_mask = getattr(module, "create_sliding_window_causal_mask", None)
    base_output_cls = getattr(module, "BaseModelOutputWithPast", None)
    if not all((dynamic_cache_cls, create_causal_mask, create_sliding_window_causal_mask, base_output_cls)):
        return False

    _restore_block_forward(text_model)
    text_model._original_manual_split_forward = text_model.forward
    text_model._manual_split_gpu_layers = gpu_layers
    text_model._manual_split_device = device

    def _move_nested(x: Any, target: torch.device) -> Any:
        if isinstance(x, torch.Tensor):
            return x.to(target, non_blocking=True) if x.device != target else x
        if isinstance(x, tuple):
            return tuple(_move_nested(v, target) for v in x)
        if isinstance(x, list):
            return [_move_nested(v, target) for v in x]
        if isinstance(x, dict):
            return {k: _move_nested(v, target) for k, v in x.items()}
        return x

    def patched_forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        per_layer_inputs: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                per_layer_inputs = self.get_per_layer_inputs(input_ids, inputs_embeds)
            per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        if use_cache and past_key_values is None:
            past_key_values = dynamic_cache_cls(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = {
            layer_type: self.rotary_emb(hidden_states, position_ids, layer_type)
            for layer_type in self.unique_layer_types
        }

        gpu_layers_local = min(self._manual_split_gpu_layers, self.config.num_hidden_layers)
        cpu_device = hidden_states.device
        gpu_device = self._manual_split_device

        gpu_position_ids = None
        gpu_mask_mapping = None
        gpu_position_embeddings = None
        gpu_per_layer_inputs = None

        if gpu_layers_local > 0:
            hidden_states = hidden_states.to(gpu_device, non_blocking=True)
            gpu_position_ids = position_ids.to(gpu_device, non_blocking=True)
            gpu_mask_mapping = _move_nested(causal_mask_mapping, gpu_device)
            gpu_position_embeddings = _move_nested(position_embeddings, gpu_device)
            if per_layer_inputs is not None:
                gpu_per_layer_inputs = per_layer_inputs[:, :, :gpu_layers_local, :].to(
                    gpu_device,
                    non_blocking=True,
                )

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if i == gpu_layers_local and hidden_states.device != cpu_device:
                hidden_states = hidden_states.to(cpu_device, non_blocking=True)

            if i < gpu_layers_local:
                per_layer_input = (
                    gpu_per_layer_inputs[:, :, i, :] if gpu_per_layer_inputs is not None else None
                )
                local_position_embeddings = gpu_position_embeddings[self.config.layer_types[i]]
                local_attention_mask = gpu_mask_mapping[self.config.layer_types[i]]
                local_position_ids = gpu_position_ids
            else:
                per_layer_input = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
                local_position_embeddings = position_embeddings[self.config.layer_types[i]]
                local_attention_mask = causal_mask_mapping[self.config.layer_types[i]]
                local_position_ids = position_ids

            hidden_states = decoder_layer(
                hidden_states,
                per_layer_input,
                position_embeddings=local_position_embeddings,
                attention_mask=local_attention_mask,
                position_ids=local_position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        norm_device = next(self.norm.parameters()).device
        if hidden_states.device != norm_device:
            hidden_states = hidden_states.to(norm_device, non_blocking=True)
        hidden_states = self.norm(hidden_states)

        return base_output_cls(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    text_model.forward = types.MethodType(patched_forward, text_model)
    print(f"  Installed block forward patch for first {gpu_layers} GPU layers")
    return True


def release_manual_gpu_residency(model: Any) -> None:
    """Move manually placed decoder layers back to CPU and clear hooks."""
    gpu_layers = getattr(model, "_manual_gpu_layers", 0)
    if not gpu_layers:
        _remove_device_boundary_hooks(model)
        return

    text_model = _find_text_model(model)
    if text_model is None:
        _remove_device_boundary_hooks(model)
        return

    _remove_device_boundary_hooks(model)
    for i in range(gpu_layers):
        text_model.layers[i].to("cpu")

    for attr in (
        "_manual_gpu_layers",
        "_manual_total_layers",
        "_manual_gpu_used_bytes",
    ):
        if hasattr(model, attr):
            delattr(model, attr)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def restore_manual_gpu_residency(
    model: Any,
    gpu_memory_gb: float | None = None,
) -> None:
    """Re-apply manual GPU placement after a model was demoted to CPU."""
    budget_gb = gpu_memory_gb
    if budget_gb is None:
        budget_gb = getattr(model, "_manual_gpu_budget_gb", 3.5)
    _move_layers_to_gpu(model, gpu_memory_gb=float(budget_gb))


def _install_device_boundary_hooks(
    text_model: Any,
    gpu_layers: int,
    device: torch.device,
) -> list[Any]:
    """Install pre/post forward hooks at GPU/CPU boundaries.

    The text model's forward loop passes CPU tensors (position_embeddings,
    attention_mask, per_layer_input) to each decoder layer from its own
    local scope.  So EVERY GPU layer needs a pre-hook to move all inputs
    to CUDA, and the last GPU layer needs a post-hook to move hidden_states
    back to CPU for the remaining CPU layers.
    """
    handles = []
    text_model._boundary_transfer_cache = {}

    def _to_device(x: Any, target: torch.device, cache: dict[int, torch.Tensor] | None = None) -> Any:
        """Recursively move tensors/tuples/dicts to target device."""
        if isinstance(x, torch.Tensor):
            if x.device == target:
                return x
            cache_key = id(x)
            if cache is not None and cache_key in cache:
                return cache[cache_key]
            moved = x.to(target, non_blocking=True)
            if cache is not None:
                cache[cache_key] = moved
            return moved
        if isinstance(x, tuple):
            return tuple(_to_device(v, target, cache) for v in x)
        if isinstance(x, list):
            return [_to_device(v, target, cache) for v in x]
        if isinstance(x, dict):
            return {k: _to_device(v, target, cache) for k, v in x.items()}
        return x

    # Pre-hook on every GPU layer: move all tensor inputs to CUDA.
    # Non-tensor args (past_key_values DynamicCache) pass through unchanged.
    def make_pre_hook(target_device, *, clear_cache: bool = False):
        def pre_hook(module, args, kwargs):
            if clear_cache:
                text_model._boundary_transfer_cache = {}
            cache = getattr(text_model, "_boundary_transfer_cache", None)
            return _to_device(args, target_device, cache), _to_device(kwargs, target_device, cache)
        return pre_hook

    for i in range(gpu_layers):
        h = text_model.layers[i].register_forward_pre_hook(
            make_pre_hook(device, clear_cache=(i == 0)), with_kwargs=True,
        )
        handles.append(h)

    # Post-hook on last GPU layer: move output back to CPU for remaining layers
    if gpu_layers < len(text_model.layers):
        cpu_device = torch.device("cpu")

        def post_hook_to_cpu(module, args, output):
            try:
                return _to_device(output, cpu_device)
            finally:
                text_model._boundary_transfer_cache = {}

        h = text_model.layers[gpu_layers - 1].register_forward_hook(post_hook_to_cpu)
        handles.append(h)

    print(f"  Installed {gpu_layers} pre-hooks (->GPU) + "
          f"{'1 post-hook (->CPU)' if gpu_layers < len(text_model.layers) else 'no post-hook (all on GPU)'}")
    return handles


def _analyze_device_placement(model: Any) -> dict[str, Any]:
    """Analyze which layers ended up on which device."""
    gpu_params = 0
    cpu_params = 0
    gpu_bytes = 0
    cpu_bytes = 0
    layers_on_gpu = 0
    layers_on_cpu = 0

    # Check manual placement first (set by _move_layers_to_gpu)
    if hasattr(model, "_manual_gpu_layers"):
        layers_on_gpu = model._manual_gpu_layers
        layers_on_cpu = model._manual_total_layers - model._manual_gpu_layers
    else:
        try:
            device_map = getattr(model, "hf_device_map", {})
            for module_name, device in device_map.items():
                if "layer" in module_name.lower():
                    if str(device).startswith("cuda") or device == 0:
                        layers_on_gpu += 1
                    else:
                        layers_on_cpu += 1
        except Exception:
            pass

    seen: set[int] = set()
    for tensor in list(model.parameters()) + list(model.buffers()):
        ident = id(tensor)
        if ident in seen:
            continue
        seen.add(ident)

        n = tensor.numel()
        b = tensor.nbytes
        if tensor.device.type == "cuda":
            gpu_params += n
            gpu_bytes += b
        elif tensor.device.type != "meta":
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
