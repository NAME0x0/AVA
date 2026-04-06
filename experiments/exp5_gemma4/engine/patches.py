"""Model patches — hooks TurboQuant and YaRN into Gemma 4 attention layers.

These patches modify the model's forward pass in-place to:
  1. Replace RoPE with YaRN-extended frequencies on global layers
  2. Attach TurboQuant references to global attention layers for
     KV cache compression (compression hooks in generate loop)

Gemma 4 multimodal model structure:
  Gemma4ForConditionalGeneration
    .model → Gemma4Model
      .language_model → Gemma4TextModel
        .rotary_emb → Gemma4TextRotaryEmbedding
          .full_attention_inv_freq     (shape: [rotary_dim/2])
          .full_attention_attention_scaling  (float)
          .sliding_attention_inv_freq  (shape: [rotary_dim/2])
          .sliding_attention_attention_scaling  (float)
        .layers → ModuleList[Gemma4TextDecoderLayer]
          .self_attn → Gemma4TextAttention
            .q_proj, .k_proj, .v_proj, .o_proj
            .q_norm, .k_norm, .v_norm
    .lm_head
"""
from __future__ import annotations

from typing import Any

import torch

from .turboquant import TurboQuantV3
from .yarn import YarnContextExtender, compute_yarn_frequencies


def patch_model_for_yarn(
    model: Any,
    yarn: YarnContextExtender,
) -> None:
    """Patch the global rotary embedding to use YaRN-extended frequencies.

    In Gemma 4, the rotary embedding is a single module at the text model
    level with separate inv_freq buffers per layer type.  We replace
    `full_attention_inv_freq` with YaRN-modified frequencies and update
    the attention scaling factor.
    """
    text_model = _get_text_model(model)
    if text_model is None:
        print("Warning: Could not find text model to patch for YaRN")
        return

    rotary_emb = getattr(text_model, "rotary_emb", None)
    if rotary_emb is None:
        print("Warning: No rotary_emb found on text model")
        return

    # Get the current full_attention inv_freq
    orig_inv_freq = getattr(rotary_emb, "full_attention_inv_freq", None)
    if orig_inv_freq is None:
        print("Warning: No full_attention_inv_freq found")
        return

    # Read config for RoPE parameters
    config = _get_text_config(model)
    rope_params = getattr(config, "rope_parameters", {})
    full_attn_config = rope_params.get("full_attention", {})

    rope_theta = full_attn_config.get("rope_theta", 1_000_000.0)
    partial_rotary_factor = full_attn_config.get("partial_rotary_factor", 0.25)
    head_dim = getattr(config, "global_head_dim", None) or config.head_dim
    original_context = getattr(config, "max_position_embeddings", 262_144)

    # Compute YaRN-modified inverse frequencies.
    # For Gemma 4's "proportional" RoPE, all head_dim/2 frequencies are used
    # (partial_rotary_factor modifies the freq values, not the count).
    # We match the original inv_freq shape by computing frequencies for the
    # full rotary dim and using partial_rotary_factor=1.0 for count.
    rotary_dim = orig_inv_freq.shape[0] * 2  # match original frequency count
    yarn_inv_freq, attention_scale = compute_yarn_frequencies(
        head_dim=rotary_dim,  # use rotary_dim directly, not head_dim
        original_max_pos=original_context,
        target_max_pos=yarn.target_context,
        rope_theta=rope_theta,
        partial_rotary_factor=1.0,  # compute all frequencies
        beta_fast=yarn.beta_fast,
        beta_slow=yarn.beta_slow,
        device=orig_inv_freq.device,
    )

    # Store original for unpatching
    if not hasattr(rotary_emb, "_original_full_attention_inv_freq"):
        rotary_emb._original_full_attention_inv_freq = orig_inv_freq.clone()
        rotary_emb._original_full_attention_attention_scaling = getattr(
            rotary_emb, "full_attention_attention_scaling", 1.0
        )

    # Replace the inv_freq buffer — must use register_buffer or direct assignment
    rotary_emb.full_attention_inv_freq = yarn_inv_freq.to(
        dtype=orig_inv_freq.dtype, device=orig_inv_freq.device
    )
    rotary_emb.full_attention_attention_scaling = attention_scale

    # Update max position embeddings in config
    config.max_position_embeddings = yarn.target_context

    print(f"YaRN: patched full_attention_inv_freq ({orig_inv_freq.shape[0]} frequencies)")
    print(f"YaRN: attention_scaling {rotary_emb._original_full_attention_attention_scaling:.4f}"
          f" -> {attention_scale:.4f}")
    print(f"YaRN: context extended from {original_context:,} to {yarn.target_context:,}")
    print(f"YaRN: rope_theta={rope_theta}, partial_rotary={partial_rotary_factor}")


def unpatch_model_yarn(model: Any) -> None:
    """Reverse YaRN patches."""
    text_model = _get_text_model(model)
    if text_model is None:
        return
    rotary_emb = getattr(text_model, "rotary_emb", None)
    if rotary_emb is None:
        return
    orig = getattr(rotary_emb, "_original_full_attention_inv_freq", None)
    if orig is not None:
        rotary_emb.full_attention_inv_freq = orig
        rotary_emb.full_attention_attention_scaling = (
            rotary_emb._original_full_attention_attention_scaling
        )
        del rotary_emb._original_full_attention_inv_freq
        del rotary_emb._original_full_attention_attention_scaling
        print("YaRN: unpatched")


def patch_model_for_turboquant(
    model: Any,
    tq: TurboQuantV3,
) -> dict[str, Any]:
    """Attach TurboQuant references to global attention layers.

    Stores TQ compressor on each global attention layer so the KV cache
    compression can be applied during the generate loop.

    Returns:
        Dict with patching metadata.
    """
    text_model = _get_text_model(model)
    if text_model is None:
        print("Warning: Could not find text model to patch for TurboQuant")
        return {}

    patched = 0
    for layer_idx in tq.global_layer_indices:
        layer = _get_layer(text_model, layer_idx)
        if layer is None:
            continue

        attn = _get_attention_module(layer)
        if attn is None:
            continue

        # Store TQ reference and layer index on the attention module
        attn._turboquant = tq
        attn._tq_layer_idx = layer_idx
        patched += 1

    config = _get_text_config(model)
    head_dim = getattr(config, "global_head_dim", None) or config.head_dim
    ratios = tq.estimate_compression_ratio(head_dim)
    print(f"TurboQuant: patched {patched}/{len(tq.global_layer_indices)} global attention modules")
    print(f"TurboQuant: K{tq.key_compressor.bits}/V{tq.value_compressor.bits}, "
          f"~{ratios['average_compression']:.1f}x average compression")

    return {"patched": patched, **ratios}


def get_model_memory_footprint(model: Any) -> dict[str, float]:
    """Detailed memory footprint breakdown."""
    result = {
        "total_params": 0,
        "total_bytes": 0,
        "gpu_bytes": 0,
        "cpu_bytes": 0,
        "by_component": {},
    }

    for name, param in model.named_parameters():
        nbytes = param.nbytes
        result["total_params"] += param.numel()
        result["total_bytes"] += nbytes

        if param.device.type == "cuda":
            result["gpu_bytes"] += nbytes
        else:
            result["cpu_bytes"] += nbytes

        # Categorize by top-level component
        component = name.split(".")[0]
        if component not in result["by_component"]:
            result["by_component"][component] = {"params": 0, "bytes": 0}
        result["by_component"][component]["params"] += param.numel()
        result["by_component"][component]["bytes"] += nbytes

    result["total_gb"] = result["total_bytes"] / 1e9
    result["gpu_gb"] = result["gpu_bytes"] / 1e9
    result["cpu_gb"] = result["cpu_bytes"] / 1e9

    return result


# --- Internal helpers to navigate Gemma 4 model structure ---

def _get_text_model(model: Any) -> Any:
    """Extract the text model with layers from Gemma4ForConditionalGeneration.

    Gemma 4 multimodal structure:
      Gemma4ForConditionalGeneration
        .model → Gemma4Model
          .language_model → Gemma4TextModel  ← has .layers and .rotary_emb
    """
    # Descend through wrapper layers until we find one with .layers
    current = model
    for attr in ["model", "language_model", "text_model"]:
        sub = getattr(current, attr, None)
        if sub is not None:
            current = sub
            if hasattr(current, "layers") and hasattr(current.layers, "__len__"):
                return current
    # Fallback: try common paths from the original model
    for path in [
        ("model", "language_model"),
        ("model", "text_model"),
        ("language_model",),
        ("text_model",),
        ("model",),
    ]:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "layers"):
            return obj
    return model


def _get_text_config(model: Any) -> Any:
    """Get the text config from the model."""
    config = model.config
    return getattr(config, "text_config", config)


def _get_layer(text_model: Any, idx: int) -> Any:
    """Get a specific transformer layer by index."""
    for attr in ["layers", "h", "blocks", "decoder_layers"]:
        layers = getattr(text_model, attr, None)
        if layers is not None and idx < len(layers):
            return layers[idx]
    return None


def _get_attention_module(layer: Any) -> Any:
    """Get the attention module from a transformer layer."""
    for attr in ["self_attn", "attn", "attention"]:
        attn = getattr(layer, attr, None)
        if attn is not None:
            return attn
    return None
