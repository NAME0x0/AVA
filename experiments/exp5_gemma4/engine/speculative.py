"""Exact speculative-decoding helpers for Gemma 4 experiments.

These utilities keep the streamed 26B runtime as the verifier and load a
smaller draft model through the existing loader. The goal is to make exact
assisted decoding easy to benchmark without disturbing the current default
generation path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .benchmark import get_model_input_device
from .loader import load_model, resolve_model_source


@dataclass(slots=True)
class AssistedDecodingConfig:
    """Configuration for exact draft/verifier decoding."""

    assistant_model_id: str = "google/gemma-4-E4B-it"
    assistant_quantization: str = "quanto-int4"
    assistant_dtype: str = "bfloat16"
    assistant_gpu_memory_gb: float = 0.0
    assistant_cpu_memory_gb: float = 10.0
    num_assistant_tokens: int = 8
    num_assistant_tokens_schedule: str = "heuristic_transient"
    assistant_confidence_threshold: float | None = 0.4
    prompt_lookup_num_tokens: int | None = None

    def to_metadata(self) -> dict[str, Any]:
        """Serialize config into benchmark-friendly metadata."""
        return asdict(self)

def configure_assistant_model(
    assistant_model: Any,
    *,
    num_assistant_tokens: int = 8,
    num_assistant_tokens_schedule: str = "heuristic_transient",
    assistant_confidence_threshold: float | None = 0.4,
    prompt_lookup_num_tokens: int | None = None,
) -> Any:
    """Set assistant generation knobs on the draft model in-place."""
    generation_config = assistant_model.generation_config
    generation_config.num_assistant_tokens = num_assistant_tokens
    generation_config.num_assistant_tokens_schedule = num_assistant_tokens_schedule
    generation_config.assistant_confidence_threshold = assistant_confidence_threshold
    generation_config.prompt_lookup_num_tokens = prompt_lookup_num_tokens
    return assistant_model


def build_assisted_generation_kwargs(
    assistant_model: Any | None,
    *,
    num_assistant_tokens: int = 8,
    num_assistant_tokens_schedule: str = "heuristic_transient",
    assistant_confidence_threshold: float | None = 0.4,
    prompt_lookup_num_tokens: int | None = None,
) -> dict[str, Any]:
    """Return generate kwargs for exact assisted decoding."""
    if assistant_model is None:
        return {}

    configure_assistant_model(
        assistant_model,
        num_assistant_tokens=num_assistant_tokens,
        num_assistant_tokens_schedule=num_assistant_tokens_schedule,
        assistant_confidence_threshold=assistant_confidence_threshold,
        prompt_lookup_num_tokens=prompt_lookup_num_tokens,
    )
    return {"assistant_model": assistant_model}


def load_assistant_model(
    config: AssistedDecodingConfig | None = None,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Load and configure the draft model used for exact assisted decoding."""
    cfg = config or AssistedDecodingConfig()
    resolved_model_source = resolve_model_source(cfg.assistant_model_id)
    assistant_model, assistant_processor, load_meta = load_model(
        model_id=resolved_model_source,
        quantization=cfg.assistant_quantization,
        gpu_memory_gb=cfg.assistant_gpu_memory_gb,
        cpu_memory_gb=cfg.assistant_cpu_memory_gb,
        dtype=cfg.assistant_dtype,
    )
    generation_kwargs = build_assisted_generation_kwargs(
        assistant_model,
        num_assistant_tokens=cfg.num_assistant_tokens,
        num_assistant_tokens_schedule=cfg.num_assistant_tokens_schedule,
        assistant_confidence_threshold=cfg.assistant_confidence_threshold,
        prompt_lookup_num_tokens=cfg.prompt_lookup_num_tokens,
    )
    load_meta = {
        **load_meta,
        "requested_model_id": cfg.assistant_model_id,
        "resolved_model_source": resolved_model_source,
        "assistant_input_device": str(get_model_input_device(assistant_model)),
        "assisted_decoding": cfg.to_metadata(),
    }
    return assistant_model, assistant_processor, load_meta, generation_kwargs
