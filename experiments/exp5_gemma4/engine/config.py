"""Configuration for Gemma 4 experiments."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Which model to load and how."""

    model_id: str = "google/gemma-4-26B-A4B-it"
    dtype: str = "bfloat16"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    gpu_memory_gb: float = 3.5
    cpu_memory_gb: float = 28.0
    offload_folder: str | None = None


@dataclass
class OffloadConfig:
    """MoE-aware expert offloading settings."""

    enabled: bool = True
    expert_cache_size: int = 16  # max experts kept on GPU at once
    prefetch: bool = True  # prefetch next layer's experts
    pin_memory: bool = True  # use CUDA pinned memory for CPU experts
    double_buffer: bool = True  # overlap transfer with compute


@dataclass
class TurboQuantConfig:
    """TurboQuant V3 KV cache compression settings."""

    enabled: bool = True
    key_bits: int = 4
    value_bits: int = 2
    group_size: int = 128  # elements per quantization group
    protected_layers: int = 1  # first/last N global layers kept at full precision
    seed: int = 42  # for reproducible rotation matrices


@dataclass
class YarnConfig:
    """YaRN RoPE context extension settings."""

    enabled: bool = True
    target_context: int = 1_048_576  # 1M tokens
    original_context: int = 262_144  # 256K (Gemma 4 native)
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    # Only applied to the 5 global attention layers


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    offload: OffloadConfig = field(default_factory=OffloadConfig)
    turboquant: TurboQuantConfig = field(default_factory=TurboQuantConfig)
    yarn: YarnConfig = field(default_factory=YarnConfig)
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 64
