from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class TokenizerConfig:
    kind: str = "byte"
    vocab_size: int = 260
    path: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TokenizerConfig":
        data = data or {}
        path = data.get("path")
        return cls(
            kind=str(data.get("kind", "byte")),
            vocab_size=int(data.get("vocab_size", 260)),
            path=str(path) if path else None,
        )


@dataclass(slots=True)
class ModelConfig:
    block_size: int = 256
    n_layer: int = 12
    n_head: int = 10
    n_embd: int = 640
    dropout: float = 0.0
    bias: bool = False
    architecture: str = "transformer"
    loop_repeats: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ModelConfig":
        data = data or {}
        return cls(
            block_size=int(data.get("block_size", 256)),
            n_layer=int(data.get("n_layer", 12)),
            n_head=int(data.get("n_head", 10)),
            n_embd=int(data.get("n_embd", 640)),
            dropout=float(data.get("dropout", 0.0)),
            bias=bool(data.get("bias", False)),
            architecture=str(data.get("architecture", "transformer")),
            loop_repeats=max(1, int(data.get("loop_repeats", 1))),
        )

    def effective_layers(self) -> int:
        if self.architecture == "looped":
            return self.n_layer * self.loop_repeats
        return self.n_layer

    def estimated_parameters(self, vocab_size: int) -> int:
        per_block = (12 * self.n_embd * self.n_embd) + (13 * self.n_embd)
        token_embeddings = vocab_size * self.n_embd
        positional_embeddings = self.block_size * self.n_embd
        final_norm = 2 * self.n_embd
        loop_parameters = self.loop_repeats * self.n_embd if self.architecture == "looped" else 0
        return token_embeddings + positional_embeddings + (self.n_layer * per_block) + final_norm + loop_parameters


@dataclass(slots=True)
class TrainingConfig:
    device: str = "cuda"
    dtype: str = "bfloat16"
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 20_000
    warmup_steps: int = 500
    grad_clip: float = 1.0
    loss_mode: str = "raw_lm"
    init_checkpoint: str | None = None
    trainable_patterns: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TrainingConfig":
        data = data or {}
        patterns = data.get("trainable_patterns") or []
        return cls(
            device=str(data.get("device", "cuda")),
            dtype=str(data.get("dtype", "bfloat16")),
            micro_batch_size=int(data.get("micro_batch_size", 1)),
            gradient_accumulation_steps=int(data.get("gradient_accumulation_steps", 64)),
            learning_rate=float(data.get("learning_rate", 3e-4)),
            weight_decay=float(data.get("weight_decay", 0.1)),
            max_steps=int(data.get("max_steps", 20_000)),
            warmup_steps=int(data.get("warmup_steps", 500)),
            grad_clip=float(data.get("grad_clip", 1.0)),
            loss_mode=str(data.get("loss_mode", "raw_lm")),
            init_checkpoint=str(data["init_checkpoint"]) if data.get("init_checkpoint") else None,
            trainable_patterns=tuple(str(item) for item in patterns),
        )


@dataclass(slots=True)
class MemoryConfig:
    enabled: bool = True
    max_items: int = 256
    write_surprise_threshold: float = 0.45
    top_k: int = 4

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MemoryConfig":
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", True)),
            max_items=int(data.get("max_items", 256)),
            write_surprise_threshold=float(data.get("write_surprise_threshold", 0.45)),
            top_k=int(data.get("top_k", 4)),
        )


@dataclass(slots=True)
class ToolConfig:
    calculator: bool = True
    prompt_protocol: str = "compact_tags"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ToolConfig":
        data = data or {}
        return cls(
            calculator=bool(data.get("calculator", True)),
            prompt_protocol=str(data.get("prompt_protocol", "compact_tags")),
        )


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    tokenizer: TokenizerConfig
    model: ModelConfig
    training: TrainingConfig
    memory: MemoryConfig
    tools: ToolConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            name=str(data.get("name", "ava-experiment")),
            tokenizer=TokenizerConfig.from_dict(data.get("tokenizer")),
            model=ModelConfig.from_dict(data.get("model")),
            training=TrainingConfig.from_dict(data.get("training")),
            memory=MemoryConfig.from_dict(data.get("memory")),
            tools=ToolConfig.from_dict(data.get("tools")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return ExperimentConfig.from_dict(payload)
