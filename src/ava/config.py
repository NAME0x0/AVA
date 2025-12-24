"""
AVA Configuration System
========================

Centralized configuration with sensible defaults.
Can be loaded from YAML or environment variables.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class OllamaConfig:
    """Ollama backend configuration."""
    host: str = "http://localhost:11434"
    timeout: int = 120

    # Model selection
    fast_model: str = "gemma3:4b"      # Quick responses
    deep_model: str = "gemma3:4b"      # Deep reasoning (use larger if available)
    embedding_model: str = "gemma3:4b"  # For embeddings

    def __post_init__(self):
        # Allow environment override
        self.host = os.environ.get("OLLAMA_HOST", self.host)


@dataclass
class EngineConfig:
    """Core engine configuration."""
    ollama: OllamaConfig = field(default_factory=OllamaConfig)

    # Routing thresholds
    cortex_surprise_threshold: float = 0.5   # When to use deep thinking
    cortex_complexity_threshold: float = 0.4

    # Keywords that trigger deep thinking
    cortex_keywords: list[str] = field(default_factory=lambda: [
        "analyze", "explain", "compare", "why", "how does",
        "what if", "debug", "optimize", "step by step", "think carefully"
    ])

    # Response settings
    max_tokens: int = 2048
    temperature: float = 0.7

    # Accuracy settings
    verify_responses: bool = True       # Double-check important responses
    use_reflection: bool = True         # Allow self-correction


@dataclass
class ToolsConfig:
    """Tool system configuration."""
    enabled: bool = True

    # Built-in tools
    enable_calculator: bool = True
    enable_web_search: bool = True
    enable_file_access: bool = True

    # MCP (Model Context Protocol) settings
    mcp_enabled: bool = True
    mcp_servers: list[dict[str, Any]] = field(default_factory=list)

    # Tool execution
    timeout_seconds: int = 30
    max_retries: int = 2


@dataclass
class MemoryConfig:
    """Memory and context configuration."""
    max_history: int = 50           # Messages to keep in memory
    context_window: int = 8192      # Token context window

    # Persistence
    persist_conversations: bool = True
    data_dir: str = "data/conversations"

    # Learning
    enable_learning: bool = True
    learning_data_dir: str = "data/learning"


@dataclass
class UIConfig:
    """UI/API server configuration."""
    host: str = "127.0.0.1"
    port: int = 8085
    enable_cors: bool = True
    enable_streaming: bool = True


@dataclass
class AVAConfig:
    """Master configuration for AVA."""
    engine: EngineConfig = field(default_factory=EngineConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    # Logging
    log_level: str = "INFO"
    log_file: str | None = None

    # Data directory
    data_dir: str = "data"

    def __post_init__(self):
        # Create data directories
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.memory.data_dir).mkdir(parents=True, exist_ok=True)
        if self.memory.enable_learning:
            Path(self.memory.learning_data_dir).mkdir(parents=True, exist_ok=True)


def load_config(path: str = None) -> AVAConfig:
    """
    Load configuration from file.

    Args:
        path: Path to YAML config file. If None, uses default config.

    Returns:
        AVAConfig instance
    """
    if path is None:
        # Check standard locations
        for candidate in ["config/ava.yaml", "ava.yaml", "config.yaml"]:
            if Path(candidate).exists():
                path = candidate
                break

    if path and Path(path).exists():
        with open(path) as f:
            data = yaml.safe_load(f)
        return _dict_to_config(data)

    return AVAConfig()


def _dict_to_config(data: dict) -> AVAConfig:
    """Convert dictionary to config dataclass."""
    config = AVAConfig()

    if "engine" in data:
        engine_data = data["engine"]
        if "ollama" in engine_data:
            config.engine.ollama = OllamaConfig(**engine_data.pop("ollama"))
        for k, v in engine_data.items():
            if hasattr(config.engine, k):
                setattr(config.engine, k, v)

    if "tools" in data:
        for k, v in data["tools"].items():
            if hasattr(config.tools, k):
                setattr(config.tools, k, v)

    if "memory" in data:
        for k, v in data["memory"].items():
            if hasattr(config.memory, k):
                setattr(config.memory, k, v)

    if "ui" in data:
        for k, v in data["ui"].items():
            if hasattr(config.ui, k):
                setattr(config.ui, k, v)

    for k in ["log_level", "log_file", "data_dir"]:
        if k in data:
            setattr(config, k, data[k])

    return config


def save_config(config: AVAConfig, path: str = "config/ava.yaml"):
    """Save configuration to file."""
    import dataclasses

    def to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
        return obj

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(to_dict(config), f, default_flow_style=False)
