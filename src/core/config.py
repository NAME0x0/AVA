#!/usr/bin/env python3
"""
AVA Core Configuration Management Module

This module provides comprehensive configuration management for AVA (Afsah's Virtual Assistant),
handling all system configurations including model settings, hardware constraints, interface
settings, and runtime parameters. Optimized for NVIDIA RTX A2000 4GB VRAM constraints.

Author: Assistant
Date: 2024
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from datetime import datetime, timedelta
import tempfile
import platform
import psutil
import torch


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ModelBackend(Enum):
    """Available model backends."""
    OLLAMA = "ollama"
    TRANSFORMERS = "transformers"
    CUSTOM = "custom"


class InterfaceMode(Enum):
    """Available interface modes."""
    CLI = "cli"
    GUI = "gui"
    API = "api"
    HYBRID = "hybrid"


class QuantizationMethod(Enum):
    """Quantization methods."""
    INT4 = "int4"
    NF4 = "nf4"
    FP4 = "fp4"
    INT8 = "int8"
    BFLOAT16 = "bfloat16"


@dataclass
class HardwareConfig:
    """Hardware configuration and constraints."""
    gpu_memory_limit_gb: float = 4.0
    gpu_device_index: int = 0
    max_cpu_cores: Optional[int] = None
    max_ram_gb: Optional[float] = None
    enable_gpu_acceleration: bool = True
    force_cpu_mode: bool = False
    mixed_precision: bool = True
    enable_memory_mapping: bool = True
    cache_size_mb: int = 512
    temp_dir: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "gemma-3n-4b"
    model_path: Optional[str] = None
    backend: ModelBackend = ModelBackend.OLLAMA
    quantization_method: QuantizationMethod = QuantizationMethod.NF4
    max_context_length: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    max_new_tokens: int = 512
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    use_cache: bool = True
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InterfaceConfig:
    """Interface configuration."""
    mode: InterfaceMode = InterfaceMode.CLI
    cli_prompt: str = "AVA> "
    gui_theme: str = "dark"
    gui_port: int = 8080
    gui_host: str = "127.0.0.1"
    api_port: int = 8000
    api_host: str = "127.0.0.1"
    enable_remote_access: bool = False
    remote_tunnel_provider: str = "localtonet"
    enable_streaming: bool = True
    stream_chunk_size: int = 64
    response_timeout_seconds: int = 30
    max_concurrent_requests: int = 5
    enable_authentication: bool = False
    auth_token: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    """Agentic behavior configuration."""
    max_function_calls: int = 10
    function_timeout_seconds: int = 30
    enable_reasoning_trace: bool = True
    reasoning_method: str = "chain_of_thought"
    enable_memory: bool = True
    memory_size_limit: int = 1000
    conversation_context_limit: int = 50
    enable_tool_use: bool = True
    tool_safety_mode: bool = True
    enable_code_execution: bool = False
    enable_web_search: bool = True
    enable_file_access: bool = True
    enable_mcp: bool = True
    mcp_timeout_seconds: int = 15
    structured_output_mode: bool = True
    validation_mode: str = "strict"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: LogLevel = LogLevel.INFO
    enable_file_logging: bool = True
    log_file_path: Optional[str] = None
    max_log_size_mb: int = 100
    log_rotation_count: int = 5
    enable_console_logging: bool = True
    enable_performance_logging: bool = True
    enable_error_tracking: bool = True
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityConfig:
    """Security configuration."""
    enable_input_validation: bool = True
    max_input_length: int = 10000
    enable_output_filtering: bool = True
    blocked_patterns: List[str] = field(default_factory=list)
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 60
    enable_secure_headers: bool = True
    enable_cors: bool = False
    allowed_origins: List[str] = field(default_factory=list)
    session_timeout_minutes: int = 60
    enable_encryption: bool = False
    encryption_key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    enable_compression: bool = True
    compression_level: int = 6
    enable_async_processing: bool = True
    max_worker_threads: int = 4
    batch_size: int = 1
    enable_profiling: bool = False
    profile_output_path: Optional[str] = None
    enable_metrics_collection: bool = True
    metrics_interval_seconds: int = 60
    enable_memory_monitoring: bool = True
    memory_threshold_mb: int = 3500  # Conservative for 4GB VRAM
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AVAConfig:
    """Main AVA configuration container."""
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    interface: InterfaceConfig = field(default_factory=InterfaceConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """Configuration manager for AVA system."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager."""
        self.logger = logging.getLogger(__name__)
        self.config_path = Path(config_path) if config_path else self._get_default_config_path()
        self.config: Optional[AVAConfig] = None
        self._runtime_overrides: Dict[str, Any] = {}
        self._validation_errors: List[str] = []
        
        # Create config directory if it doesn't exist
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.load_config()
    
    def _get_default_config_path(self) -> Path:
        """Get default configuration file path."""
        config_dir = Path.home() / ".ava"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "config.yaml"
    
    def load_config(self) -> bool:
        """Load configuration from file."""
        try:
            if self.config_path.exists():
                self.logger.info(f"Loading configuration from {self.config_path}")
                
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    if self.config_path.suffix.lower() == '.json':
                        config_data = json.load(f)
                    else:  # Default to YAML
                        config_data = yaml.safe_load(f)
                
                self.config = self._dict_to_config(config_data)
                self._validate_config()
                self.logger.info("Configuration loaded successfully")
                return True
            else:
                self.logger.info("No configuration file found, creating default")
                self.config = AVAConfig()
                self.save_config()
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            self.config = AVAConfig()
            return False
    
    def save_config(self) -> bool:
        """Save configuration to file."""
        try:
            self.config.updated_at = datetime.now().isoformat()
            config_data = asdict(self.config)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                if self.config_path.suffix.lower() == '.json':
                    json.dump(config_data, f, indent=2, default=str)
                else:  # Default to YAML
                    yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            
            self.logger.info(f"Configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
    
    def _dict_to_config(self, config_data: Dict[str, Any]) -> AVAConfig:
        """Convert dictionary to AVAConfig."""
        try:
            # Extract each section and create dataclass instances
            hardware_data = config_data.get('hardware', {})
            model_data = config_data.get('model', {})
            interface_data = config_data.get('interface', {})
            agent_data = config_data.get('agent', {})
            logging_data = config_data.get('logging', {})
            security_data = config_data.get('security', {})
            performance_data = config_data.get('performance', {})
            
            # Handle enum conversions
            if 'backend' in model_data:
                model_data['backend'] = ModelBackend(model_data['backend'])
            if 'quantization_method' in model_data:
                model_data['quantization_method'] = QuantizationMethod(model_data['quantization_method'])
            if 'mode' in interface_data:
                interface_data['mode'] = InterfaceMode(interface_data['mode'])
            if 'level' in logging_data:
                logging_data['level'] = LogLevel(logging_data['level'])
            
            return AVAConfig(
                hardware=HardwareConfig(**hardware_data),
                model=ModelConfig(**model_data),
                interface=InterfaceConfig(**interface_data),
                agent=AgentConfig(**agent_data),
                logging=LoggingConfig(**logging_data),
                security=SecurityConfig(**security_data),
                performance=PerformanceConfig(**performance_data),
                version=config_data.get('version', '1.0.0'),
                created_at=config_data.get('created_at', datetime.now().isoformat()),
                updated_at=config_data.get('updated_at', datetime.now().isoformat()),
                metadata=config_data.get('metadata', {})
            )
            
        except Exception as e:
            self.logger.error(f"Failed to convert dictionary to config: {e}")
            return AVAConfig()
    
    def _validate_config(self) -> bool:
        """Validate configuration."""
        self._validation_errors.clear()
        
        try:
            # Validate hardware constraints
            if self.config.hardware.gpu_memory_limit_gb > 4.1:
                self._validation_errors.append("GPU memory limit exceeds RTX A2000 4GB constraint")
            
            # Validate model settings
            if self.config.model.max_context_length > 8192:
                self._validation_errors.append("Context length too large for memory constraints")
            
            # Validate performance settings
            if self.config.performance.memory_threshold_mb > 3800:
                self._validation_errors.append("Memory threshold too high for 4GB VRAM")
            
            # Check for CUDA availability if GPU acceleration is enabled
            if self.config.hardware.enable_gpu_acceleration and not torch.cuda.is_available():
                self._validation_errors.append("GPU acceleration enabled but CUDA not available")
            
            # Validate file paths
            if self.config.model.model_path and not Path(self.config.model.model_path).exists():
                self._validation_errors.append(f"Model path does not exist: {self.config.model.model_path}")
            
            if self._validation_errors:
                self.logger.warning(f"Configuration validation warnings: {self._validation_errors}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def get_config(self) -> AVAConfig:
        """Get current configuration."""
        return self.config
    
    def update_config(self, **kwargs) -> bool:
        """Update configuration with new values."""
        try:
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    self.logger.info(f"Updated config.{key} = {value}")
            
            self._validate_config()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            return False
    
    def set_runtime_override(self, key: str, value: Any) -> None:
        """Set runtime configuration override."""
        self._runtime_overrides[key] = value
        self.logger.debug(f"Set runtime override: {key} = {value}")
    
    def get_runtime_override(self, key: str, default: Any = None) -> Any:
        """Get runtime configuration override."""
        return self._runtime_overrides.get(key, default)
    
    def clear_runtime_overrides(self) -> None:
        """Clear all runtime overrides."""
        self._runtime_overrides.clear()
        self.logger.debug("Cleared all runtime overrides")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for configuration validation."""
        try:
            gpu_info = {}
            if torch.cuda.is_available():
                gpu_info = {
                    'gpu_count': torch.cuda.device_count(),
                    'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                    'gpu_memory': [torch.cuda.get_device_properties(i).total_memory / 1e9 
                                  for i in range(torch.cuda.device_count())],
                    'cuda_version': torch.version.cuda
                }
            
            return {
                'platform': platform.platform(),
                'python_version': sys.version,
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / 1e9,
                'disk_space_gb': psutil.disk_usage('/').total / 1e9,
                'gpu_info': gpu_info,
                'pytorch_version': torch.__version__
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system info: {e}")
            return {}
    
    def export_config(self, export_path: Union[str, Path], format: str = 'yaml') -> bool:
        """Export configuration to file."""
        try:
            export_path = Path(export_path)
            config_data = asdict(self.config)
            
            with open(export_path, 'w', encoding='utf-8') as f:
                if format.lower() == 'json':
                    json.dump(config_data, f, indent=2, default=str)
                else:
                    yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            
            self.logger.info(f"Configuration exported to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {e}")
            return False
    
    def import_config(self, import_path: Union[str, Path]) -> bool:
        """Import configuration from file."""
        try:
            import_path = Path(import_path)
            if not import_path.exists():
                raise FileNotFoundError(f"Import file not found: {import_path}")
            
            with open(import_path, 'r', encoding='utf-8') as f:
                if import_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    config_data = yaml.safe_load(f)
            
            self.config = self._dict_to_config(config_data)
            self._validate_config()
            self.save_config()
            
            self.logger.info(f"Configuration imported from {import_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import configuration: {e}")
            return False
    
    def reset_to_defaults(self) -> bool:
        """Reset configuration to defaults."""
        try:
            self.config = AVAConfig()
            self.save_config()
            self.logger.info("Configuration reset to defaults")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reset configuration: {e}")
            return False
    
    def get_validation_errors(self) -> List[str]:
        """Get configuration validation errors."""
        return self._validation_errors.copy()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ConfigManager(config_path={self.config_path}, valid={len(self._validation_errors) == 0})"


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[Union[str, Path]] = None) -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def get_config() -> AVAConfig:
    """Get current configuration."""
    return get_config_manager().get_config()


def update_config(**kwargs) -> bool:
    """Update global configuration."""
    return get_config_manager().update_config(**kwargs)


def save_config() -> bool:
    """Save global configuration."""
    return get_config_manager().save_config()


# Testing and validation functions
def test_config_manager():
    """Test configuration manager functionality."""
    print("Testing AVA Configuration Manager...")
    
    # Test default configuration creation
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    print(f"Default config created: {config.version}")
    print(f"GPU memory limit: {config.hardware.gpu_memory_limit_gb}GB")
    print(f"Model backend: {config.model.backend.value}")
    print(f"Interface mode: {config.interface.mode.value}")
    
    # Test configuration updates
    config_manager.update_config(version="1.0.1")
    print(f"Updated version: {config_manager.get_config().version}")
    
    # Test system info
    system_info = config_manager.get_system_info()
    print(f"System info: {system_info.get('platform', 'Unknown')}")
    
    # Test validation
    validation_errors = config_manager.get_validation_errors()
    print(f"Validation errors: {len(validation_errors)}")
    
    print("Configuration manager test completed!")


if __name__ == "__main__":
    test_config_manager()
