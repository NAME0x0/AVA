#!/usr/bin/env python3
"""
Enhanced Model Quantization Script for AVA
Optimized for NVIDIA RTX A2000 4GB VRAM - Production-Ready Implementation
"""

import argparse
import json
import logging
import os
import gc
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import warnings

import torch
import psutil
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class QuantizationStrategy(Enum):
    """Different quantization strategies for model compression."""
    BITSANDBYTES_4BIT = "bnb_4bit"
    BITSANDBYTES_8BIT = "bnb_8bit"
    DYNAMIC_4BIT = "dynamic_4bit"


class QuantizationType(Enum):
    """Quantization data types."""
    NF4 = "nf4"  # Normal Float 4-bit
    FP4 = "fp4"  # Float Point 4-bit
    INT8 = "int8"
    INT4 = "int4"


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    strategy: QuantizationStrategy = QuantizationStrategy.BITSANDBYTES_4BIT
    quantization_type: QuantizationType = QuantizationType.NF4
    compute_dtype: torch.dtype = torch.bfloat16
    double_quantization: bool = True
    use_nested_quantization: bool = False
    bnb_4bit_use_double_quant: bool = True
    max_memory_gb: float = 3.5  # Leave buffer for RTX A2000 4GB
    device_map: str = "auto"
    trust_remote_code: bool = False
    torch_dtype: torch.dtype = torch.float16
    low_cpu_mem_usage: bool = True
    
    # Performance optimization
    use_flash_attention: bool = False
    gradient_checkpointing: bool = False
    
    # Validation settings
    test_generation: bool = True
    test_prompt: str = "The future of AI is"
    max_new_tokens: int = 50


@dataclass
class ModelInfo:
    """Information about the model being quantized."""
    model_id: str
    model_size_mb: float = 0.0
    param_count: int = 0
    architecture: str = ""
    vocab_size: int = 0
    hidden_size: int = 0
    num_layers: int = 0
    quantized_size_mb: float = 0.0
    compression_ratio: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelQuantizer:
    """Enhanced model quantizer with VRAM optimization and validation."""
    
    def __init__(self, config: Optional[QuantizationConfig] = None):
        """Initialize the model quantizer."""
        self.config = config or QuantizationConfig()
        self.original_model = None
        self.quantized_model = None
        self.tokenizer = None
        self.model_info = None
        
        # Setup device and memory management
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_memory_management()
        
        logger.info(f"ModelQuantizer initialized for device: {self.device}")
        logger.info(f"Available CUDA memory: {self._get_cuda_memory_info()}")
    
    def _setup_memory_management(self):
        """Setup memory management for optimal VRAM usage."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Set memory fraction to leave buffer
            torch.cuda.set_per_process_memory_fraction(0.9)
    
    def _get_cuda_memory_info(self) -> Dict[str, float]:
        """Get CUDA memory information."""
        if not torch.cuda.is_available():
            return {"total": 0.0, "available": 0.0, "used": 0.0}
        
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        allocated = torch.cuda.memory_allocated(0) / 1e9
        
        return {
            "total_gb": total,
            "reserved_gb": reserved,
            "allocated_gb": allocated,
            "available_gb": total - reserved
        }
    
    def _get_model_info(self, model_id: str) -> ModelInfo:
        """Extract information about the model."""
        try:
            # Load model config to get architecture info
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=self.config.trust_remote_code)
            
            model_info = ModelInfo(
                model_id=model_id,
                architecture=config.architectures[0] if hasattr(config, 'architectures') and config.architectures else "unknown",
                vocab_size=getattr(config, 'vocab_size', 0),
                hidden_size=getattr(config, 'hidden_size', 0),
                num_layers=getattr(config, 'num_hidden_layers', 0),
            )
            
            # Estimate parameter count
            if hasattr(config, 'num_parameters'):
                model_info.param_count = config.num_parameters
            else:
                # Rough estimation for transformer models
                model_info.param_count = model_info.vocab_size * model_info.hidden_size * 2 + model_info.num_layers * model_info.hidden_size * model_info.hidden_size * 4
            
            # Estimate model size (parameters * 2 bytes for fp16)
            model_info.model_size_mb = (model_info.param_count * 2) / (1024 * 1024)
            
            logger.info(f"Model Info - Architecture: {model_info.architecture}, "
                       f"Parameters: {model_info.param_count:,}, "
                       f"Estimated Size: {model_info.model_size_mb:.1f}MB")
            
            return model_info
            
        except Exception as e:
            logger.warning(f"Could not extract full model info: {e}")
            return ModelInfo(model_id=model_id)
    
    def _create_quantization_config(self) -> BitsAndBytesConfig:
        """Create BitsAndBytesConfig based on strategy."""
        if self.config.strategy == QuantizationStrategy.BITSANDBYTES_4BIT:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=self.config.quantization_type.value,
                bnb_4bit_compute_dtype=self.config.compute_dtype,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        elif self.config.strategy == QuantizationStrategy.BITSANDBYTES_8BIT:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        else:
            raise ValueError(f"Unsupported quantization strategy: {self.config.strategy}")
    
    def quantize_model(
        self, 
        model_id: str, 
        output_path: str,
        save_tokenizer: bool = True,
        validate_output: bool = True
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Quantize a model with comprehensive error handling and validation.
        
        Args:
            model_id: HuggingFace model ID or local path
            output_path: Path to save quantized model
            save_tokenizer: Whether to save tokenizer alongside model
            validate_output: Whether to validate the quantized model
            
        Returns:
            Tuple of (success, results_dict)
        """
        start_time = time.time()
        results = {"success": False, "error": None, "metrics": {}}
        
        try:
            logger.info(f"Starting quantization of {model_id}")
            
            # Get model information
            self.model_info = self._get_model_info(model_id)
            
            # Create output directory
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Clear VRAM before loading
            torch.cuda.empty_cache()
            gc.collect()
            
            # Load tokenizer first
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=self.config.trust_remote_code
            )
            
            # Create quantization config
            quantization_config = self._create_quantization_config()
            
            # Load and quantize model
            logger.info(f"Loading model with {self.config.strategy.value} quantization...")
            memory_before = self._get_cuda_memory_info()
            
            self.quantized_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map=self.config.device_map,
                torch_dtype=self.config.torch_dtype,
                trust_remote_code=self.config.trust_remote_code,
                low_cpu_mem_usage=self.config.low_cpu_mem_usage
            )
            
            memory_after = self._get_cuda_memory_info()
            actual_vram_usage = memory_after["allocated_gb"] - memory_before["allocated_gb"]
            
            logger.info(f"Model loaded successfully. VRAM usage: {actual_vram_usage:.2f}GB")
            
            # Validate model if requested
            if validate_output and self.config.test_generation:
                validation_success = self._validate_quantized_model()
                if not validation_success:
                    logger.warning("Model validation failed, but continuing with save...")
            
            # Save quantized model
            logger.info(f"Saving quantized model to {output_path}")
            self.quantized_model.save_pretrained(
                output_path,
                safe_serialization=True,
                max_shard_size="2GB"
            )
            
            # Save tokenizer if requested
            if save_tokenizer:
                self.tokenizer.save_pretrained(output_path)
            
            # Save quantization metadata
            self._save_quantization_metadata(output_path, actual_vram_usage)
            
            # Calculate metrics
            quantized_size = self._get_directory_size(output_path)
            compression_ratio = self.model_info.model_size_mb / (quantized_size / (1024 * 1024)) if quantized_size > 0 else 0
            
            results.update({
                "success": True,
                "metrics": {
                    "original_size_mb": self.model_info.model_size_mb,
                    "quantized_size_mb": quantized_size / (1024 * 1024),
                    "compression_ratio": compression_ratio,
                    "vram_usage_gb": actual_vram_usage,
                    "quantization_time_s": time.time() - start_time,
                    "strategy": self.config.strategy.value,
                    "quantization_type": self.config.quantization_type.value
                }
            })
            
            logger.info(f"Quantization completed successfully in {time.time() - start_time:.2f}s")
            logger.info(f"Compression ratio: {compression_ratio:.2f}x")
            
            return True, results
            
        except Exception as e:
            logger.error(f"Quantization failed: {str(e)}")
            results["error"] = str(e)
            return False, results
        
        finally:
            # Cleanup
            self._cleanup_memory()
    
    def _validate_quantized_model(self) -> bool:
        """Validate the quantized model by running a test generation."""
        try:
            logger.info("Validating quantized model...")
            
            inputs = self.tokenizer(
                self.config.test_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.quantized_model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Generated text: {generated_text}")
            
            return len(generated_text) > len(self.config.test_prompt)
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
    
    def _save_quantization_metadata(self, output_path: Path, vram_usage: float):
        """Save quantization metadata for future reference."""
        metadata = {
            "model_info": {
                "model_id": self.model_info.model_id,
                "architecture": self.model_info.architecture,
                "param_count": self.model_info.param_count,
                "original_size_mb": self.model_info.model_size_mb
            },
            "quantization_config": {
                "strategy": self.config.strategy.value,
                "quantization_type": self.config.quantization_type.value,
                "compute_dtype": str(self.config.compute_dtype),
                "double_quantization": self.config.double_quantization
            },
            "performance": {
                "vram_usage_gb": vram_usage,
                "target_hardware": "NVIDIA RTX A2000 4GB"
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        metadata_path = output_path / "quantization_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Quantization metadata saved to {metadata_path}")
    
    def _get_directory_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        return total_size
    
    def _cleanup_memory(self):
        """Clean up memory after quantization."""
        if hasattr(self, 'quantized_model') and self.quantized_model is not None:
            del self.quantized_model
        if hasattr(self, 'original_model') and self.original_model is not None:
            del self.original_model
        
        torch.cuda.empty_cache()
        gc.collect()


def test_quantization_pipeline():
    """Test the quantization pipeline with a small model."""
    logger.info("=== Testing Quantization Pipeline ===")
    
    # Use a small model for testing
    test_model_id = "microsoft/DialoGPT-small"
    output_path = "./models/test_quantized"
    
    config = QuantizationConfig(
        strategy=QuantizationStrategy.BITSANDBYTES_4BIT,
        quantization_type=QuantizationType.NF4,
        test_generation=True,
        test_prompt="Hello, I am"
    )
    
    quantizer = ModelQuantizer(config)
    success, results = quantizer.quantize_model(test_model_id, output_path)
    
    if success:
        logger.info("Test quantization successful!")
        logger.info(f"Results: {results}")
    else:
        logger.error(f"Test quantization failed: {results.get('error')}")
    
    return success


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Enhanced Model Quantization for AVA - RTX A2000 4GB Optimized",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/quantize_model.py --model_id google/gemma-7b --output_path ./models/quantized/gemma-7b-4bit
  python scripts/quantize_model.py --model_id microsoft/DialoGPT-medium --output_path ./models/quantized/dialogpt-4bit --strategy bnb_4bit
  python scripts/quantize_model.py --test  # Run test with small model
        """
    )
    
    parser.add_argument("--model_id", type=str, help="HuggingFace model ID or local path")
    parser.add_argument("--output_path", type=str, help="Path to save quantized model")
    parser.add_argument("--strategy", type=str, choices=[s.value for s in QuantizationStrategy], 
                       default="bnb_4bit", help="Quantization strategy")
    parser.add_argument("--quantization_type", type=str, choices=[q.value for q in QuantizationType], 
                       default="nf4", help="Quantization data type")
    parser.add_argument("--max_memory_gb", type=float, default=3.5, 
                       help="Maximum memory usage in GB (default: 3.5 for RTX A2000)")
    parser.add_argument("--test_generation", action="store_true", 
                       help="Test generation after quantization")
    parser.add_argument("--test", action="store_true", 
                       help="Run test quantization with small model")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.test:
        return test_quantization_pipeline()
    
    if not args.model_id or not args.output_path:
        parser.error("--model_id and --output_path are required unless using --test")
    
    # Create configuration
    config = QuantizationConfig(
        strategy=QuantizationStrategy(args.strategy),
        quantization_type=QuantizationType(args.quantization_type),
        max_memory_gb=args.max_memory_gb,
        test_generation=args.test_generation
    )
    
    # Run quantization
    quantizer = ModelQuantizer(config)
    success, results = quantizer.quantize_model(args.model_id, args.output_path)
    
    if success:
        logger.info("Quantization completed successfully!")
        print(f"\nResults: {json.dumps(results['metrics'], indent=2)}")
    else:
        logger.error(f"Quantization failed: {results.get('error')}")
        return False
    
    return True


if __name__ == "__main__":
    main() 
