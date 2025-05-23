#!/usr/bin/env python3
"""
Enhanced Text Generation Module for AVA
Production-Ready Text Generation with Multiple Backends and Advanced Features
"""

import asyncio
import aiohttp
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import warnings

# Optional imports with fallbacks
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    warnings.warn("transformers library not available - Ollama backend only")

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    warnings.warn("ollama library not available - transformers backend only")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenerationBackend(Enum):
    """Text generation backend options."""
    OLLAMA = "ollama"
    TRANSFORMERS = "transformers"
    AUTO = "auto"  # Automatically select available backend


class GenerationStrategy(Enum):
    """Text generation strategies."""
    GREEDY = "greedy"
    SAMPLING = "sampling"
    BEAM_SEARCH = "beam_search"
    NUCLEUS = "nucleus"  # Top-p sampling
    TOP_K = "top_k"


class OutputFormat(Enum):
    """Output format options."""
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"
    STRUCTURED = "structured"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    num_beams: int = 1
    do_sample: bool = True
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    early_stopping: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    strategy: GenerationStrategy = GenerationStrategy.SAMPLING
    seed: Optional[int] = None


@dataclass
class GenerationRequest:
    """Request for text generation."""
    prompt: str
    config: GenerationConfig = field(default_factory=GenerationConfig)
    output_format: OutputFormat = OutputFormat.TEXT
    system_prompt: Optional[str] = None
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Result of text generation."""
    success: bool
    text: str = ""
    tokens_generated: int = 0
    generation_time_ms: float = 0.0
    backend_used: Optional[GenerationBackend] = None
    model_used: str = ""
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTextGenerator(ABC):
    """Abstract base class for text generators."""
    
    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate text based on the request."""
        pass
    
    @abstractmethod
    async def stream_generate(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Stream text generation token by token."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the generator is available and ready."""
        pass
    
    @abstractmethod
    async def close(self):
        """Clean up resources."""
        pass


class OllamaTextGenerator(BaseTextGenerator):
    """Text generator using Ollama backend."""
    
    def __init__(self, model_name: str = "ava-agent", host: str = "localhost", port: int = 11434):
        self.model_name = model_name
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate text using Ollama."""
        start_time = time.time()
        
        try:
            if not await self.is_available():
                return GenerationResult(
                    success=False,
                    error="Ollama service not available",
                    backend_used=GenerationBackend.OLLAMA
                )
            
            # Prepare the prompt
            full_prompt = self._build_prompt(request)
            
            # Prepare request data
            data = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": request.config.temperature,
                    "top_p": request.config.top_p,
                    "top_k": request.config.top_k,
                    "repeat_penalty": request.config.repetition_penalty,
                    "num_predict": request.config.max_length,
                    "seed": request.config.seed or -1
                }
            }
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    generated_text = result.get("response", "")
                    
                    # Format output if needed
                    formatted_text = self._format_output(generated_text, request.output_format)
                    
                    generation_time = (time.time() - start_time) * 1000
                    
                    return GenerationResult(
                        success=True,
                        text=formatted_text,
                        tokens_generated=len(generated_text.split()),
                        generation_time_ms=generation_time,
                        backend_used=GenerationBackend.OLLAMA,
                        model_used=self.model_name,
                        metadata={
                            "eval_count": result.get("eval_count", 0),
                            "eval_duration": result.get("eval_duration", 0),
                            "prompt_eval_count": result.get("prompt_eval_count", 0)
                        }
                    )
                else:
                    error_text = await response.text()
                    return GenerationResult(
                        success=False,
                        error=f"Ollama API error {response.status}: {error_text}",
                        backend_used=GenerationBackend.OLLAMA,
                        generation_time_ms=(time.time() - start_time) * 1000
                    )
                    
        except Exception as e:
            logger.error(f"Ollama generation error: {str(e)}")
            return GenerationResult(
                success=False,
                error=f"Ollama generation failed: {str(e)}",
                backend_used=GenerationBackend.OLLAMA,
                generation_time_ms=(time.time() - start_time) * 1000
            )
    
    async def stream_generate(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Stream text generation from Ollama."""
        try:
            if not await self.is_available():
                yield "Error: Ollama service not available"
                return
            
            full_prompt = self._build_prompt(request)
            
            data = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": True,
                "options": {
                    "temperature": request.config.temperature,
                    "top_p": request.config.top_p,
                    "top_k": request.config.top_k,
                    "repeat_penalty": request.config.repetition_penalty,
                    "num_predict": request.config.max_length,
                    "seed": request.config.seed or -1
                }
            }
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status == 200:
                    async for line in response.content:
                        if line:
                            try:
                                chunk = json.loads(line.decode('utf-8'))
                                if chunk.get("response"):
                                    yield chunk["response"]
                                if chunk.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue
                else:
                    yield f"Error: Ollama API returned status {response.status}"
                    
        except Exception as e:
            logger.error(f"Ollama streaming error: {str(e)}")
            yield f"Error: {str(e)}"
    
    async def is_available(self) -> bool:
        """Check if Ollama service is available."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(
                f"{self.base_url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model["name"] for model in data.get("models", [])]
                    return self.model_name in models
                return False
        except:
            return False
    
    def _build_prompt(self, request: GenerationRequest) -> str:
        """Build the complete prompt from request components."""
        parts = []
        
        if request.system_prompt:
            parts.append(f"System: {request.system_prompt}")
        
        if request.context:
            parts.append(f"Context: {request.context}")
        
        parts.append(f"User: {request.prompt}")
        parts.append("Assistant:")
        
        return "\n\n".join(parts)
    
    def _format_output(self, text: str, format_type: OutputFormat) -> str:
        """Format the output according to the specified format."""
        if format_type == OutputFormat.TEXT:
            return text.strip()
        elif format_type == OutputFormat.JSON:
            try:
                # Try to extract JSON from the text
                import re
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    return json_match.group(0)
                else:
                    return json.dumps({"response": text.strip()})
            except:
                return json.dumps({"response": text.strip()})
        elif format_type == OutputFormat.MARKDOWN:
            return f"```\n{text.strip()}\n```"
        else:
            return text.strip()
    
    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None


class TransformersTextGenerator(BaseTextGenerator):
    """Text generator using transformers backend."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._initialized = False
        
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate text using transformers."""
        if not HAS_TRANSFORMERS:
            return GenerationResult(
                success=False,
                error="transformers library not available",
                backend_used=GenerationBackend.TRANSFORMERS
            )
        
        start_time = time.time()
        
        try:
            if not self._initialized:
                await self._initialize()
            
            full_prompt = self._build_prompt(request)
            
            # Generate text
            outputs = self.pipeline(
                full_prompt,
                max_length=request.config.max_length,
                temperature=request.config.temperature,
                top_p=request.config.top_p,
                top_k=request.config.top_k,
                do_sample=request.config.do_sample,
                num_beams=request.config.num_beams,
                repetition_penalty=request.config.repetition_penalty,
                length_penalty=request.config.length_penalty,
                early_stopping=request.config.early_stopping,
                pad_token_id=request.config.pad_token_id,
                eos_token_id=request.config.eos_token_id,
                return_full_text=False
            )
            
            generated_text = outputs[0]["generated_text"]
            formatted_text = self._format_output(generated_text, request.output_format)
            
            generation_time = (time.time() - start_time) * 1000
            
            return GenerationResult(
                success=True,
                text=formatted_text,
                tokens_generated=len(generated_text.split()),
                generation_time_ms=generation_time,
                backend_used=GenerationBackend.TRANSFORMERS,
                model_used=self.model_name
            )
            
        except Exception as e:
            logger.error(f"Transformers generation error: {str(e)}")
            return GenerationResult(
                success=False,
                error=f"Transformers generation failed: {str(e)}",
                backend_used=GenerationBackend.TRANSFORMERS,
                generation_time_ms=(time.time() - start_time) * 1000
            )
    
    async def stream_generate(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Stream text generation from transformers (simulated)."""
        try:
            result = await self.generate(request)
            if result.success:
                # Simulate streaming by yielding words
                words = result.text.split()
                for word in words:
                    yield word + " "
                    await asyncio.sleep(0.05)  # Simulate delay
            else:
                yield f"Error: {result.error}"
        except Exception as e:
            yield f"Error: {str(e)}"
    
    async def is_available(self) -> bool:
        """Check if transformers backend is available."""
        return HAS_TRANSFORMERS
    
    async def _initialize(self):
        """Initialize the model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Set pad token if not available
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Determine device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device if self.device == "cuda" else None
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            self._initialized = True
            logger.info(f"Initialized transformers backend with {self.model_name} on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize transformers backend: {str(e)}")
            raise
    
    def _build_prompt(self, request: GenerationRequest) -> str:
        """Build the complete prompt from request components."""
        parts = []
        
        if request.system_prompt:
            parts.append(request.system_prompt)
        
        if request.context:
            parts.append(request.context)
        
        parts.append(request.prompt)
        
        return "\n".join(parts)
    
    def _format_output(self, text: str, format_type: OutputFormat) -> str:
        """Format the output according to the specified format."""
        if format_type == OutputFormat.TEXT:
            return text.strip()
        elif format_type == OutputFormat.JSON:
            try:
                import re
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    return json_match.group(0)
                else:
                    return json.dumps({"response": text.strip()})
            except:
                return json.dumps({"response": text.strip()})
        elif format_type == OutputFormat.MARKDOWN:
            return f"```\n{text.strip()}\n```"
        else:
            return text.strip()
    
    async def close(self):
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class EnhancedTextGenerator:
    """Enhanced text generator with multiple backend support."""
    
    def __init__(
        self,
        backend: GenerationBackend = GenerationBackend.AUTO,
        ollama_model: str = "ava-agent",
        transformers_model: str = "microsoft/DialoGPT-medium",
        ollama_host: str = "localhost",
        ollama_port: int = 11434
    ):
        self.backend_type = backend
        self.generators: Dict[GenerationBackend, BaseTextGenerator] = {}
        self.active_generator: Optional[BaseTextGenerator] = None
        
        # Initialize generators based on backend choice
        if backend in [GenerationBackend.OLLAMA, GenerationBackend.AUTO] and HAS_OLLAMA:
            self.generators[GenerationBackend.OLLAMA] = OllamaTextGenerator(
                ollama_model, ollama_host, ollama_port
            )
        
        if backend in [GenerationBackend.TRANSFORMERS, GenerationBackend.AUTO] and HAS_TRANSFORMERS:
            self.generators[GenerationBackend.TRANSFORMERS] = TransformersTextGenerator(
                transformers_model
            )
        
        if not self.generators:
            raise RuntimeError("No text generation backends available")
    
    async def initialize(self):
        """Initialize the text generator."""
        if self.backend_type == GenerationBackend.AUTO:
            # Auto-select the best available backend
            for backend_type in [GenerationBackend.OLLAMA, GenerationBackend.TRANSFORMERS]:
                if backend_type in self.generators:
                    generator = self.generators[backend_type]
                    if await generator.is_available():
                        self.active_generator = generator
                        logger.info(f"Auto-selected {backend_type.value} backend")
                        break
        else:
            if self.backend_type in self.generators:
                self.active_generator = self.generators[self.backend_type]
            else:
                raise RuntimeError(f"Backend {self.backend_type.value} not available")
        
        if not self.active_generator:
            raise RuntimeError("No available text generation backend")
    
    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        output_format: OutputFormat = OutputFormat.TEXT,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate text using the active backend.
        
        Args:
            prompt: The input prompt
            config: Generation configuration
            output_format: Desired output format
            system_prompt: Optional system prompt
            context: Optional context information
            **kwargs: Additional parameters
            
        Returns:
            GenerationResult with the generated text or error information
        """
        if not self.active_generator:
            await self.initialize()
        
        if config is None:
            config = GenerationConfig()
        
        # Override config with kwargs if provided
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        request = GenerationRequest(
            prompt=prompt,
            config=config,
            output_format=output_format,
            system_prompt=system_prompt,
            context=context
        )
        
        return await self.active_generator.generate(request)
    
    async def stream_generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        output_format: OutputFormat = OutputFormat.TEXT,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream text generation using the active backend.
        
        Args:
            prompt: The input prompt
            config: Generation configuration
            output_format: Desired output format
            system_prompt: Optional system prompt
            context: Optional context information
            **kwargs: Additional parameters
            
        Yields:
            Text tokens as they are generated
        """
        if not self.active_generator:
            await self.initialize()
        
        if config is None:
            config = GenerationConfig()
        
        # Override config with kwargs if provided
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        request = GenerationRequest(
            prompt=prompt,
            config=config,
            output_format=output_format,
            system_prompt=system_prompt,
            context=context
        )
        
        async for token in self.active_generator.stream_generate(request):
            yield token
    
    def run(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Synchronous interface for function calling compatibility.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with generation result or error information
        """
        # Extract parameters
        config = GenerationConfig()
        output_format = OutputFormat(kwargs.get('output_format', 'text'))
        system_prompt = kwargs.get('system_prompt')
        context = kwargs.get('context')
        
        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Run async generation in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                self.generate(prompt, config, output_format, system_prompt, context)
            )
        finally:
            loop.close()
        
        if result.success:
            return {
                "text": result.text,
                "tokens_generated": result.tokens_generated,
                "generation_time_ms": result.generation_time_ms,
                "backend_used": result.backend_used.value if result.backend_used else "unknown",
                "model_used": result.model_used,
                "warnings": result.warnings,
                "metadata": result.metadata
            }
        else:
            return {
                "error": result.error,
                "backend_used": result.backend_used.value if result.backend_used else "unknown",
                "generation_time_ms": result.generation_time_ms
            }
    
    async def close(self):
        """Close all generators and clean up resources."""
        for generator in self.generators.values():
            await generator.close()
        self.generators.clear()
        self.active_generator = None


def test_text_generation():
    """Test the enhanced text generation module."""
    logger.info("=== Testing Enhanced Text Generation ===")
    
    # Initialize text generator
    generator = EnhancedTextGenerator(backend=GenerationBackend.AUTO)
    
    test_prompts = [
        "Explain the concept of 4-bit quantization in neural networks.",
        "Write a short Python function to calculate factorial.",
        "What are the key advantages of local AI deployment?",
        "Describe the process of knowledge distillation in machine learning."
    ]
    
    for prompt in test_prompts:
        logger.info(f"\nTesting prompt: {prompt[:50]}...")
        result = generator.run(
            prompt,
            max_length=200,
            temperature=0.7,
            output_format="text"
        )
        
        if "error" in result:
            logger.error(f"Error: {result['error']}")
        else:
            logger.info(f"Generated {result['tokens_generated']} tokens")
            logger.info(f"Generation time: {result['generation_time_ms']:.2f}ms")
            logger.info(f"Backend: {result['backend_used']}")
            logger.info(f"Text preview: {result['text'][:100]}...")


async def main():
    """Main function for standalone testing."""
    test_text_generation()


if __name__ == "__main__":
    asyncio.run(main())
