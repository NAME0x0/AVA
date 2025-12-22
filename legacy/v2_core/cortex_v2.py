"""
CORTEX V2 - Deep Reasoning via Ollama Large Model
=================================================

This implementation provides deep reasoning using a larger Ollama model
when the Medulla determines the query is too complex.

The key insight from the blueprint:
- Cortex is for DEEP THOUGHT - complex reasoning, analysis, planning
- It should be invoked rarely (maybe 5% of interactions)
- Latency is acceptable (~5-30 seconds) because user expects "thinking"

In production with RTX A2000:
- Could use AirLLM for 70B model layer-wise inference
- For now, we use a larger Ollama model (e.g., llama3:8b, mixtral)

The Cortex receives:
1. The user query
2. The projected Medulla state (conversation context)
3. Any retrieved memories from Titans
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CortexState(Enum):
    """Operating states for the Cortex."""
    DORMANT = "dormant"          # Not active
    INITIALIZING = "initializing"
    PROCESSING = "processing"
    GENERATING = "generating"


@dataclass
class CortexConfig:
    """Configuration for the Cortex with Ollama backend."""
    
    # Ollama Configuration
    ollama_host: str = "http://localhost:11434"
    
    # Model selection - use larger model for deep reasoning
    deep_model: str = "gemma3:4b"          # Primary reasoning model
    fallback_model: str = "gemma3:4b"      # If large model unavailable
    
    # Generation settings - more tokens, lower temperature for reasoning
    max_tokens: int = 1024
    temperature: float = 0.5              # Lower = more focused reasoning
    top_p: float = 0.9
    
    # Context settings
    max_context_length: int = 4096
    
    # Timing
    timeout: float = 120.0                # Allow longer for deep thought


@dataclass
class GenerationResult:
    """Result of a Cortex generation."""
    text: str = ""
    tokens_generated: int = 0
    total_time_seconds: float = 0.0
    tokens_per_second: float = 0.0
    model_used: str = ""
    error: Optional[str] = None


class CortexV2:
    """
    The Cortex V2 - Deep Reasoning using Ollama.
    
    This is activated when:
    1. Medulla detects high surprise (novel/complex input)
    2. Query complexity exceeds Medulla's capabilities
    3. User explicitly requests deep analysis
    4. Agency determines DEEP_THOUGHT policy
    
    The Cortex is designed to:
    - Take its time (5-30 seconds acceptable)
    - Provide thorough, well-reasoned responses
    - Handle complex multi-step reasoning
    - Do analysis, planning, and code generation
    """
    
    def __init__(self, config: Optional[CortexConfig] = None):
        self.config = config or CortexConfig()
        self.state = CortexState.DORMANT
        
        # HTTP session
        self._session = None
        
        # Statistics
        self.generation_count = 0
        self.total_tokens = 0
        self.total_time = 0.0
        
        # Available models cache
        self._available_models: List[str] = []
        self._active_model: Optional[str] = None
        
        # System prompt for deep reasoning
        self.system_prompt = """You are AVA's CORTEX - the deep reasoning component.

You have been activated because the query requires careful thought. You should:
- Think through problems step by step
- Provide comprehensive, well-reasoned answers
- Be thorough but stay focused on the question
- Use your full analytical capabilities

Context from the Medulla (reflexive component) is provided below.
Build on this context to provide a deep, insightful response."""

        logger.info(f"CortexV2 initialized with model: {self.config.deep_model}")

    async def _get_session(self):
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def initialize(self) -> bool:
        """
        Initialize the Cortex - check model availability.
        
        Returns:
            True if a suitable model is available
        """
        self.state = CortexState.INITIALIZING
        
        try:
            session = await self._get_session()
            
            # Get available models
            async with session.get(f"{self.config.ollama_host}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    self._available_models = [m["name"] for m in data.get("models", [])]
                    logger.info(f"Cortex - available models: {self._available_models}")
                    
                    # Select best available model
                    if any(self.config.deep_model in m for m in self._available_models):
                        self._active_model = self.config.deep_model
                    elif any(self.config.fallback_model in m for m in self._available_models):
                        self._active_model = self.config.fallback_model
                        logger.warning(f"Using fallback model: {self._active_model}")
                    elif self._available_models:
                        # Use whatever is available
                        self._active_model = self._available_models[0]
                        logger.warning(f"Using available model: {self._active_model}")
                    else:
                        logger.error("No models available in Ollama")
                        return False
                    
                    logger.info(f"Cortex active model: {self._active_model}")
                    self.state = CortexState.DORMANT
                    return True
                    
        except Exception as e:
            logger.error(f"Cortex initialization failed: {e}")
        
        self.state = CortexState.DORMANT
        return False

    async def generate(
        self,
        query: str,
        context: str = "",
        conversation_history: Optional[List[Dict[str, str]]] = None,
        projected_state: Optional[np.ndarray] = None,
    ) -> GenerationResult:
        """
        Generate a deep reasoning response.
        
        Args:
            query: The user's query that requires deep thought
            context: Additional context from Medulla
            conversation_history: Recent conversation for context
            projected_state: Medulla state projection (for future use)
            
        Returns:
            GenerationResult with response and metrics
        """
        result = GenerationResult()
        start_time = time.time()
        
        if self._active_model is None:
            await self.initialize()
            if self._active_model is None:
                result.error = "No model available"
                return result
        
        self.state = CortexState.PROCESSING
        logger.info("Cortex processing query...")
        
        try:
            session = await self._get_session()
            
            # Build messages
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add context if provided
            if context:
                messages.append({
                    "role": "system", 
                    "content": f"Context from recent conversation:\n{context}"
                })
            
            # Add conversation history
            if conversation_history:
                for msg in conversation_history[-6:]:  # Last 6 messages
                    messages.append(msg)
            
            # Add the query
            messages.append({"role": "user", "content": query})
            
            self.state = CortexState.GENERATING
            
            # Call Ollama
            payload = {
                "model": self._active_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                    "top_p": self.config.top_p,
                },
            }
            
            async with session.post(
                f"{self.config.ollama_host}/api/chat",
                json=payload,
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    result.text = data.get("message", {}).get("content", "")
                    result.model_used = self._active_model
                    
                    # Estimate tokens (rough)
                    result.tokens_generated = len(result.text.split())
                else:
                    error = await response.text()
                    logger.error(f"Cortex generation error: {error}")
                    result.error = error
            
            # Calculate metrics
            result.total_time_seconds = time.time() - start_time
            if result.tokens_generated > 0:
                result.tokens_per_second = result.tokens_generated / result.total_time_seconds
            
            # Update stats
            self.generation_count += 1
            self.total_tokens += result.tokens_generated
            self.total_time += result.total_time_seconds
            
            logger.info(
                f"Cortex complete: {result.tokens_generated} tokens "
                f"in {result.total_time_seconds:.2f}s"
            )
            
        except asyncio.TimeoutError:
            result.error = "Generation timed out"
            logger.error("Cortex generation timed out")
        except Exception as e:
            result.error = str(e)
            logger.error(f"Cortex generation failed: {e}")
        finally:
            self.state = CortexState.DORMANT
        
        return result

    async def stream_generate(
        self,
        query: str,
        context: str = "",
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ):
        """
        Stream generation token by token.
        
        Yields:
            Tokens as they are generated
        """
        if self._active_model is None:
            await self.initialize()
            if self._active_model is None:
                yield "Error: No model available"
                return
        
        self.state = CortexState.GENERATING
        
        try:
            session = await self._get_session()
            
            # Build messages
            messages = [{"role": "system", "content": self.system_prompt}]
            
            if context:
                messages.append({
                    "role": "system",
                    "content": f"Context:\n{context}"
                })
            
            if conversation_history:
                for msg in conversation_history[-6:]:
                    messages.append(msg)
            
            messages.append({"role": "user", "content": query})
            
            payload = {
                "model": self._active_model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            }
            
            async with session.post(
                f"{self.config.ollama_host}/api/chat",
                json=payload,
            ) as response:
                if response.status == 200:
                    async for line in response.content:
                        if line:
                            try:
                                import json
                                data = json.loads(line)
                                content = data.get("message", {}).get("content", "")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue
                else:
                    yield f"Error: {await response.text()}"
                    
        except Exception as e:
            logger.error(f"Stream generation error: {e}")
            yield f"Error: {str(e)}"
        finally:
            self.state = CortexState.DORMANT

    def get_stats(self) -> Dict[str, Any]:
        """Get Cortex statistics."""
        return {
            "generation_count": self.generation_count,
            "total_tokens": self.total_tokens,
            "total_time": self.total_time,
            "avg_tokens_per_call": self.total_tokens / max(1, self.generation_count),
            "active_model": self._active_model,
            "state": self.state.value,
        }
