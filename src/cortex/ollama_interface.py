"""
Ollama Interface for AVA
========================

Provides async interface to Ollama for:
- Text generation (chat/completion)
- Embeddings (for Titans Neural Memory)
- Model management

This is the "lifeblood" of the Titans memory system -
without embeddings, the neural memory cannot function.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    """Configuration for Ollama connection."""
    host: str = "http://localhost:11434"
    model: str = "llama3.2:latest"
    embedding_model: str = "nomic-embed-text"
    timeout: float = 120.0

    # Generation defaults
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    num_predict: int = 2048

    # Context
    num_ctx: int = 4096


class OllamaInterface:
    """
    Async interface to Ollama API.

    Provides methods for:
    - generate(): Text generation
    - get_embedding(): Vector embeddings (critical for Titans)
    - chat(): Multi-turn conversation
    - stream_generate(): Streaming generation
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3.2:latest",
        embedding_model: str = "nomic-embed-text",
        timeout: float = 120.0,
    ):
        """
        Initialize Ollama interface.

        Args:
            host: Ollama API host
            model: Default model for generation
            embedding_model: Model for embeddings
            timeout: Request timeout in seconds
        """
        self.config = OllamaConfig(
            host=host.rstrip("/"),
            model=model,
            embedding_model=embedding_model,
            timeout=timeout,
        )

        self._session: aiohttp.ClientSession | None = None

        logger.info(f"OllamaInterface initialized: {self.config.host}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def health_check(self) -> bool:
        """
        Check if Ollama is running and accessible.

        Returns:
            True if Ollama is healthy
        """
        try:
            session = await self._get_session()
            async with session.get(f"{self.config.host}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = [m["name"] for m in data.get("models", [])]
                    logger.info(f"Ollama healthy. Available models: {models}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    async def list_models(self) -> list[str]:
        """List available models."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.config.host}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        return []

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        system: str | None = None,
    ) -> str:
        """
        Generate text completion.

        Args:
            prompt: The input prompt
            model: Model to use (defaults to config)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            system: System prompt

        Returns:
            Generated text
        """
        try:
            session = await self._get_session()

            payload = {
                "model": model or self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature or self.config.temperature,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                    "num_predict": max_tokens or self.config.num_predict,
                    "num_ctx": self.config.num_ctx,
                },
            }

            if system:
                payload["system"] = system

            if stop:
                payload["options"]["stop"] = stop

            async with session.post(
                f"{self.config.host}/api/generate",
                json=payload,
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("response", "")
                else:
                    error_text = await response.text()
                    logger.error(f"Generation failed: {error_text}")
                    return ""

        except asyncio.TimeoutError:
            logger.error("Generation timed out")
            return ""
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return ""

    async def get_embedding(
        self,
        text: str,
        model: str | None = None,
    ) -> list[float]:
        """
        Get vector embedding for text.

        THIS IS CRITICAL FOR TITANS NEURAL MEMORY.
        The embedding is how we interface with the neural memory MLP.

        Args:
            text: Text to embed
            model: Embedding model (defaults to nomic-embed-text)

        Returns:
            Embedding vector as list of floats
        """
        try:
            session = await self._get_session()

            payload = {
                "model": model or self.config.embedding_model,
                "prompt": text,
            }

            async with session.post(
                f"{self.config.host}/api/embeddings",
                json=payload,
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("embedding", [])
                else:
                    error_text = await response.text()
                    logger.error(f"Embedding failed: {error_text}")
                    return []

        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return []

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Multi-turn chat completion.

        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            model: Model to use
            temperature: Sampling temperature

        Returns:
            Assistant's response
        """
        try:
            session = await self._get_session()

            payload = {
                "model": model or self.config.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature or self.config.temperature,
                    "num_ctx": self.config.num_ctx,
                },
            }

            async with session.post(
                f"{self.config.host}/api/chat",
                json=payload,
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("message", {}).get("content", "")
                else:
                    error_text = await response.text()
                    logger.error(f"Chat failed: {error_text}")
                    return ""

        except Exception as e:
            logger.error(f"Chat error: {e}")
            return ""

    async def stream_generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream text generation token by token.

        Args:
            prompt: Input prompt
            model: Model to use
            temperature: Sampling temperature

        Yields:
            Generated tokens one at a time
        """
        try:
            session = await self._get_session()

            payload = {
                "model": model or self.config.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature or self.config.temperature,
                },
            }

            async with session.post(
                f"{self.config.host}/api/generate",
                json=payload,
            ) as response:
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line)
                            if token := data.get("response"):
                                yield token
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.error(f"Streaming error: {e}")

    async def pull_model(self, model: str) -> bool:
        """
        Pull a model from Ollama library.

        Args:
            model: Model name to pull

        Returns:
            True if successful
        """
        try:
            session = await self._get_session()

            # Increase timeout for model download
            timeout = aiohttp.ClientTimeout(total=3600)  # 1 hour

            async with session.post(
                f"{self.config.host}/api/pull",
                json={"name": model},
                timeout=timeout,
            ) as response:
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line)
                            status = data.get("status", "")
                            if "pulling" in status or "downloading" in status:
                                logger.info(f"Pulling {model}: {status}")
                        except json.JSONDecodeError:
                            continue
                return response.status == 200

        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False


async def create_ollama_interface(
    host: str = "http://localhost:11434",
    model: str = "llama3.2:latest",
    embedding_model: str = "nomic-embed-text",
) -> OllamaInterface:
    """
    Factory function to create and validate Ollama interface.

    Args:
        host: Ollama API host
        model: Default generation model
        embedding_model: Embedding model

    Returns:
        Configured OllamaInterface
    """
    interface = OllamaInterface(
        host=host,
        model=model,
        embedding_model=embedding_model,
    )

    # Validate connection
    if not await interface.health_check():
        logger.warning(
            "Ollama is not responding. Make sure it's running with: ollama serve"
        )

    # Check if required models are available
    available = await interface.list_models()

    if model not in available:
        logger.warning(f"Model {model} not found. Pull with: ollama pull {model}")

    if embedding_model not in available:
        logger.warning(
            f"Embedding model {embedding_model} not found. "
            f"Pull with: ollama pull {embedding_model}"
        )

    return interface
