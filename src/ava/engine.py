"""
AVA Engine - Unified Cortex-Medulla Brain
==========================================

This is the core brain of AVA, implementing:
1. Medulla: Fast, reflexive responses
2. Cortex: Deep, thoughtful reasoning
3. Intelligent routing between them
4. Tool integration
5. Accuracy verification

Design Philosophy:
- Accuracy over speed (but still responsive)
- Use all available tools when helpful
- Verify important responses
- Learn from interactions
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import httpx
import numpy as np

from .config import EngineConfig

logger = logging.getLogger(__name__)


class CognitiveState(Enum):
    """Current cognitive processing state."""

    FLOW = "FLOW"  # Confident, smooth processing
    HESITATION = "HESITATION"  # Uncertain, needs more thought
    CONFUSION = "CONFUSION"  # High uncertainty
    CREATIVE = "CREATIVE"  # Exploring possibilities
    VERIFYING = "VERIFYING"  # Double-checking response


class ProcessingMode(Enum):
    """Which processing path to use."""

    MEDULLA = auto()  # Fast, reflexive
    CORTEX = auto()  # Deep, thoughtful
    HYBRID = auto()  # Both (verify with Cortex)


@dataclass
class _EngineResponse:
    """Internal response from the engine. Use Response from src.ava for public API."""

    text: str = ""
    used_cortex: bool = False
    cognitive_state: CognitiveState = CognitiveState.FLOW
    surprise: float = 0.0
    complexity: float = 0.0
    confidence: float = 0.8
    response_time_ms: float = 0.0
    tools_used: list[str] = field(default_factory=list)
    verified: bool = False
    error: str | None = None


class AVAEngine:
    """
    The brain of AVA - handles all thinking and reasoning.

    Architecture:
    - Medulla: Quick responses using small model
    - Cortex: Deep reasoning using full model capacity
    - Router: Decides which to use based on query analysis
    - Verifier: Optional double-check for accuracy
    """

    def __init__(self, config: EngineConfig = None):
        self.config = config or EngineConfig()
        self._client: httpx.AsyncClient | None = None

        # State
        self.is_initialized = False
        self._available_models: list[str] = []
        self._active_fast_model: str = ""
        self._active_deep_model: str = ""

        # Conversation context
        self.conversation_history: list[dict[str, str]] = []

        # Embedding cache for surprise calculation
        self._embedding_centroid: np.ndarray | None = None
        self._embedding_history: list[np.ndarray] = []

        # Statistics
        self.total_requests = 0
        self.cortex_requests = 0
        self.verification_count = 0

        logger.info("AVA Engine created")

    async def initialize(self) -> bool:
        """
        Initialize the engine.

        Returns:
            True if successful
        """
        logger.info("=" * 60)
        logger.info("INITIALIZING AVA ENGINE")
        logger.info("=" * 60)

        # Create HTTP client
        self._client = httpx.AsyncClient(
            base_url=self.config.ollama.host, timeout=self.config.ollama.timeout
        )

        # Check Ollama health
        if not await self._check_ollama():
            logger.error("Ollama is not running. Please start it with: ollama serve")
            return False

        # Get available models
        self._available_models = await self._get_models()
        logger.info(f"Available models: {self._available_models}")

        if not self._available_models:
            logger.error("No models available. Pull a model with: ollama pull gemma3:4b")
            return False

        # Select models
        self._active_fast_model = self._select_model(self.config.ollama.fast_model)
        self._active_deep_model = self._select_model(self.config.ollama.deep_model)

        logger.info(f"Fast model (Medulla): {self._active_fast_model}")
        logger.info(f"Deep model (Cortex): {self._active_deep_model}")
        logger.info("=" * 60)
        logger.info("AVA ENGINE READY")
        logger.info("=" * 60)

        self.is_initialized = True
        return True

    async def _check_ollama(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = await self._client.get("/")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")
            return False

    async def _get_models(self) -> list[str]:
        """Get list of available models."""
        try:
            response = await self._client.get("/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.warning(f"Failed to get models: {e}")
        return []

    def _select_model(self, preferred: str) -> str:
        """Select the best available model."""
        if preferred in self._available_models:
            return preferred
        # Use first available
        return self._available_models[0] if self._available_models else ""

    async def process(
        self,
        user_input: str,
        memory=None,
        tools=None,
        force_cortex: bool = False,
        verify: bool = None,
    ) -> _EngineResponse:
        """
        Process user input and generate response.

        This is the main entry point. It:
        1. Analyzes the query complexity
        2. Calculates surprise (how novel is this?)
        3. Decides processing mode (Medulla vs Cortex)
        4. Executes any needed tools
        5. Generates response
        6. Optionally verifies for accuracy

        Args:
            user_input: The user's message
            memory: Optional conversation memory
            tools: Optional tool manager
            force_cortex: Force deep thinking mode
            verify: Override verification setting

        Returns:
            _EngineResponse with text and metadata
        """
        if not self.is_initialized:
            await self.initialize()

        start_time = time.time()
        result = _EngineResponse()

        try:
            # 1. Analyze the query
            complexity = self._estimate_complexity(user_input)
            surprise = await self._calculate_surprise(user_input)
            result.complexity = complexity
            result.surprise = surprise

            # 2. Decide processing mode
            mode, reason = self._decide_mode(user_input, complexity, surprise, force_cortex)
            logger.info(f"Processing mode: {mode.name} ({reason})")

            # 3. Check if tools are needed
            tools_used = []
            tool_context = ""
            if tools and self._needs_tools(user_input):
                tools_used, tool_context = await self._execute_tools(user_input, tools)
                result.tools_used = tools_used

            # 4. Generate response
            if mode == ProcessingMode.MEDULLA:
                result.text = await self._generate_medulla(user_input, tool_context)
                result.used_cortex = False
            else:
                result.text = await self._generate_cortex(user_input, tool_context)
                result.used_cortex = True
                self.cortex_requests += 1

            # 5. Verify if needed
            should_verify = verify if verify is not None else self.config.verify_responses
            if should_verify and self._should_verify(user_input, result.text):
                result.verified = await self._verify_response(user_input, result.text)
                self.verification_count += 1
                result.cognitive_state = CognitiveState.VERIFYING
            else:
                result.cognitive_state = self._assess_cognitive_state(
                    surprise, complexity, result.used_cortex
                )

            # 6. Update history
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": result.text})

            # Keep history bounded
            if len(self.conversation_history) > 100:
                self.conversation_history = self.conversation_history[-100:]

            self.total_requests += 1

        except Exception as e:
            logger.error(f"Error processing input: {e}")
            result.error = str(e)
            result.text = f"I encountered an error: {str(e)}"

        result.response_time_ms = (time.time() - start_time) * 1000
        return result

    def _estimate_complexity(self, text: str) -> float:
        """
        Estimate query complexity (0-1).

        Higher complexity = more likely to need Cortex.
        """
        score = 0.0
        text_lower = text.lower()

        # Question complexity
        if "?" in text:
            score += 0.1
        if text.count("?") > 1:
            score += 0.1  # Multiple questions

        # Deep thinking indicators
        deep_keywords = [
            "explain",
            "analyze",
            "compare",
            "contrast",
            "why",
            "how does",
            "what if",
            "implications",
            "evaluate",
            "step by step",
            "reasoning",
            "logic",
            "proof",
        ]
        for kw in deep_keywords:
            if kw in text_lower:
                score += 0.15

        # Length (longer = potentially more complex)
        words = len(text.split())
        if words > 50:
            score += 0.1
        if words > 100:
            score += 0.1

        # Technical terms
        technical = [
            "algorithm",
            "function",
            "variable",
            "database",
            "api",
            "server",
            "protocol",
            "architecture",
        ]
        for term in technical:
            if term in text_lower:
                score += 0.05

        return min(1.0, score)

    async def _calculate_surprise(self, text: str) -> float:
        """
        Calculate how surprising/novel this input is.

        Uses embedding distance from conversation centroid.
        Higher surprise = more novel = might need Cortex.
        """
        try:
            # Get embedding
            embedding = await self._get_embedding(text)
            if embedding is None:
                return 0.3  # Default moderate surprise

            # First message
            if self._embedding_centroid is None:
                self._embedding_centroid = embedding
                self._embedding_history = [embedding]
                return 0.5  # Neutral for first message

            # Calculate distance from centroid
            distance = np.linalg.norm(embedding - self._embedding_centroid)

            # Normalize to 0-1 range (typical distances are 0.1-2.0)
            surprise = min(1.0, distance / 1.5)

            # Update centroid (moving average)
            self._embedding_history.append(embedding)
            if len(self._embedding_history) > 20:
                self._embedding_history = self._embedding_history[-20:]
            self._embedding_centroid = np.mean(self._embedding_history, axis=0)

            return surprise

        except Exception as e:
            logger.warning(f"Surprise calculation failed: {e}")
            return 0.3

    async def _get_embedding(self, text: str) -> np.ndarray | None:
        """Get embedding vector for text."""
        try:
            response = await self._client.post(
                "/api/embeddings", json={"model": self._active_fast_model, "prompt": text}
            )
            if response.status_code == 200:
                data = response.json()
                return np.array(data["embedding"])
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
        return None

    def _decide_mode(
        self,
        text: str,
        complexity: float,
        surprise: float,
        force_cortex: bool,
    ) -> tuple[ProcessingMode, str]:
        """Decide whether to use Medulla or Cortex."""

        if force_cortex:
            return ProcessingMode.CORTEX, "forced"

        # Check keywords
        text_lower = text.lower()
        for keyword in self.config.cortex_keywords:
            if keyword in text_lower:
                return ProcessingMode.CORTEX, f"keyword: {keyword}"

        # Thresholds
        if surprise >= self.config.cortex_surprise_threshold:
            return ProcessingMode.CORTEX, f"high surprise: {surprise:.2f}"

        if complexity >= self.config.cortex_complexity_threshold:
            return ProcessingMode.CORTEX, f"high complexity: {complexity:.2f}"

        return ProcessingMode.MEDULLA, "routine query"

    def _needs_tools(self, text: str) -> bool:
        """Check if the query needs external tools."""
        text_lower = text.lower()
        tool_indicators = [
            "calculate",
            "search",
            "find",
            "look up",
            "weather",
            "what is",
            "who is",
            "when did",
            "how many",
            "convert",
        ]
        return any(ind in text_lower for ind in tool_indicators)

    async def _execute_tools(
        self,
        query: str,
        tools,
    ) -> tuple[list[str], str]:
        """Execute relevant tools and return results."""
        # Tool execution will be handled by ToolManager
        # For now, return empty
        return [], ""

    async def _generate_medulla(self, query: str, context: str = "") -> str:
        """Generate quick response via Medulla (fast model)."""
        system_prompt = """You are AVA, a helpful AI assistant.
Be concise but accurate. If you're unsure, say so.
Focus on being helpful and correct."""

        messages = self._build_messages(system_prompt, query, context)

        return await self._chat_completion(
            self._active_fast_model, messages, temperature=self.config.temperature
        )

    async def _generate_cortex(self, query: str, context: str = "") -> str:
        """Generate thoughtful response via Cortex (deep model)."""
        system_prompt = """You are AVA, a research-grade AI assistant.
Think carefully and provide accurate, well-reasoned responses.
Break down complex problems step by step.
If you're uncertain about something, explain your uncertainty.
Prioritize accuracy over brevity."""

        messages = self._build_messages(system_prompt, query, context)

        return await self._chat_completion(
            self._active_deep_model,
            messages,
            temperature=self.config.temperature * 0.8,  # Slightly lower for reasoning
        )

    def _build_messages(
        self,
        system_prompt: str,
        query: str,
        context: str = "",
    ) -> list[dict[str, str]]:
        """Build message list for chat completion."""
        messages = [{"role": "system", "content": system_prompt}]

        # Add context if available
        if context:
            messages.append({"role": "system", "content": f"Additional context:\n{context}"})

        # Add recent conversation history
        history_window = self.conversation_history[-10:]  # Last 5 exchanges
        messages.extend(history_window)

        # Add current query
        messages.append({"role": "user", "content": query})

        return messages

    async def _chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
    ) -> str:
        """Call Ollama chat completion."""
        try:
            response = await self._client.post(
                "/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": self.config.max_tokens,
                    },
                },
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("message", {}).get("content", "")
            else:
                logger.error(f"Chat completion failed: {response.text}")
                return f"Error: API returned {response.status_code}"

        except Exception as e:
            logger.error(f"Chat completion error: {e}")
            return f"Error: {str(e)}"

    def _should_verify(self, query: str, response: str) -> bool:
        """Decide if response needs verification."""
        # Verify factual claims, code, math
        verify_triggers = [
            "calculate",
            "how many",
            "what is the",
            "when did",
            "code",
            "function",
            "algorithm",
            "```",
        ]
        query_lower = query.lower()
        return any(t in query_lower or t in response.lower() for t in verify_triggers)

    async def _verify_response(self, query: str, response: str) -> bool:
        """Verify response accuracy using a second pass."""
        verify_prompt = f"""Review this response for accuracy:

Question: {query}
Response: {response}

Is this response accurate and complete?
Reply with only "VERIFIED" if correct, or "NEEDS_CORRECTION: [issue]" if there's a problem."""

        messages = [
            {"role": "system", "content": "You are a fact-checker. Be strict about accuracy."},
            {"role": "user", "content": verify_prompt},
        ]

        result = await self._chat_completion(
            self._active_deep_model, messages, temperature=0.3  # Low temperature for verification
        )

        return "VERIFIED" in result.upper()

    def _assess_cognitive_state(
        self,
        surprise: float,
        complexity: float,
        used_cortex: bool,
    ) -> CognitiveState:
        """Assess current cognitive state based on processing."""
        if used_cortex:
            return CognitiveState.CREATIVE
        if surprise > 0.7:
            return CognitiveState.CONFUSION
        if complexity > 0.5:
            return CognitiveState.HESITATION
        return CognitiveState.FLOW

    async def shutdown(self):
        """Cleanup resources."""
        if self._client:
            await self._client.aclose()
        logger.info("AVA Engine shutdown complete")

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            "total_requests": self.total_requests,
            "cortex_requests": self.cortex_requests,
            "cortex_ratio": self.cortex_requests / max(1, self.total_requests),
            "verification_count": self.verification_count,
            "history_length": len(self.conversation_history),
            "models": {
                "fast": self._active_fast_model,
                "deep": self._active_deep_model,
            },
        }
