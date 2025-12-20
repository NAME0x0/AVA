"""
MEDULLA V2 - Real Working Reflexive Core
=========================================

This implementation uses Ollama as the actual backend for:
1. Fast reflexive responses (small model like phi3 or tinyllama)
2. Embedding-based surprise calculation
3. Real conversation understanding

The key difference from v1:
- Actually works with real models via Ollama
- Real surprise calculation from embeddings
- Proper state tracking

VRAM Note: If using Ollama with GPU offloading, a small model like
phi3:mini (3.8B) uses ~2GB VRAM, leaving room for the Cortex.
"""

import asyncio
import logging
import time
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class CognitiveLabel(Enum):
    """Cognitive state labels based on entropy/varentropy analysis."""
    FLOW = "FLOW"               # Low entropy, low varentropy - confident response
    HESITATION = "HESITATION"   # High entropy, low varentropy - uniformly confused
    CONFUSION = "CONFUSION"     # Low entropy, high varentropy - knows it doesn't know
    CREATIVE = "CREATIVE"       # High entropy, high varentropy - exploring possibilities
    UNKNOWN = "UNKNOWN"         # Default state


class MedullaState(Enum):
    """Operating states for the Medulla."""
    IDLE = "idle"
    PERCEIVING = "perceiving"
    RESPONDING = "responding"
    ROUTING = "routing"


@dataclass
class MedullaConfig:
    """Configuration for the Medulla with Ollama backend."""
    
    # Ollama Configuration
    ollama_host: str = "http://localhost:11434"
    
    # Model selection - use small, fast models for Medulla
    fast_model: str = "gemma3:4b"          # Primary fast model
    embedding_model: str = "gemma3:4b"     # For surprise calculation (if nomic unavailable)
    
    # Surprise thresholds (calibrated for nomic embeddings)
    low_surprise_threshold: float = 0.3    # Below = routine
    high_surprise_threshold: float = 0.6   # Above = needs Cortex
    
    # Response settings
    max_tokens: int = 256
    temperature: float = 0.7
    
    # Conversation context
    max_context_messages: int = 10
    
    # State tracking
    embedding_dim: int = 768   # nomic-embed-text dimension


@dataclass
class SurpriseSignal:
    """Surprise metric from embedding distance."""
    value: float = 0.0
    normalized: float = 0.0
    is_high: bool = False
    requires_cortex: bool = False
    query_complexity: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass  
class CognitiveState:
    """Current cognitive state based on response analysis."""
    label: CognitiveLabel = CognitiveLabel.UNKNOWN
    entropy: float = 0.0           # Token distribution entropy
    varentropy: float = 0.0        # Variance of entropy
    confidence: float = 0.0        # Model confidence
    surprise: float = 0.0          # Input surprise
    should_use_tools: bool = False
    should_think: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label.value,
            "entropy": self.entropy,
            "varentropy": self.varentropy,
            "confidence": self.confidence,
            "surprise": self.surprise,
            "should_use_tools": self.should_use_tools,
            "should_think": self.should_think,
        }


class MedullaV2:
    """
    The Medulla V2 - Working Reflexive Core using Ollama.
    
    This provides:
    1. Fast responses via small Ollama model
    2. Embedding-based surprise calculation
    3. Proper cognitive state tracking
    4. Real conversation context management
    """
    
    def __init__(self, config: Optional[MedullaConfig] = None):
        self.config = config or MedullaConfig()
        self.state = MedullaState.IDLE
        
        # Embedding state for surprise calculation
        self._last_embedding: Optional[np.ndarray] = None
        self._embedding_history: List[np.ndarray] = []
        self._max_embedding_history = 50
        
        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []
        
        # Surprise tracking
        self._last_surprise = 0.0
        self.surprise_history: List[float] = []
        
        # Cognitive state
        self.cognitive_state = CognitiveState()
        
        # Statistics
        self.interaction_count = 0
        self.cortex_routes = 0
        
        # HTTP session
        self._session = None
        
        # System prompt for the Medulla
        self.system_prompt = """You are AVA, an AI assistant with a Cortex-Medulla architecture.
        
You are the MEDULLA - the fast, reflexive component. You should:
- Give direct, concise answers to simple questions
- Be helpful and conversational
- Remember context from the conversation
- If a question is complex, indicate you might need to think deeper

Your responses should be natural and not robotic. You have personality."""

        logger.info(f"MedullaV2 initialized with model: {self.config.fast_model}")

    async def _get_session(self):
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=60.0)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def health_check(self) -> bool:
        """Check if Ollama is available."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.config.ollama_host}/api/tags") as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    async def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding vector for text."""
        try:
            session = await self._get_session()
            payload = {
                "model": self.config.embedding_model,
                "prompt": text,
            }
            
            async with session.post(
                f"{self.config.ollama_host}/api/embeddings",
                json=payload,
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    embedding = data.get("embedding", [])
                    if embedding:
                        return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
        return None

    def _calculate_surprise(self, current_embedding: np.ndarray) -> SurpriseSignal:
        """
        Calculate surprise based on embedding distance from context.
        
        Surprise is measured as the cosine distance from the centroid
        of recent embeddings. High surprise = input is different from
        what we've been discussing.
        """
        signal = SurpriseSignal()
        
        if len(self._embedding_history) < 2:
            # Not enough history - moderate surprise
            signal.value = 0.4
            signal.normalized = 0.4
            self._embedding_history.append(current_embedding)
            return signal
        
        # Calculate centroid of recent embeddings
        history_array = np.array(self._embedding_history[-10:])
        centroid = np.mean(history_array, axis=0)
        
        # Cosine similarity
        similarity = np.dot(current_embedding, centroid) / (
            np.linalg.norm(current_embedding) * np.linalg.norm(centroid) + 1e-8
        )
        
        # Convert to distance (0 = identical, 1 = orthogonal)
        distance = 1.0 - similarity
        
        # Normalize to [0, 1] range
        signal.value = float(distance)
        signal.normalized = min(1.0, max(0.0, distance))
        
        # Determine thresholds
        signal.is_high = signal.value >= self.config.low_surprise_threshold
        signal.requires_cortex = signal.value >= self.config.high_surprise_threshold
        
        # Add to history
        self._embedding_history.append(current_embedding)
        if len(self._embedding_history) > self._max_embedding_history:
            self._embedding_history = self._embedding_history[-self._max_embedding_history:]
        
        # Estimate query complexity from text length and embedding variance
        signal.query_complexity = signal.value  # Simple proxy for now
        
        return signal

    def _estimate_complexity(self, text: str) -> float:
        """
        Estimate query complexity from text features.
        
        Factors:
        - Length (longer = more complex)
        - Question words (why, how = more complex)
        - Technical terms
        """
        complexity = 0.0
        text_lower = text.lower()
        
        # Length factor
        word_count = len(text.split())
        complexity += min(0.3, word_count / 50)
        
        # Question complexity
        if any(w in text_lower for w in ["why", "how", "explain", "analyze", "compare"]):
            complexity += 0.3
        elif any(w in text_lower for w in ["what is", "who is", "when"]):
            complexity += 0.1
        
        # Technical indicators
        if any(w in text_lower for w in ["code", "algorithm", "function", "implement", "debug"]):
            complexity += 0.2
        
        # Multi-part questions
        if text.count("?") > 1:
            complexity += 0.2
        
        return min(1.0, complexity)

    def _update_cognitive_state(
        self,
        surprise: float,
        response_length: int,
        response_time: float,
    ) -> CognitiveState:
        """
        Update cognitive state based on processing.
        
        This maps the system's state to cognitive labels:
        - FLOW: Confident, quick response
        - HESITATION: Slow response, moderate confidence
        - CONFUSION: High surprise, uncertain
        - CREATIVE: Complex query being explored
        """
        state = CognitiveState()
        state.surprise = surprise
        
        # Estimate entropy from response characteristics
        # (In a real system, this would come from token probabilities)
        if response_time < 1.0 and surprise < 0.3:
            state.entropy = 0.2
            state.varentropy = 0.1
            state.confidence = 0.9
            state.label = CognitiveLabel.FLOW
        elif surprise > 0.6:
            state.entropy = 0.8
            state.varentropy = 0.6
            state.confidence = 0.3
            state.label = CognitiveLabel.CONFUSION
            state.should_think = True
        elif response_length > 500 or response_time > 3.0:
            state.entropy = 0.6
            state.varentropy = 0.7
            state.confidence = 0.5
            state.label = CognitiveLabel.CREATIVE
        else:
            state.entropy = 0.4
            state.varentropy = 0.3
            state.confidence = 0.7
            state.label = CognitiveLabel.HESITATION
        
        self.cognitive_state = state
        return state

    async def perceive(
        self,
        input_text: str,
    ) -> Tuple[SurpriseSignal, Optional[str]]:
        """
        Process input and return surprise signal.
        
        This is the main sensing method. It:
        1. Embeds the input
        2. Calculates surprise from embedding distance
        3. Updates cognitive state
        
        Returns:
            Tuple of (SurpriseSignal, None) - response generated separately
        """
        self.state = MedullaState.PERCEIVING
        self.interaction_count += 1
        
        # Get embedding
        embedding = await self.get_embedding(input_text)
        
        if embedding is not None:
            # Calculate surprise
            surprise = self._calculate_surprise(embedding)
            self._last_surprise = surprise.value
            self.surprise_history.append(surprise.value)
            
            # Add complexity estimation
            surprise.query_complexity = self._estimate_complexity(input_text)
            
            # Re-evaluate if Cortex needed based on complexity
            if surprise.query_complexity > 0.5:
                surprise.requires_cortex = True
                
            self._last_embedding = embedding
        else:
            # Fallback if embedding fails
            surprise = SurpriseSignal(
                value=self._estimate_complexity(input_text),
                normalized=self._estimate_complexity(input_text),
            )
            surprise.requires_cortex = surprise.value > 0.5
        
        self.state = MedullaState.IDLE
        return surprise, None

    async def respond(
        self,
        input_text: str,
        include_context: bool = True,
    ) -> Tuple[str, CognitiveState]:
        """
        Generate a response using the fast model.
        
        Args:
            input_text: User input
            include_context: Whether to include conversation history
            
        Returns:
            Tuple of (response_text, cognitive_state)
        """
        self.state = MedullaState.RESPONDING
        start_time = time.time()
        
        try:
            session = await self._get_session()
            
            # Build messages
            messages = [{"role": "system", "content": self.system_prompt}]
            
            if include_context and self.conversation_history:
                # Add recent context
                for msg in self.conversation_history[-self.config.max_context_messages:]:
                    messages.append(msg)
            
            messages.append({"role": "user", "content": input_text})
            
            # Call Ollama chat endpoint
            payload = {
                "model": self.config.fast_model,
                "messages": messages,
                "stream": False,
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
                    data = await response.json()
                    response_text = data.get("message", {}).get("content", "")
                else:
                    error = await response.text()
                    logger.error(f"Ollama error: {error}")
                    response_text = "I'm having trouble processing that right now."
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": input_text})
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            # Trim history
            if len(self.conversation_history) > self.config.max_context_messages * 2:
                self.conversation_history = self.conversation_history[-(self.config.max_context_messages * 2):]
            
            # Update cognitive state
            elapsed = time.time() - start_time
            cognitive_state = self._update_cognitive_state(
                surprise=self._last_surprise,
                response_length=len(response_text),
                response_time=elapsed,
            )
            
            return response_text, cognitive_state
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return f"I encountered an error: {str(e)}", CognitiveState()
        finally:
            self.state = MedullaState.IDLE

    def get_state_vector(self) -> np.ndarray:
        """
        Get current state as a vector for Bridge projection.
        
        This returns the last embedding as the state representation,
        which captures the semantic content of recent interaction.
        """
        if self._last_embedding is not None:
            return self._last_embedding
        return np.zeros(self.config.embedding_dim, dtype=np.float32)

    def get_stats(self) -> Dict[str, Any]:
        """Get Medulla statistics."""
        return {
            "interaction_count": self.interaction_count,
            "cortex_routes": self.cortex_routes,
            "avg_surprise": np.mean(self.surprise_history) if self.surprise_history else 0.0,
            "cognitive_state": self.cognitive_state.label.value,
            "state": self.state.value,
        }

    def clear_history(self):
        """Clear conversation and embedding history."""
        self.conversation_history = []
        self._embedding_history = []
        self._last_embedding = None
        self.surprise_history = []
