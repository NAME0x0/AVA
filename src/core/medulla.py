"""
MEDULLA - The Reflexive Core
============================

The Medulla is the always-on sensory processing engine built on:
1. 1-bit BitNet b1.58 models for extreme memory efficiency
2. State Space Models (Mamba) for O(N) inference with O(1) state

Hardware Requirements:
- VRAM: ~1.5 GB total (Monitor + Talker + State buffers)
- Operation: 24/7 continuous, never offloaded

Components:
- Monitor: Bi-Mamba SSM for continuous sensory ingestion
- Talker: BitNet 3B for reflexive responses
- Neural Memory: Titans module for test-time learning

The Medulla maintains a fixed-size hidden state that encapsulates
the entire interaction history, eliminating KV cache growth.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class MedullaState(Enum):
    """Operating states for the Medulla."""
    IDLE = "idle"                    # Passively monitoring
    LISTENING = "listening"          # Actively processing audio
    PERCEIVING = "perceiving"        # Processing text/visual input
    RESPONDING = "responding"        # Generating reflexive response
    ROUTING = "routing"              # Deciding action (Medulla vs Cortex)


@dataclass
class MedullaConfig:
    """
    Configuration for the Medulla (Reflexive Core).
    
    VRAM Budget Breakdown (Target: 1.5 GB total):
    - Monitor (Bi-Mamba 2.7B 1-bit): ~800 MB
    - Talker (BitNet 3B 1.58-bit): ~700 MB
    - Neural Memory (Titans): ~200 MB
    - Activation buffers: ~150 MB
    """
    
    # Model Configuration
    monitor_model: str = "slender-mamba-2.7b"  # 1-bit Mamba SSM
    talker_model: str = "bitnet-3b"             # 1.58-bit BitNet for responses
    
    # Hidden state dimensions (Mamba state size)
    hidden_dim: int = 2560                      # Mamba hidden dimension
    state_dim: int = 16                         # SSM state dimension
    
    # Neural Memory (Titans) Configuration
    memory_dim: int = 768                       # Memory MLP input/output
    memory_hidden_dim: int = 1024               # Memory MLP hidden
    memory_learning_rate: float = 1e-3          # Test-time learning rate
    memory_momentum: float = 0.9                # Gradient momentum
    memory_forget_alpha: float = 0.01           # Forgetting rate
    
    # Surprise Thresholds
    low_surprise_threshold: float = 0.3         # Below = routine/familiar
    high_surprise_threshold: float = 2.0        # Above = invoke Cortex
    
    # Response Configuration
    max_reflex_tokens: int = 32                 # Max tokens for quick response
    reflex_timeout_ms: int = 200                # Target reflex latency
    
    # Sensory Input Configuration
    audio_sample_rate: int = 16000
    audio_chunk_ms: int = 100                   # Process audio in chunks
    
    # State Persistence
    state_save_path: str = "data/memory/medulla_state.pkl"
    state_save_interval: int = 100              # Save every N interactions
    
    # Device Configuration
    device: str = "cuda"
    use_fp16: bool = True                       # Use FP16 for activations


@dataclass
class SurpriseSignal:
    """
    Surprise metric calculated from the Medulla's perception.
    
    Surprise = -log P(x_t | h_{t-1})
    
    High surprise indicates novel/unexpected input that may require
    the Cortex for deeper processing.
    """
    value: float = 0.0                          # Raw surprise value
    normalized: float = 0.0                     # Normalized [0, 1]
    is_high: bool = False                       # Exceeds threshold
    requires_cortex: bool = False               # Should invoke Cortex
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "normalized": self.normalized,
            "is_high": self.is_high,
            "requires_cortex": self.requires_cortex,
            "timestamp": self.timestamp.isoformat(),
        }


class MambaStateManager:
    """
    Manages the Mamba SSM hidden state.
    
    The key advantage of Mamba over Transformers is the fixed-size state.
    This manager handles state updates and provides the state vector for
    projection to the Cortex.
    """
    
    def __init__(self, hidden_dim: int = 2560, state_dim: int = 16, dtype=np.float16):
        """
        Initialize the state manager.
        
        Args:
            hidden_dim: Mamba hidden dimension
            state_dim: SSM state dimension per channel
            dtype: Data type for state storage
        """
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.dtype = dtype
        
        # Initialize state tensors
        # Shape: [hidden_dim, state_dim] for selective state space
        self.h = np.zeros((hidden_dim, state_dim), dtype=dtype)
        
        # Running statistics for normalization
        self.running_mean = np.zeros(hidden_dim, dtype=dtype)
        self.running_var = np.ones(hidden_dim, dtype=dtype)
        self.update_count = 0
        
    def update(self, new_state: np.ndarray) -> None:
        """
        Update the hidden state with new values.
        
        Args:
            new_state: New state tensor from Mamba forward pass
        """
        self.h = new_state.astype(self.dtype)
        
        # Update running statistics
        state_flat = self.h.mean(axis=1)
        self.update_count += 1
        
        # Exponential moving average
        alpha = 0.01
        self.running_mean = (1 - alpha) * self.running_mean + alpha * state_flat
        self.running_var = (1 - alpha) * self.running_var + alpha * (state_flat - self.running_mean) ** 2
        
    def get_projection_vector(self) -> np.ndarray:
        """
        Get a normalized vector suitable for projection to Cortex.
        
        Returns:
            Normalized state vector [hidden_dim]
        """
        # Flatten state and normalize
        state_flat = self.h.mean(axis=1)
        normalized = (state_flat - self.running_mean) / (np.sqrt(self.running_var) + 1e-8)
        return normalized.astype(np.float32)
    
    def save(self, path: str) -> None:
        """Save state to disk."""
        np.savez(
            path,
            h=self.h,
            running_mean=self.running_mean,
            running_var=self.running_var,
            update_count=self.update_count,
        )
        
    def load(self, path: str) -> None:
        """Load state from disk."""
        data = np.load(path)
        self.h = data["h"]
        self.running_mean = data["running_mean"]
        self.running_var = data["running_var"]
        self.update_count = int(data["update_count"])


class Medulla:
    """
    The Medulla - Always-On Reflexive Core.
    
    This component runs continuously on the GPU, processing sensory
    inputs and maintaining a coherent internal state. It handles:
    
    1. Wake-word detection and user intent classification
    2. Phatic/reflexive responses ("I see", "One moment", etc.)
    3. Surprise calculation for Cortex routing
    4. Test-time memory updates via Titans
    
    The Medulla uses State Space Models (Mamba) which have O(N) time
    complexity and O(1) memory for the hidden state, enabling
    infinite-length interactions without KV cache explosion.
    """
    
    def __init__(
        self,
        config: Optional[MedullaConfig] = None,
        titans_memory: Optional[Any] = None,
    ):
        """
        Initialize the Medulla.
        
        Args:
            config: Medulla configuration
            titans_memory: External Titans neural memory module
        """
        self.config = config or MedullaConfig()
        self.titans_memory = titans_memory
        
        # Current state
        self.state = MedullaState.IDLE
        self.is_initialized = False
        
        # Initialize SSM state manager
        self.state_manager = MambaStateManager(
            hidden_dim=self.config.hidden_dim,
            state_dim=self.config.state_dim,
        )
        
        # Surprise tracking
        self.surprise_history: List[SurpriseSignal] = []
        self.max_surprise_history = 1000
        
        # Interaction counters
        self.interaction_count = 0
        self.cortex_invocations = 0
        self.reflex_responses = 0
        
        # Placeholders for models (initialized lazily)
        self._monitor_model = None
        self._talker_model = None
        
        # Callbacks for Cortex activation
        self._cortex_callback: Optional[Callable] = None
        
        logger.info(f"Medulla initialized with config: {self.config}")
    
    async def initialize(self) -> None:
        """
        Initialize the Medulla models and load state.
        
        This should be called once at startup. It loads the 1-bit
        Monitor and Talker models into VRAM.
        """
        if self.is_initialized:
            return
            
        logger.info("Initializing Medulla models...")
        
        try:
            # Attempt to load Mamba model
            await self._load_monitor_model()
            
            # Attempt to load BitNet talker
            await self._load_talker_model()
            
            # Load persisted state if available
            state_path = Path(self.config.state_save_path)
            if state_path.exists():
                logger.info(f"Loading Medulla state from {state_path}")
                self.state_manager.load(str(state_path))
            
            self.is_initialized = True
            logger.info("Medulla initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize Medulla: {e}")
            # Fall back to CPU simulation mode
            logger.warning("Falling back to simulation mode")
            self.is_initialized = True
    
    async def _load_monitor_model(self) -> None:
        """
        Load the Monitor (Bi-Mamba SSM) model.
        
        In full implementation, this would load the actual 1-bit Mamba model.
        For now, we provide a simulation interface.
        """
        try:
            # Attempt to import mamba_ssm
            from mamba_ssm import MambaLMHeadModel
            
            logger.info(f"Loading monitor model: {self.config.monitor_model}")
            # self._monitor_model = MambaLMHeadModel.from_pretrained(
            #     self.config.monitor_model,
            #     dtype=torch.float16 if self.config.use_fp16 else torch.float32,
            # ).cuda()
            
            # Placeholder until actual model is available
            logger.warning("Using simulated Mamba model")
            self._monitor_model = None
            
        except ImportError:
            logger.warning("mamba_ssm not installed - using simulation mode")
            self._monitor_model = None
    
    async def _load_talker_model(self) -> None:
        """
        Load the Talker (BitNet 1.58-bit) model.
        
        In full implementation, this would load the actual BitNet model.
        For now, we provide a simulation interface.
        """
        try:
            # Attempt to import bitnet
            # from bitnet import BitNetModel
            
            logger.info(f"Loading talker model: {self.config.talker_model}")
            # self._talker_model = BitNetModel.from_pretrained(
            #     self.config.talker_model,
            # ).cuda()
            
            # Placeholder until actual model is available
            logger.warning("Using simulated BitNet model")
            self._talker_model = None
            
        except ImportError:
            logger.warning("bitnet not installed - using simulation mode")
            self._talker_model = None
    
    def set_cortex_callback(self, callback: Callable) -> None:
        """
        Register a callback for Cortex activation.
        
        Args:
            callback: Function to call when Cortex should be activated
        """
        self._cortex_callback = callback
    
    async def perceive(
        self,
        input_text: Optional[str] = None,
        input_audio: Optional[np.ndarray] = None,
        input_embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[SurpriseSignal, Optional[str]]:
        """
        Process sensory input and return surprise signal.
        
        This is the main entry point for sensory processing. The Medulla
        ingests the input, updates its hidden state, calculates surprise,
        and optionally generates a reflexive response.
        
        Args:
            input_text: Text input from user or logs
            input_audio: Raw audio waveform
            input_embeddings: Pre-computed embeddings
            
        Returns:
            Tuple of (SurpriseSignal, Optional reflexive response)
        """
        self.state = MedullaState.PERCEIVING
        self.interaction_count += 1
        
        # Convert input to embeddings if needed
        if input_embeddings is None:
            embeddings = await self._embed_input(input_text, input_audio)
        else:
            embeddings = input_embeddings
        
        # Forward pass through Monitor (Mamba SSM)
        new_state, logits = await self._forward_monitor(embeddings)
        
        # Update hidden state
        if new_state is not None:
            self.state_manager.update(new_state)
        
        # Calculate surprise
        surprise = self._calculate_surprise(logits, embeddings)
        self.surprise_history.append(surprise)
        
        # Trim history if needed
        if len(self.surprise_history) > self.max_surprise_history:
            self.surprise_history = self.surprise_history[-self.max_surprise_history:]
        
        # Update Titans neural memory if surprise is significant
        if self.titans_memory and surprise.value >= self.config.low_surprise_threshold:
            await self._update_titans_memory(embeddings, surprise.value)
        
        # Determine if Cortex should be invoked
        reflex_response = None
        
        if surprise.requires_cortex:
            # High surprise - need deeper processing
            self.state = MedullaState.ROUTING
            logger.info(f"High surprise ({surprise.value:.2f}) - routing to Cortex")
            self.cortex_invocations += 1
            
            # Generate placeholder response while Cortex processes
            reflex_response = await self._generate_waiting_response()
            
        else:
            # Low surprise - handle with reflex
            self.state = MedullaState.RESPONDING
            reflex_response = await self._generate_reflex_response(input_text, logits)
            self.reflex_responses += 1
        
        # Save state periodically
        if self.interaction_count % self.config.state_save_interval == 0:
            self._save_state()
        
        self.state = MedullaState.IDLE
        return surprise, reflex_response
    
    async def _embed_input(
        self,
        text: Optional[str],
        audio: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Convert raw input to embeddings.
        
        Args:
            text: Text input
            audio: Audio waveform
            
        Returns:
            Embedding vector
        """
        # Placeholder embedding generation
        # In production, this would use nomic-embed-text or similar
        if text:
            # Simple hash-based placeholder
            np.random.seed(hash(text) % (2**32))
            return np.random.randn(self.config.memory_dim).astype(np.float32)
        elif audio is not None:
            # Would run through whisper for transcription then embed
            np.random.seed(int(audio.sum() * 1000) % (2**32))
            return np.random.randn(self.config.memory_dim).astype(np.float32)
        else:
            return np.zeros(self.config.memory_dim, dtype=np.float32)
    
    async def _forward_monitor(
        self,
        embeddings: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Forward pass through the Monitor (Mamba SSM).
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Tuple of (new_state, logits)
        """
        if self._monitor_model is not None:
            # Real model forward pass
            # output = self._monitor_model(embeddings)
            # return output.last_hidden_state, output.logits
            pass
        
        # Simulation mode
        # Simulate state update with exponential smoothing
        current_state = self.state_manager.h
        noise = np.random.randn(*current_state.shape) * 0.01
        new_state = 0.95 * current_state + 0.05 * noise
        
        # Simulate logits based on input
        logits = np.random.randn(self.config.hidden_dim).astype(np.float32)
        
        return new_state.astype(np.float16), logits
    
    def _calculate_surprise(
        self,
        logits: Optional[np.ndarray],
        embeddings: np.ndarray,
    ) -> SurpriseSignal:
        """
        Calculate surprise from the prediction error.
        
        Surprise = -log P(x_t | h_{t-1})
        
        High surprise indicates the input was unexpected given the
        current hidden state, suggesting novel information.
        
        Args:
            logits: Model output logits
            embeddings: Input embeddings
            
        Returns:
            SurpriseSignal with calculated metrics
        """
        if logits is None:
            # Default moderate surprise
            raw_surprise = 1.0
        else:
            # Calculate prediction error as proxy for surprise
            # In full implementation, this would use proper likelihood
            
            # Normalize logits to probabilities (softmax over sample)
            exp_logits = np.exp(logits - logits.max())
            probs = exp_logits / exp_logits.sum()
            
            # Entropy as surprise proxy
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            raw_surprise = float(entropy)
        
        # Normalize to [0, ~5] range
        normalized = raw_surprise / (self.config.high_surprise_threshold * 2)
        normalized = min(normalized, 1.0)
        
        # Determine thresholds
        is_high = raw_surprise >= self.config.high_surprise_threshold
        requires_cortex = raw_surprise >= self.config.high_surprise_threshold
        
        return SurpriseSignal(
            value=raw_surprise,
            normalized=normalized,
            is_high=is_high,
            requires_cortex=requires_cortex,
        )
    
    async def _update_titans_memory(
        self,
        embeddings: np.ndarray,
        surprise: float,
    ) -> None:
        """
        Update Titans neural memory with surprise-weighted learning.
        
        Args:
            embeddings: Input embeddings to memorize
            surprise: Surprise value for weighting the update
        """
        if self.titans_memory is None:
            return
            
        try:
            # The Titans module learns at test-time
            # Update magnitude is scaled by surprise
            loss = self.titans_memory.memorize(
                embedding=embeddings,
                target=None,  # Autoencoder behavior
                surprise=surprise,
            )
            logger.debug(f"Titans memory update: loss={loss:.4f}, surprise={surprise:.2f}")
        except Exception as e:
            logger.error(f"Failed to update Titans memory: {e}")
    
    async def _generate_waiting_response(self) -> str:
        """
        Generate a placeholder response while Cortex processes.
        
        These are phatic responses that acknowledge the user while
        the system performs deeper reasoning.
        """
        waiting_phrases = [
            "Let me think about that...",
            "One moment while I process that.",
            "Hmm, let me consider this carefully.",
            "Processing your request...",
            "I need a moment to think.",
            "Give me a second to work through this.",
        ]
        
        # Select based on interaction count for variety
        idx = self.interaction_count % len(waiting_phrases)
        return waiting_phrases[idx]
    
    async def _generate_reflex_response(
        self,
        input_text: Optional[str],
        logits: Optional[np.ndarray],
    ) -> str:
        """
        Generate a quick reflexive response using the Talker model.
        
        Args:
            input_text: Original input text
            logits: Prediction logits from Monitor
            
        Returns:
            Reflexive response string
        """
        if self._talker_model is not None:
            # Real model generation
            # response = self._talker_model.generate(...)
            pass
        
        # Simulation mode - simple pattern matching
        if input_text:
            text_lower = input_text.lower()
            
            # Simple reflexive responses
            if any(w in text_lower for w in ["hello", "hi", "hey"]):
                return "Hello! How can I help you?"
            elif any(w in text_lower for w in ["thanks", "thank you"]):
                return "You're welcome!"
            elif "?" in input_text:
                return "That's an interesting question. Let me think about it."
            elif any(w in text_lower for w in ["goodbye", "bye", "see you"]):
                return "Goodbye! Take care."
            else:
                return "I understand."
        
        return "I'm listening."
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get the current hidden state as a projection vector.
        
        This vector is used by the Bridge to project the Medulla's
        context into the Cortex's embedding space.
        
        Returns:
            Normalized state vector suitable for projection
        """
        return self.state_manager.get_projection_vector()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Medulla statistics."""
        recent_surprises = [s.value for s in self.surprise_history[-100:]]
        
        return {
            "state": self.state.value,
            "interaction_count": self.interaction_count,
            "cortex_invocations": self.cortex_invocations,
            "reflex_responses": self.reflex_responses,
            "cortex_ratio": self.cortex_invocations / max(1, self.interaction_count),
            "avg_surprise": np.mean(recent_surprises) if recent_surprises else 0.0,
            "max_surprise": np.max(recent_surprises) if recent_surprises else 0.0,
            "state_update_count": self.state_manager.update_count,
        }
    
    def _save_state(self) -> None:
        """Save Medulla state to disk."""
        try:
            state_path = Path(self.config.state_save_path)
            state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_manager.save(str(state_path))
            logger.debug(f"Saved Medulla state to {state_path}")
        except Exception as e:
            logger.error(f"Failed to save Medulla state: {e}")
    
    async def shutdown(self) -> None:
        """Clean shutdown of the Medulla."""
        logger.info("Shutting down Medulla...")
        self._save_state()
        
        # Cleanup models
        if self._monitor_model is not None:
            del self._monitor_model
        if self._talker_model is not None:
            del self._talker_model
            
        self.is_initialized = False
        logger.info("Medulla shutdown complete")
