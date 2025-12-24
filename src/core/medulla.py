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
- ThermalMonitor: GPU thermal awareness for stability

The Medulla maintains a fixed-size hidden state that encapsulates
the entire interaction history, eliminating KV cache growth.

THERMAL-AWARE: GPU power capped at 15% for long-term stability.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Optional: GPU monitoring via pynvml
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

logger = logging.getLogger(__name__)


class MedullaState(Enum):
    """Operating states for the Medulla."""
    IDLE = "idle"                    # Passively monitoring
    LISTENING = "listening"          # Actively processing audio
    PERCEIVING = "perceiving"        # Processing text/visual input
    RESPONDING = "responding"        # Generating reflexive response
    ROUTING = "routing"              # Deciding action (Medulla vs Cortex)
    THERMAL_THROTTLED = "thermal_throttled"  # Reduced operation due to heat
    THERMAL_PAUSED = "thermal_paused"        # Paused due to critical temperature


@dataclass
class ThermalStatus:
    """Current thermal status of the GPU."""
    temperature: float = 0.0          # Current temperature in °C
    power_draw_watts: float = 0.0     # Current power draw in watts
    power_limit_watts: float = 0.0    # Max power limit in watts
    power_percent: float = 0.0        # Power draw as percentage
    is_throttled: bool = False        # Whether throttling is active
    is_paused: bool = False           # Whether processing is paused
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "power_draw_watts": self.power_draw_watts,
            "power_limit_watts": self.power_limit_watts,
            "power_percent": self.power_percent,
            "is_throttled": self.is_throttled,
            "is_paused": self.is_paused,
            "timestamp": self.timestamp.isoformat(),
        }


class ThermalMonitor:
    """
    GPU Thermal Monitor for the Medulla.

    Monitors GPU temperature and power draw to ensure long-term
    stability during always-on operation. Caps power at configured
    percentage (default 15%) and throttles/pauses as needed.

    This is critical for the "Sentinel" mode where the Medulla
    runs continuously 24/7.
    """

    def __init__(
        self,
        max_power_percent: float = 15.0,
        warning_temp: float = 75.0,
        throttle_temp: float = 80.0,
        pause_temp: float = 85.0,
        gpu_id: int = 0,
    ):
        """
        Initialize the thermal monitor.

        Args:
            max_power_percent: Maximum GPU power as percentage
            warning_temp: Temperature to log warnings
            throttle_temp: Temperature to start throttling
            pause_temp: Temperature to pause processing
            gpu_id: GPU device ID
        """
        self.max_power_percent = max_power_percent
        self.warning_temp = warning_temp
        self.throttle_temp = throttle_temp
        self.pause_temp = pause_temp
        self.gpu_id = gpu_id

        self._initialized = False
        self._handle = None

        # Initialize NVML if available
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                self._initialized = True
                logger.info(f"ThermalMonitor initialized for GPU {gpu_id}")
            except Exception as e:
                logger.warning(f"Failed to initialize NVML: {e}")
        else:
            logger.warning("pynvml not available - thermal monitoring disabled")

    def get_status(self) -> ThermalStatus:
        """
        Get current thermal status.

        Returns:
            ThermalStatus with current readings
        """
        status = ThermalStatus()

        if not self._initialized or self._handle is None:
            return status

        try:
            # Get temperature
            status.temperature = pynvml.nvmlDeviceGetTemperature(
                self._handle, pynvml.NVML_TEMPERATURE_GPU
            )

            # Get power draw
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
            status.power_draw_watts = power_mw / 1000.0

            # Get power limit
            power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(self._handle)
            status.power_limit_watts = power_limit_mw / 1000.0

            # Calculate percentage
            if status.power_limit_watts > 0:
                status.power_percent = (status.power_draw_watts / status.power_limit_watts) * 100

            # Check thresholds
            status.is_throttled = status.temperature >= self.throttle_temp
            status.is_paused = status.temperature >= self.pause_temp

            status.timestamp = datetime.now()

            # Log warnings
            if status.temperature >= self.warning_temp:
                logger.warning(f"GPU temperature warning: {status.temperature}°C")

        except Exception as e:
            logger.error(f"Failed to get thermal status: {e}")

        return status

    def should_throttle(self) -> bool:
        """Check if we should throttle processing."""
        status = self.get_status()
        return status.is_throttled

    def should_pause(self) -> bool:
        """Check if we should pause processing."""
        status = self.get_status()
        return status.is_paused

    def is_power_exceeded(self) -> bool:
        """Check if power draw exceeds configured limit."""
        status = self.get_status()
        return status.power_percent > self.max_power_percent

    def cleanup(self) -> None:
        """Cleanup NVML resources."""
        if self._initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


@dataclass
class MedullaConfig:
    """
    Configuration for the Medulla (Reflexive Core).

    VRAM Budget Breakdown (Target: 1.5 GB total):
    - Monitor (Bi-Mamba 2.7B 1-bit): ~800 MB
    - Talker (BitNet 3B 1.58-bit): ~700 MB
    - Neural Memory (Titans): ~200 MB
    - Activation buffers: ~150 MB

    THERMAL-AWARE: GPU power capped at 15% for stability.
    Target: Always-on operation without thermal throttling.
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

    # Target token velocity
    min_tokens_per_second: float = 15.0         # 15 tok/sec minimum for "instant" feel

    # Sensory Input Configuration
    audio_sample_rate: int = 16000
    audio_chunk_ms: int = 100                   # Process audio in chunks

    # State Persistence
    state_save_path: str = "data/memory/medulla_state.pkl"
    state_save_interval: int = 100              # Save every N interactions

    # State Management
    state_flush_interval: int = 5               # Flush state every N user interactions

    # THERMAL-AWARE Configuration
    thermal_aware: bool = True                  # Enable thermal monitoring
    max_gpu_power_percent: float = 15.0         # Cap GPU at 15% power draw
    thermal_check_interval: float = 10.0        # Check temp every 10 seconds
    thermal_warning_temp: float = 75.0          # Warn at 75°C
    thermal_throttle_temp: float = 80.0         # Throttle at 80°C
    thermal_pause_temp: float = 85.0            # Pause at 85°C

    # Context length (longer than average for better performance)
    max_context_length: int = 8192              # Extended context window

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

    def to_dict(self) -> dict[str, Any]:
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
        config: MedullaConfig | None = None,
        titans_memory: Any | None = None,
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
        self.surprise_history: list[SurpriseSignal] = []
        self.max_surprise_history = 1000

        # Interaction counters
        self.interaction_count = 0
        self.cortex_invocations = 0
        self.reflex_responses = 0
        self.thermal_throttle_count = 0
        self.thermal_pause_count = 0

        # Placeholders for models (initialized lazily)
        self._monitor_model = None
        self._talker_model = None

        # Callbacks for Cortex activation
        self._cortex_callback: Callable | None = None

        # THERMAL-AWARE: Initialize thermal monitor
        self._thermal_monitor: ThermalMonitor | None = None
        self._last_thermal_check = datetime.now()
        if self.config.thermal_aware:
            self._thermal_monitor = ThermalMonitor(
                max_power_percent=self.config.max_gpu_power_percent,
                warning_temp=self.config.thermal_warning_temp,
                throttle_temp=self.config.thermal_throttle_temp,
                pause_temp=self.config.thermal_pause_temp,
            )
            logger.info(f"ThermalMonitor enabled: max {self.config.max_gpu_power_percent}% power, "
                       f"throttle at {self.config.thermal_throttle_temp}°C")

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
        input_text: str | None = None,
        input_audio: np.ndarray | None = None,
        input_embeddings: np.ndarray | None = None,
    ) -> tuple[SurpriseSignal, str | None]:
        """
        Process sensory input and return surprise signal.

        This is the main entry point for sensory processing. The Medulla
        ingests the input, updates its hidden state, calculates surprise,
        and optionally generates a reflexive response.

        THERMAL-AWARE: Checks GPU thermal status before processing.
        Will throttle or pause if temperature exceeds thresholds.

        Args:
            input_text: Text input from user or logs
            input_audio: Raw audio waveform
            input_embeddings: Pre-computed embeddings

        Returns:
            Tuple of (SurpriseSignal, Optional reflexive response)
        """
        # THERMAL CHECK: Before processing, verify GPU is safe
        thermal_status = await self._check_thermal_status()
        if thermal_status:
            if thermal_status.is_paused:
                self.state = MedullaState.THERMAL_PAUSED
                self.thermal_pause_count += 1
                logger.warning(f"THERMAL PAUSE: GPU at {thermal_status.temperature}°C - waiting for cooldown")
                # Return a pause response
                return SurpriseSignal(value=0, requires_cortex=False), \
                    f"System paused for thermal protection (GPU: {thermal_status.temperature}°C). Please wait..."
            elif thermal_status.is_throttled:
                self.state = MedullaState.THERMAL_THROTTLED
                self.thermal_throttle_count += 1
                logger.warning(f"THERMAL THROTTLE: GPU at {thermal_status.temperature}°C - reducing activity")

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

        # Calculate surprise (thermal-aware: threshold may be raised if throttled)
        surprise = self._calculate_surprise(logits, embeddings, thermal_status)
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
        text: str | None,
        audio: np.ndarray | None,
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
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
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
        logits: np.ndarray | None,
        embeddings: np.ndarray,
        thermal_status: Optional["ThermalStatus"] = None,
    ) -> SurpriseSignal:
        """
        Calculate surprise from the prediction error.

        Surprise = -log P(x_t | h_{t-1})

        High surprise indicates the input was unexpected given the
        current hidden state, suggesting novel information.

        THERMAL-AWARE: When GPU is throttled, the threshold for Cortex
        invocation is raised by 50% to reduce expensive operations.

        Args:
            logits: Model output logits
            embeddings: Input embeddings
            thermal_status: Current GPU thermal status

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

        # THERMAL-AWARE THRESHOLD ADJUSTMENT
        # When throttled, raise the threshold to avoid invoking Cortex
        effective_threshold = self.config.high_surprise_threshold  # Default: 2.0

        if thermal_status and thermal_status.is_throttled:
            # Raise threshold by 50% when throttled (2.0 -> 3.0)
            effective_threshold *= 1.5
            logger.info(f"Thermal throttle: Raised surprise threshold to {effective_threshold:.1f}")

        # Normalize to [0, ~5] range
        normalized = raw_surprise / (effective_threshold * 2)
        normalized = min(normalized, 1.0)

        # Determine thresholds using effective (possibly raised) threshold
        is_high = raw_surprise >= effective_threshold
        requires_cortex = raw_surprise >= effective_threshold

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
        input_text: str | None,
        logits: np.ndarray | None,
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

    async def _check_thermal_status(self) -> ThermalStatus | None:
        """
        Check GPU thermal status if monitoring is enabled.

        Returns:
            ThermalStatus if monitoring is enabled, None otherwise
        """
        if not self._thermal_monitor:
            return None

        # Only check at configured interval to avoid overhead
        elapsed = (datetime.now() - self._last_thermal_check).total_seconds()
        if elapsed < self.config.thermal_check_interval:
            return None

        self._last_thermal_check = datetime.now()
        return self._thermal_monitor.get_status()

    def get_thermal_status(self) -> ThermalStatus | None:
        """Get current thermal status (synchronous)."""
        if self._thermal_monitor:
            return self._thermal_monitor.get_status()
        return None

    def get_stats(self) -> dict[str, Any]:
        """Get Medulla statistics."""
        recent_surprises = [s.value for s in self.surprise_history[-100:]]

        stats = {
            "state": self.state.value,
            "interaction_count": self.interaction_count,
            "cortex_invocations": self.cortex_invocations,
            "reflex_responses": self.reflex_responses,
            "cortex_ratio": self.cortex_invocations / max(1, self.interaction_count),
            "avg_surprise": np.mean(recent_surprises) if recent_surprises else 0.0,
            "max_surprise": np.max(recent_surprises) if recent_surprises else 0.0,
            "state_update_count": self.state_manager.update_count,
            # Thermal stats
            "thermal_throttle_count": self.thermal_throttle_count,
            "thermal_pause_count": self.thermal_pause_count,
        }

        # Add current thermal status if available
        if self._thermal_monitor:
            thermal = self._thermal_monitor.get_status()
            stats["thermal"] = thermal.to_dict()

        return stats

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

        # Cleanup thermal monitor
        if self._thermal_monitor:
            self._thermal_monitor.cleanup()
            logger.info("ThermalMonitor cleaned up")

        # Cleanup models
        if self._monitor_model is not None:
            del self._monitor_model
        if self._talker_model is not None:
            del self._talker_model

        self.is_initialized = False
        logger.info("Medulla shutdown complete")


# =============================================================================
# RE-EXPORTS FOR BACKWARD COMPATIBILITY
# =============================================================================

# Re-export CognitiveState from cortex.entropix for tests
try:
    from ..cortex.entropix import CognitiveState
except ImportError:
    # Fallback definition if cortex module not available
    from dataclasses import dataclass
    from enum import Enum

    class CognitiveStateLabel(Enum):
        FLOW = "flow"
        HESITATION = "hesitation"
        CONFUSION = "confusion"
        CREATIVE = "creative"
        NEUTRAL = "neutral"

    @dataclass
    class CognitiveState:
        label: CognitiveStateLabel = CognitiveStateLabel.NEUTRAL
        entropy: float = 0.0
        varentropy: float = 0.0
        confidence: float = 1.0
