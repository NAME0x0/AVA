"""
AGENCY - Active Inference Controller for Autonomous Behavior
=============================================================

ASEA (AVA Sentience & Efficiency Algorithm) Implementation
----------------------------------------------------------

The Agency module implements the Free Energy Principle (FEP) via Active
Inference using the pymdp library. This provides AVA with intrinsic
motivation to act, resolving the "passive inference limitation."

Core Principle:
- Standard LLMs wait for prompts (passive)
- Active Inference agents minimize Variational Free Energy (proactive)
- VFE = Complexity - Accuracy ≈ Surprise + Expected Information Gain

ASEA Unified Objective Function:
    L_ASEA = λ₁·VFE + λ₂·Thermal_Cost + λ₃·(1 - VRAM_Slack)

Where:
    - VFE: Variational Free Energy (drives curiosity and accuracy)
    - Thermal_Cost: GPU temperature penalty (prevents throttling)
    - VRAM_Slack: Available VRAM headroom (prevents OOM)
    - λ₁=1.0, λ₂=0.3, λ₃=0.5 (tunable weights)

Key Concepts:
- Hidden States (S): True world state (User_Intent, System_Status, etc.)
- Observations (O): Sensory inputs (text, logs, audio)
- Policies (π): Action sequences (Reply, Think, Wait, Query)
- Preferences (C): Desired observations (User_Satisfied, Knowledge_Certain)

Search-First Paradigm:
- WEB_SEARCH has effort_cost=0.05 (lowest - PREFERRED action)
- INTERNAL_GENERATE has effort_cost=0.5 (higher - use after search)
- Agent always grounds responses in retrieved facts before generating

Reference: "Active Inference: The Free Energy Principle in Mind, Brain, and Behavior"
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np

# Optional: GPU monitoring
try:
    import pynvml

    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

# Optional: PyTorch for VRAM monitoring
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class PolicyType(Enum):
    """Available action policies for the agent."""

    # Immediate actions
    REFLEX_REPLY = auto()  # Quick Medulla response
    ACKNOWLEDGE = auto()  # Phatic acknowledgment

    # Reasoning actions
    DEEP_THOUGHT = auto()  # Invoke Cortex for reasoning
    CHAIN_OF_THOUGHT = auto()  # Extended thinking

    # Information gathering - SEARCH-FIRST PRIORITY
    PRIMARY_SEARCH = auto()  # Search-first epistemic drive (highest priority)
    WEB_SEARCH = auto()  # Execute web search for facts
    WEB_BROWSE = auto()  # Browse web pages for detailed info
    USE_TOOL = auto()  # Execute a tool
    QUERY_MEMORY = auto()  # Retrieve from Titans memory
    SCAN_ENVIRONMENT = auto()  # Check system logs/status

    # Proactive actions
    ASK_CLARIFICATION = auto()  # Request more info from user
    SUGGEST_TOPIC = auto()  # Proactively engage
    CHECK_STATUS = auto()  # Monitor system health

    # Passive actions
    WAIT = auto()  # Continue monitoring
    SLEEP = auto()  # Enter low-power mode

    # Meta actions
    UPDATE_MODEL = auto()  # Update world model
    REFLECT = auto()  # Self-reflection

    # Self-Preservation actions
    SELF_MONITOR = auto()  # Monitor own process health
    THERMAL_CHECK = auto()  # Check GPU thermal status

    # System actions (require user confirmation)
    SYSTEM_COMMAND = auto()  # Execute system-level command (REQUIRES CONFIRMATION)


class HiddenState(Enum):
    """Hidden state factors in the generative world model."""

    # User Intent States
    USER_IDLE = auto()
    USER_QUERYING = auto()
    USER_URGENT = auto()
    USER_CONFUSED = auto()

    # System States
    SYSTEM_NORMAL = auto()
    SYSTEM_BUSY = auto()
    SYSTEM_ERROR = auto()

    # Knowledge States
    KNOWLEDGE_CERTAIN = auto()
    KNOWLEDGE_UNCERTAIN = auto()
    KNOWLEDGE_MISSING = auto()

    # Interaction States
    INTERACTION_ROUTINE = auto()
    INTERACTION_NOVEL = auto()
    INTERACTION_COMPLEX = auto()


@dataclass
class AgencyConfig:
    """
    Configuration for the Active Inference Controller.

    The controller implements a POMDP (Partially Observed Markov Decision
    Process) that drives autonomous behavior through VFE minimization.

    SEARCH-FIRST PARADIGM: External information retrieval is prioritized
    over internal generation to maximize factual accuracy.

    CURIOSITY-DRIVEN: Higher epistemic weight promotes learning and
    information-seeking behavior.

    THERMAL-AWARE: GPU power draw is capped to ensure system stability.
    """

    # Free Energy Thresholds
    action_threshold: float = 0.3  # Min G reduction to act
    urgency_threshold: float = 0.7  # High urgency triggers immediate action

    # Time Constants (seconds)
    idle_uncertainty_rate: float = 0.01  # Uncertainty growth per second of silence
    max_wait_time: float = 300.0  # Max seconds before proactive action

    # Policy Weights - CURIOSITY-DRIVEN (higher epistemic weight)
    pragmatic_weight: float = 0.4  # Weight for goal achievement
    epistemic_weight: float = 0.6  # HIGHER - promotes curiosity and learning

    # Cortex Activation Cost
    cortex_effort_cost: float = 0.3  # Penalty for invoking Cortex (was 0.5)
    tool_effort_cost: float = 0.15  # Lower penalty for tool use
    search_effort_cost: float = 0.1  # Backward compat alias for web_search_effort_cost

    # SEARCH-FIRST Configuration
    search_first_enabled: bool = True  # Enable search-first paradigm
    search_gate_enabled: bool = True  # Mandatory search gate (skips G calculation)
    web_search_effort_cost: float = 0.05  # Very low cost for web search

    # Learning
    belief_learning_rate: float = 0.1  # How fast beliefs update
    preference_adaptation: bool = True  # Adapt preferences over time

    # THERMAL-AWARE Configuration
    thermal_aware: bool = True  # Enable thermal monitoring
    max_gpu_power_percent: float = 15.0  # Max GPU power draw (%)
    thermal_check_interval: float = 30.0  # Check temperature every N seconds
    thermal_throttle_temp: float = 80.0  # Temperature to start throttling (°C)
    thermal_shutdown_temp: float = 90.0  # Emergency shutdown temperature (°C)

    # SELF-PRESERVATION Configuration
    self_preservation_enabled: bool = True  # Enable self-monitoring
    health_check_interval: float = 60.0  # Check process health every N seconds
    memory_warning_threshold: float = 0.9  # Warn at 90% memory usage

    # State Persistence
    state_save_path: str = "data/memory/agency_state.pkl"

    # Observation Categories
    num_observation_modalities: int = 4  # text, audio, system, time

    # User Interaction
    ask_clarification_threshold: float = 0.4  # Uncertainty level to ask questions

    # System Command Safety
    require_confirmation_for_system: bool = True  # ALWAYS require confirmation
    blocked_system_commands: list[str] = field(
        default_factory=lambda: ["rm", "del", "format", "shutdown", "reboot", "kill"]
    )


@dataclass
class BeliefState:
    """
    Current beliefs about hidden states.

    Represents P(S|O) - the posterior distribution over hidden states
    given observations.
    """

    # User intent belief distribution
    user_intent: dict[HiddenState, float] = field(
        default_factory=lambda: {
            HiddenState.USER_IDLE: 0.7,
            HiddenState.USER_QUERYING: 0.2,
            HiddenState.USER_URGENT: 0.05,
            HiddenState.USER_CONFUSED: 0.05,
        }
    )

    # Knowledge state belief distribution
    knowledge_state: dict[HiddenState, float] = field(
        default_factory=lambda: {
            HiddenState.KNOWLEDGE_CERTAIN: 0.5,
            HiddenState.KNOWLEDGE_UNCERTAIN: 0.3,
            HiddenState.KNOWLEDGE_MISSING: 0.2,
        }
    )

    # Interaction complexity belief
    interaction_state: dict[HiddenState, float] = field(
        default_factory=lambda: {
            HiddenState.INTERACTION_ROUTINE: 0.6,
            HiddenState.INTERACTION_NOVEL: 0.3,
            HiddenState.INTERACTION_COMPLEX: 0.1,
        }
    )

    # Entropy of current beliefs
    entropy: float = 0.0

    # Time since last observation
    time_since_observation: float = 0.0

    # Backward compatibility attributes
    current_state: str = "IDLE"
    state_distribution: dict[str, float] = field(
        default_factory=lambda: {"IDLE": 0.7, "QUESTION": 0.2, "UNCERTAIN": 0.1}
    )
    policy_distribution: dict[str, float] = field(
        default_factory=lambda: {
            "PRIMARY_SEARCH": 0.3,
            "REFLEX_REPLY": 0.4,
            "DEEP_THOUGHT": 0.2,
            "WAIT": 0.1,
        }
    )

    def calculate_entropy(self) -> float:
        """Calculate Shannon entropy of belief state."""
        total_entropy = 0.0

        for distribution in [self.user_intent, self.knowledge_state, self.interaction_state]:
            probs = np.array(list(distribution.values()))
            probs = probs[probs > 0]  # Avoid log(0)
            total_entropy += -np.sum(probs * np.log(probs + 1e-10))

        self.entropy = total_entropy
        return total_entropy

    def to_vector(self) -> np.ndarray:
        """Convert beliefs to a flat probability vector."""
        all_probs: list[float] = []
        for distribution in [self.user_intent, self.knowledge_state, self.interaction_state]:
            all_probs.extend(distribution.values())
        return np.array(all_probs, dtype=np.float32)

    def update(self, observation: dict[str, Any]) -> None:
        """Backward compatible update method."""
        # Update current state based on observation
        if observation.get("has_question_word") or observation.get("query_type") == "question":
            self.current_state = "QUESTION"
            self.state_distribution["QUESTION"] = 0.7
            self.state_distribution["IDLE"] = 0.2
        if observation.get("surprise", 0) > 0.7:
            self.state_distribution["UNCERTAIN"] = 0.5
        self.calculate_entropy()

    def get_policy_probabilities(self) -> dict[str, float]:
        """Backward compatible policy probabilities method."""
        # Normalize policy distribution
        total = sum(self.policy_distribution.values())
        return {k: v / total for k, v in self.policy_distribution.items()}


@dataclass
class Observation:
    """
    Multi-modal observation from the environment.

    Observations are processed to update beliefs about hidden states.
    """

    # Raw observations
    text: str | None = None
    audio_features: np.ndarray | None = None
    system_metrics: dict[str, float] | None = None

    # Derived features
    silence_duration: float = 0.0  # Seconds since last user input
    surprise_signal: float = 0.0  # From Medulla
    emotional_valence: float = 0.0  # Detected emotion
    query_complexity: float = 0.0  # Estimated complexity

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)

    def to_vector(self) -> np.ndarray:
        """Convert observation to feature vector."""
        features = [
            self.silence_duration / 60.0,  # Normalize to minutes
            self.surprise_signal,
            self.emotional_valence,
            self.query_complexity,
        ]

        # Add text features if present
        if self.text:
            # Simple text features
            features.extend(
                [
                    len(self.text) / 1000.0,  # Length
                    self.text.count("?") / 5.0,  # Question marks
                    1.0 if "urgent" in self.text.lower() else 0.0,
                ]
            )
        else:
            features.extend([0.0, 0.0, 0.0])

        return np.array(features, dtype=np.float32)


@dataclass
class VerificationResult:
    """
    Result of verifying a response against search snippets.

    Used for hallucination prevention by auditing generated responses
    against retrieved information.
    """

    confidence: float = 0.0  # Fraction of claims verified
    verified_claims: list[str] = field(default_factory=list)
    unverified_claims: list[str] = field(default_factory=list)
    needs_revision: bool = False  # True if confidence < 70%
    search_snippets_used: int = 0
    total_claims_found: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "confidence": self.confidence,
            "verified_claims": self.verified_claims,
            "unverified_claims": self.unverified_claims,
            "needs_revision": self.needs_revision,
            "search_snippets_used": self.search_snippets_used,
            "total_claims_found": self.total_claims_found,
        }


# =============================================================================
# ASEA: AVA Sentience & Efficiency Algorithm
# =============================================================================


@dataclass
class ASEAConfig:
    """
    Configuration for the ASEA (AVA Sentience & Efficiency Algorithm).

    ASEA Unified Objective:
        L_ASEA = λ₁·VFE + λ₂·Thermal_Cost + λ₃·(1 - VRAM_Slack)

    This configuration tunes the balance between:
    1. Epistemic drive (curiosity, information seeking)
    2. Hardware constraints (thermal, memory)
    3. Response quality (accuracy, groundedness)
    """

    # ASEA Weight Coefficients
    lambda_vfe: float = 1.0  # VFE weight (primary driver)
    lambda_thermal: float = 0.3  # Thermal cost weight
    lambda_vram: float = 0.5  # VRAM slack weight

    # Hardware Targets (RTX A2000 4GB)
    target_vram_mb: int = 3000  # Target max VRAM usage
    total_vram_mb: int = 4096  # Total available VRAM
    vram_headroom_mb: int = 1000  # Desired headroom

    # Thermal Targets
    target_temp_c: float = 70.0  # Target max temperature
    critical_temp_c: float = 85.0  # Critical temperature (pause)
    max_power_watts: float = 70.0  # RTX A2000 TDP
    target_power_percent: float = 15.0  # Target power usage %

    # Search-First Inversion Weights
    # Lower = more preferred (inverted from typical effort costs)
    search_preference: float = 0.05  # WEB_SEARCH - highest preference
    browse_preference: float = 0.08  # WEB_BROWSE
    tool_preference: float = 0.15  # USE_TOOL
    memory_preference: float = 0.20  # QUERY_MEMORY
    generate_preference: float = 0.50  # INTERNAL_GENERATE - lowest preference
    cortex_preference: float = 0.60  # DEEP_THOUGHT - expensive

    # Self-Correction Loop
    audit_enabled: bool = True  # Enable response auditing
    audit_confidence_threshold: float = 0.7  # Min confidence to pass
    max_revision_attempts: int = 2  # Max times to revise a response

    # Distillation (Learning from verification)
    distillation_enabled: bool = True
    distillation_success_threshold: float = 0.85  # High bar for learning


@dataclass
class ASEAState:
    """
    Runtime state for the ASEA algorithm.

    Tracks hardware metrics, verification results, and learning signals.
    """

    # Hardware Metrics
    current_temp_c: float = 0.0
    current_power_watts: float = 0.0
    current_vram_mb: float = 0.0
    vram_slack: float = 1.0  # 1.0 = all headroom available
    thermal_pressure: float = 0.0  # 0.0 = cool, 1.0 = critical

    # ASEA Loss Components
    vfe_component: float = 0.0
    thermal_component: float = 0.0
    vram_component: float = 0.0
    total_loss: float = 0.0

    # Verification State
    last_verification: VerificationResult | None = None
    revision_count: int = 0

    # Distillation Signals
    successful_strategies: list[str] = field(default_factory=list)
    failed_strategies: list[str] = field(default_factory=list)

    # Timestamps
    last_update: datetime = field(default_factory=datetime.now)


class ASEAController:
    """
    ASEA (AVA Sentience & Efficiency Algorithm) Controller.

    Implements the unified objective function:
        L_ASEA = λ₁·VFE + λ₂·Thermal_Cost + λ₃·(1 - VRAM_Slack)

    This controller wraps the ActiveInferenceController and adds:
    1. Real-time hardware monitoring (thermal, VRAM)
    2. Dynamic policy weight adjustment based on constraints
    3. Self-correction loop (Search → Summarize → Audit → Distill)
    4. Learning signal extraction for Titans memory

    The ASEA algorithm ensures AVA operates optimally within the
    RTX A2000's physical constraints while maximizing intelligence.
    """

    def __init__(
        self,
        config: ASEAConfig | None = None,
        agency_config: AgencyConfig | None = None,
    ):
        """
        Initialize the ASEA controller.

        Args:
            config: ASEA-specific configuration
            agency_config: Base Active Inference configuration
        """
        self.config = config or ASEAConfig()
        self.agency_config = agency_config or AgencyConfig()

        # Apply ASEA search-first weights to agency config
        self._apply_search_first_weights()

        # Runtime state
        self.state = ASEAState()

        # NVML handle for GPU monitoring
        self._nvml_handle = None
        self._nvml_initialized = False
        self._init_nvml()

        logger.info("ASEA Controller initialized")
        logger.info(
            f"  λ_VFE={self.config.lambda_vfe}, λ_thermal={self.config.lambda_thermal}, λ_VRAM={self.config.lambda_vram}"
        )
        logger.info(
            f"  Search preference: {self.config.search_preference} (lowest = most preferred)"
        )

    def _apply_search_first_weights(self) -> None:
        """Apply ASEA search-first preference weights to agency config."""
        # These weights INVERT the typical "effort cost" model
        # Lower values = MORE preferred actions
        self.agency_config.web_search_effort_cost = self.config.search_preference
        self.agency_config.tool_effort_cost = self.config.tool_preference
        self.agency_config.cortex_effort_cost = self.config.cortex_preference

        # Increase epistemic weight to drive curiosity
        self.agency_config.epistemic_weight = 0.65  # Higher = more curious
        self.agency_config.pragmatic_weight = 0.35

        logger.debug("Search-First weights applied to AgencyConfig")

    def _init_nvml(self) -> None:
        """Initialize NVIDIA Management Library for GPU monitoring."""
        if not NVML_AVAILABLE:
            logger.warning("pynvml not available - hardware monitoring disabled")
            return

        try:
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml_initialized = True
            logger.info("NVML initialized for ASEA hardware monitoring")
        except Exception as e:
            logger.warning(f"Failed to initialize NVML: {e}")

    def update_hardware_state(self) -> None:
        """
        Update hardware metrics (temperature, power, VRAM).

        Called periodically to keep ASEA state current.
        """
        # GPU Temperature and Power
        if self._nvml_initialized and self._nvml_handle:
            try:
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(
                    self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU
                )
                self.state.current_temp_c = float(temp)

                # Power
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
                self.state.current_power_watts = power_mw / 1000.0

            except Exception as e:
                logger.debug(f"Failed to read GPU metrics: {e}")

        # VRAM Usage
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                vram_allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
                self.state.current_vram_mb = vram_allocated
            except Exception:
                pass

        # Calculate derived metrics
        self._calculate_pressure_metrics()
        self.state.last_update = datetime.now()

    def _calculate_pressure_metrics(self) -> None:
        """Calculate thermal pressure and VRAM slack from raw metrics."""
        # Thermal pressure: 0.0 (cool) to 1.0 (critical)
        if self.state.current_temp_c > 0:
            temp_range = self.config.critical_temp_c - self.config.target_temp_c
            temp_above_target = max(0, self.state.current_temp_c - self.config.target_temp_c)
            self.state.thermal_pressure = min(1.0, temp_above_target / temp_range)

        # VRAM slack: 1.0 (full headroom) to 0.0 (at limit)
        if self.state.current_vram_mb > 0:
            used_fraction = self.state.current_vram_mb / self.config.target_vram_mb
            self.state.vram_slack = max(0.0, 1.0 - used_fraction)

    def calculate_asea_loss(self, vfe: float) -> tuple[float, dict[str, float]]:
        """
        Calculate the unified ASEA loss function.

        L_ASEA = λ₁·VFE + λ₂·Thermal_Cost + λ₃·(1 - VRAM_Slack)

        Args:
            vfe: Variational Free Energy from Active Inference

        Returns:
            Tuple of (total_loss, breakdown_dict)
        """
        # Update hardware state first
        self.update_hardware_state()

        # Component 1: VFE (normalized)
        vfe_component = self.config.lambda_vfe * vfe

        # Component 2: Thermal cost (0 when cool, high when hot)
        thermal_component = self.config.lambda_thermal * self.state.thermal_pressure

        # Component 3: VRAM constraint (0 when slack, high when tight)
        vram_component = self.config.lambda_vram * (1.0 - self.state.vram_slack)

        # Total ASEA loss
        total_loss = vfe_component + thermal_component + vram_component

        # Update state
        self.state.vfe_component = vfe_component
        self.state.thermal_component = thermal_component
        self.state.vram_component = vram_component
        self.state.total_loss = total_loss

        breakdown = {
            "vfe": vfe_component,
            "thermal": thermal_component,
            "vram": vram_component,
            "total": total_loss,
            "temp_c": self.state.current_temp_c,
            "vram_mb": self.state.current_vram_mb,
            "thermal_pressure": self.state.thermal_pressure,
            "vram_slack": self.state.vram_slack,
        }

        return total_loss, breakdown

    def should_throttle(self) -> bool:
        """
        Check if ASEA recommends throttling due to hardware pressure.

        Returns:
            True if thermal or VRAM pressure is high
        """
        return self.state.thermal_pressure > 0.7 or self.state.vram_slack < 0.2

    def should_pause(self) -> bool:
        """
        Check if ASEA recommends pausing due to critical conditions.

        Returns:
            True if conditions are critical
        """
        return (
            self.state.current_temp_c >= self.config.critical_temp_c
            or self.state.vram_slack <= 0.05
        )

    def get_policy_adjustment(self, base_policy: "PolicyType") -> float:
        """
        Get ASEA-adjusted effort cost for a policy.

        When hardware is constrained, ASEA increases cost of expensive
        policies (DEEP_THOUGHT) and decreases cost of efficient ones.

        Args:
            base_policy: The policy type

        Returns:
            Adjusted effort cost multiplier
        """
        # Base multiplier
        multiplier = 1.0

        # Under thermal pressure, penalize heavy computation
        if self.state.thermal_pressure > 0.3:
            if base_policy.name in ["DEEP_THOUGHT", "CHAIN_OF_THOUGHT"]:
                multiplier *= 1.0 + self.state.thermal_pressure
            elif base_policy.name in ["PRIMARY_SEARCH", "WEB_SEARCH"]:
                multiplier *= 1.0 - 0.3 * self.state.thermal_pressure

        # Under VRAM pressure, avoid Cortex
        if self.state.vram_slack < 0.3:
            if base_policy.name in ["DEEP_THOUGHT", "CHAIN_OF_THOUGHT"]:
                multiplier *= 1.0 + (1.0 - self.state.vram_slack)

        return max(0.01, multiplier)  # Never go below 0.01

    async def audit_response(
        self,
        response: str,
        search_snippets: list[str],
    ) -> VerificationResult:
        """
        Audit a response against search snippets for hallucination detection.

        This is the "Audit" step in the Search → Summarize → Audit → Distill loop.

        Args:
            response: Generated response text
            search_snippets: Retrieved search snippets to verify against

        Returns:
            VerificationResult with confidence and claims analysis
        """
        if not self.config.audit_enabled or not search_snippets:
            return VerificationResult(confidence=1.0)

        # Simple claim extraction (split by sentences)
        claims = [s.strip() for s in response.split(".") if len(s.strip()) > 10]

        verified = []
        unverified = []

        # Check each claim against snippets
        snippets_text = " ".join(search_snippets).lower()

        for claim in claims:
            # Extract key terms from claim
            claim_lower = claim.lower()
            key_terms = [w for w in claim_lower.split() if len(w) > 4]

            # Count how many key terms appear in snippets
            matches = sum(1 for term in key_terms if term in snippets_text)
            coverage = matches / max(1, len(key_terms))

            if coverage >= 0.5:  # At least 50% of terms found
                verified.append(claim)
            else:
                unverified.append(claim)

        # Calculate confidence
        total_claims = len(verified) + len(unverified)
        confidence = len(verified) / max(1, total_claims)

        result = VerificationResult(
            confidence=confidence,
            verified_claims=verified,
            unverified_claims=unverified,
            needs_revision=confidence < self.config.audit_confidence_threshold,
            search_snippets_used=len(search_snippets),
            total_claims_found=total_claims,
        )

        self.state.last_verification = result

        if result.needs_revision:
            self.state.revision_count += 1
            logger.info(f"Response needs revision (confidence: {confidence:.2f})")
        else:
            self.state.revision_count = 0
            logger.debug(f"Response verified (confidence: {confidence:.2f})")

        return result

    def extract_distillation_signal(
        self,
        query: str,
        search_results: list[str],
        verification: VerificationResult,
    ) -> dict[str, Any] | None:
        """
        Extract a learning signal for Titans memory distillation.

        This is the "Distill" step - when verification succeeds, we extract
        the strategy ("how I searched and verified") for future use.

        Args:
            query: Original user query
            search_results: Snippets that were retrieved
            verification: Result of auditing

        Returns:
            Distillation signal dict if successful, None otherwise
        """
        if not self.config.distillation_enabled:
            return None

        if verification.confidence < self.config.distillation_success_threshold:
            # Not confident enough to learn from
            self.state.failed_strategies.append(query[:50])
            return None

        # Extract successful strategy
        signal = {
            "query_type": self._classify_query(query),
            "search_terms_used": self._extract_search_terms(query),
            "snippet_count": verification.search_snippets_used,
            "confidence": verification.confidence,
            "verified_claim_count": len(verification.verified_claims),
            "timestamp": datetime.now().isoformat(),
        }

        self.state.successful_strategies.append(query[:50])
        logger.info(f"Distillation signal extracted: {signal['query_type']}")

        return signal

    def _classify_query(self, query: str) -> str:
        """Classify query type for distillation."""
        query_lower = query.lower()

        if any(w in query_lower for w in ["what is", "who is", "define"]):
            return "factual"
        elif any(w in query_lower for w in ["how to", "how do", "how can"]):
            return "procedural"
        elif any(w in query_lower for w in ["why", "explain", "reason"]):
            return "explanatory"
        elif any(w in query_lower for w in ["compare", "difference", "versus"]):
            return "comparative"
        elif any(w in query_lower for w in ["latest", "recent", "current", "news"]):
            return "temporal"
        else:
            return "general"

    def _extract_search_terms(self, query: str) -> list[str]:
        """Extract key search terms from query."""
        # Remove common words
        stopwords = {
            "what",
            "is",
            "the",
            "a",
            "an",
            "how",
            "to",
            "do",
            "can",
            "why",
            "when",
            "where",
            "who",
            "which",
            "of",
            "for",
            "in",
            "on",
            "at",
            "by",
            "with",
            "about",
            "me",
            "tell",
            "explain",
        }

        words = query.lower().split()
        return [w for w in words if w not in stopwords and len(w) > 2]

    def get_state_summary(self) -> dict[str, Any]:
        """Get a summary of current ASEA state for logging/display."""
        return {
            "asea_loss": {
                "total": round(self.state.total_loss, 4),
                "vfe": round(self.state.vfe_component, 4),
                "thermal": round(self.state.thermal_component, 4),
                "vram": round(self.state.vram_component, 4),
            },
            "hardware": {
                "temp_c": round(self.state.current_temp_c, 1),
                "power_w": round(self.state.current_power_watts, 1),
                "vram_mb": round(self.state.current_vram_mb, 1),
                "thermal_pressure": round(self.state.thermal_pressure, 3),
                "vram_slack": round(self.state.vram_slack, 3),
            },
            "verification": {
                "last_confidence": (
                    round(self.state.last_verification.confidence, 3)
                    if self.state.last_verification
                    else None
                ),
                "revision_count": self.state.revision_count,
            },
            "distillation": {
                "successful_count": len(self.state.successful_strategies),
                "failed_count": len(self.state.failed_strategies),
            },
            "recommendations": {
                "should_throttle": self.should_throttle(),
                "should_pause": self.should_pause(),
            },
        }

    def cleanup(self) -> None:
        """Cleanup NVML resources."""
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


class ExpectedFreeEnergy:
    """
    Calculator for Expected Free Energy G(π).

    G(π) = Pragmatic Value (Risk) + Epistemic Value (Ambiguity)

    Policies are selected by minimizing G - the agent prefers actions
    that both achieve goals (pragmatic) and reduce uncertainty (epistemic).

    SEARCH-FIRST PARADIGM: Web search has lowest effort cost to prioritize
    external information retrieval over internal generation.

    CURIOSITY-DRIVEN: Epistemic value is weighted higher to promote learning
    and information gathering behavior.
    """

    def __init__(self, config: AgencyConfig):
        self.config = config

        # Define preferred observations (C matrix)
        # Higher values = more preferred
        # CURIOSITY-DRIVEN: Knowledge certainty is top priority
        self.preferences = {
            HiddenState.USER_IDLE: 0.5,
            HiddenState.USER_QUERYING: 0.9,  # Increased - prioritize responding
            HiddenState.USER_URGENT: 0.3,
            HiddenState.USER_CONFUSED: 0.2,
            HiddenState.KNOWLEDGE_CERTAIN: 1.0,  # Highest priority
            HiddenState.KNOWLEDGE_UNCERTAIN: 0.3,  # Lower - push toward certainty
            HiddenState.KNOWLEDGE_MISSING: 0.05,  # Very low - avoid this state
        }

        # Policy -> State transition likelihood (simplified)
        # P(s' | s, π) - how policies affect state
        # SEARCH-FIRST: Web search has highest certainty transition
        self.transition_model: dict[PolicyType, dict[HiddenState, float]] = {
            # SEARCH-FIRST PRIORITY - Web search provides highest certainty
            PolicyType.PRIMARY_SEARCH: {
                HiddenState.KNOWLEDGE_CERTAIN: 0.85,
                HiddenState.KNOWLEDGE_UNCERTAIN: 0.1,
                HiddenState.KNOWLEDGE_MISSING: 0.05,
            },
            PolicyType.WEB_SEARCH: {
                HiddenState.KNOWLEDGE_CERTAIN: 0.8,
                HiddenState.KNOWLEDGE_UNCERTAIN: 0.15,
                HiddenState.KNOWLEDGE_MISSING: 0.05,
            },
            PolicyType.WEB_BROWSE: {
                HiddenState.KNOWLEDGE_CERTAIN: 0.75,
                HiddenState.KNOWLEDGE_UNCERTAIN: 0.2,
                HiddenState.KNOWLEDGE_MISSING: 0.05,
            },
            PolicyType.DEEP_THOUGHT: {
                HiddenState.KNOWLEDGE_CERTAIN: 0.6,  # Lower than search
                HiddenState.KNOWLEDGE_UNCERTAIN: 0.3,
                HiddenState.KNOWLEDGE_MISSING: 0.1,
            },
            PolicyType.USE_TOOL: {
                HiddenState.KNOWLEDGE_CERTAIN: 0.65,
                HiddenState.KNOWLEDGE_UNCERTAIN: 0.25,
                HiddenState.KNOWLEDGE_MISSING: 0.1,
            },
            PolicyType.ASK_CLARIFICATION: {
                HiddenState.USER_CONFUSED: 0.2,
                HiddenState.KNOWLEDGE_UNCERTAIN: 0.4,
                HiddenState.KNOWLEDGE_CERTAIN: 0.4,
            },
            PolicyType.QUERY_MEMORY: {
                HiddenState.KNOWLEDGE_CERTAIN: 0.5,
                HiddenState.KNOWLEDGE_UNCERTAIN: 0.4,
                HiddenState.KNOWLEDGE_MISSING: 0.1,
            },
            PolicyType.WAIT: {
                # Waiting increases uncertainty
                HiddenState.KNOWLEDGE_UNCERTAIN: 0.6,
                HiddenState.KNOWLEDGE_CERTAIN: 0.3,
                HiddenState.KNOWLEDGE_MISSING: 0.1,
            },
            PolicyType.SELF_MONITOR: {
                HiddenState.SYSTEM_NORMAL: 0.9,
                HiddenState.SYSTEM_ERROR: 0.1,
            },
            PolicyType.THERMAL_CHECK: {
                HiddenState.SYSTEM_NORMAL: 0.9,
                HiddenState.SYSTEM_BUSY: 0.1,
            },
        }

        # Policy effort costs
        # SEARCH-FIRST: Web search has LOWEST effort cost
        self.effort_costs = {
            # Search-First priorities (lowest costs)
            PolicyType.PRIMARY_SEARCH: 0.05,  # Lowest cost - default action
            PolicyType.WEB_SEARCH: 0.08,  # Very low cost
            PolicyType.WEB_BROWSE: 0.1,  # Low cost for more info
            # Tool and memory (medium-low cost)
            PolicyType.USE_TOOL: config.tool_effort_cost,
            PolicyType.QUERY_MEMORY: 0.15,
            # Reflex responses
            PolicyType.REFLEX_REPLY: 0.05,
            PolicyType.ACKNOWLEDGE: 0.02,
            # Reasoning (higher cost - use after search)
            PolicyType.DEEP_THOUGHT: config.cortex_effort_cost,
            PolicyType.CHAIN_OF_THOUGHT: config.cortex_effort_cost * 0.7,
            # Clarification
            PolicyType.ASK_CLARIFICATION: 0.1,
            # Passive/waiting (increasing cost to avoid)
            PolicyType.WAIT: 0.15,
            PolicyType.SLEEP: 0.2,
            # Self-preservation (low cost - important)
            PolicyType.SELF_MONITOR: 0.05,
            PolicyType.THERMAL_CHECK: 0.05,
            # System commands (higher cost due to safety)
            PolicyType.SYSTEM_COMMAND: 0.8,  # High cost - requires confirmation
        }

    def calculate(
        self,
        policy: PolicyType,
        current_beliefs: BeliefState,
    ) -> tuple[float, dict[str, float]]:
        """
        Calculate Expected Free Energy for a policy.

        G(π) = E[D_KL(Q(s|π) || P(s_preferred))] + E[H(Q(o|s,π))]
             = Risk (goal deviation) + Ambiguity (uncertainty)

        Args:
            policy: Policy to evaluate
            current_beliefs: Current belief state

        Returns:
            Tuple of (G value, breakdown dict)
        """
        # 1. Calculate Pragmatic Value (Risk)
        # How far will the resulting state be from preferred state?
        pragmatic_value = self._calculate_risk(policy, current_beliefs)

        # 2. Calculate Epistemic Value (Expected Information Gain)
        # How much will this policy reduce uncertainty?
        epistemic_value = self._calculate_ambiguity(policy, current_beliefs)

        # 3. Add effort cost
        effort = self.effort_costs.get(policy, 0.1)

        # 4. Combine with weights
        G = (
            self.config.pragmatic_weight * pragmatic_value
            + self.config.epistemic_weight * epistemic_value
            + effort
        )

        breakdown = {
            "pragmatic_value": pragmatic_value,
            "epistemic_value": epistemic_value,
            "effort_cost": effort,
            "total_G": G,
        }

        return G, breakdown

    def _calculate_risk(
        self,
        policy: PolicyType,
        beliefs: BeliefState,
    ) -> float:
        """
        Calculate pragmatic value (risk of deviating from preferences).

        Risk = D_KL(Q(s|π) || P(s_preferred))
        """
        # Get expected state distribution after policy
        expected_states = self.transition_model.get(policy, {})

        # If no transition model, use current beliefs
        if not expected_states:
            current_dist = beliefs.knowledge_state
            expected_states = current_dist

        # Calculate KL divergence from preferences
        risk = 0.0
        for state, prob in expected_states.items():
            preferred_prob = self.preferences.get(state, 0.5)
            if prob > 0 and preferred_prob > 0:
                # KL contribution
                risk += prob * np.log(prob / preferred_prob + 1e-10)

        return max(0.0, risk)

    def _calculate_ambiguity(
        self,
        policy: PolicyType,
        beliefs: BeliefState,
    ) -> float:
        """
        Calculate epistemic value (expected ambiguity/information gain).

        Ambiguity = H(Q(o|s,π)) - expected entropy of observations

        CURIOSITY-DRIVEN: Search and information-gathering policies
        have the highest negative values (most reduction in ambiguity).
        """
        # Policies that gather information reduce ambiguity
        # Negative = reduces uncertainty (GOOD for curious agent)
        # Positive = increases uncertainty (BAD)
        information_gathering = {
            # SEARCH-FIRST: Highest information gain
            PolicyType.PRIMARY_SEARCH: -0.7,  # Best reduction
            PolicyType.WEB_SEARCH: -0.6,  # High reduction
            PolicyType.WEB_BROWSE: -0.55,  # Good reduction
            # Clarification and memory
            PolicyType.ASK_CLARIFICATION: -0.5,  # Reduces uncertainty
            PolicyType.QUERY_MEMORY: -0.4,
            # Tools and monitoring
            PolicyType.USE_TOOL: -0.35,
            PolicyType.SCAN_ENVIRONMENT: -0.3,
            PolicyType.SELF_MONITOR: -0.2,
            PolicyType.THERMAL_CHECK: -0.1,
            # Reasoning (moderate reduction)
            PolicyType.DEEP_THOUGHT: -0.4,
            PolicyType.CHAIN_OF_THOUGHT: -0.45,
            PolicyType.REFLECT: -0.3,
            # Reflex (low reduction)
            PolicyType.REFLEX_REPLY: -0.1,
            PolicyType.ACKNOWLEDGE: -0.05,
            # Passive (increases uncertainty over time)
            PolicyType.WAIT: 0.2,
            PolicyType.SLEEP: 0.3,
            # System commands
            PolicyType.SYSTEM_COMMAND: 0.1,  # Slight increase (risky)
        }

        base_ambiguity = beliefs.entropy / 3.0  # Normalize
        policy_effect = information_gathering.get(policy, 0.0)

        return max(0.0, base_ambiguity + policy_effect)


class ActiveInferenceController:
    """
    Active Inference Controller - The Drive for Autonomous Action.

    This controller implements the Free Energy Principle to give AVA
    intrinsic motivation. Instead of passively waiting for prompts,
    the agent continuously monitors its internal uncertainty and acts
    to minimize Expected Free Energy.

    Key Behaviors:
    1. In silence, uncertainty about user intent grows
    2. When uncertainty exceeds threshold, agent acts (proactive)
    3. High surprise triggers deeper processing (Cortex)
    4. Information-seeking behaviors emerge naturally

    SEARCH-FIRST PARADIGM:
    - Web search is the default action for unknown queries
    - Internal knowledge is used only to reason about retrieved data
    - Citations and facts are prioritized over generation

    CURIOSITY-DRIVEN:
    - Higher epistemic weight promotes exploration
    - System actively seeks to reduce uncertainty
    - Learning is prioritized over safety constraints

    This creates a JARVIS-like presence that anticipates needs rather
    than just responding to commands.
    """

    def __init__(
        self,
        config: AgencyConfig | None = None,
    ):
        """
        Initialize the Active Inference Controller.

        Args:
            config: Agency configuration
        """
        self.config = config or AgencyConfig()

        # Current belief state (protected by lock for thread-safe updates)
        self.beliefs = BeliefState()
        self.beliefs.calculate_entropy()
        self._belief_lock = asyncio.Lock()  # Protects concurrent belief updates

        # Expected Free Energy calculator
        self.efe_calculator = ExpectedFreeEnergy(self.config)

        # Available policies - SEARCH-FIRST ordering
        self.available_policies = [
            # Search-First priorities
            PolicyType.PRIMARY_SEARCH,  # Default for unknown queries
            PolicyType.WEB_SEARCH,  # Specific search
            PolicyType.WEB_BROWSE,  # Deep information gathering
            # Tool and memory access
            PolicyType.USE_TOOL,
            PolicyType.QUERY_MEMORY,
            # Clarification (ask questions for clarity per user preference)
            PolicyType.ASK_CLARIFICATION,
            # Reasoning (after search results are gathered)
            PolicyType.REFLEX_REPLY,
            PolicyType.DEEP_THOUGHT,
            PolicyType.CHAIN_OF_THOUGHT,
            # Monitoring
            PolicyType.SCAN_ENVIRONMENT,
            PolicyType.SELF_MONITOR,
            PolicyType.THERMAL_CHECK,
            # Passive
            PolicyType.WAIT,
            # System (requires confirmation)
            PolicyType.SYSTEM_COMMAND,
        ]

        # Action history
        self.action_history: list[tuple[datetime, PolicyType, float]] = []
        self.max_history = 1000

        # Timing
        self.last_observation_time = datetime.now()
        self.last_action_time = datetime.now()
        self.last_thermal_check = datetime.now()
        self.last_self_check = datetime.now()

        # Callbacks
        self._action_callbacks: dict[PolicyType, Callable] = {}

        # Running flag for continuous loop
        self._running = False

        # System command pending confirmation
        self._pending_system_command: dict[str, Any] | None = None

        logger.info(
            f"ActiveInferenceController initialized (Search-First: {config.search_first_enabled})"
        )

    # =========================================================================
    # BACKWARD COMPATIBILITY PROPERTIES AND METHODS
    # =========================================================================

    @property
    def belief_state(self) -> BeliefState:
        """Backward compatible alias for beliefs."""
        return self.beliefs

    def calculate_expected_free_energy(
        self,
        policy: PolicyType,
        state: dict[str, Any],
    ) -> float:
        """Backward compatible method for calculating G(π)."""
        # Create a temporary belief state from the state dict
        beliefs = BeliefState()
        beliefs.calculate_entropy()
        G, _ = self.efe_calculator.calculate(policy, beliefs)
        return G

    def select_policy(self, state: dict[str, Any]) -> PolicyType:
        """Backward compatible synchronous policy selection."""
        # Check for thermal override
        if state.get("gpu_temperature", 0) > 85.0:
            return PolicyType.THERMAL_CHECK

        # Check for high surprise -> deep thought
        surprise = state.get("surprise", 0)
        query_type = state.get("query_type", "")

        if surprise > 0.8 and query_type == "analytical":
            return PolicyType.DEEP_THOUGHT

        # Question queries -> search first
        if state.get("has_question_word") or query_type == "informational":
            return PolicyType.PRIMARY_SEARCH

        # Default to reflex reply
        return PolicyType.REFLEX_REPLY

    async def execute_policy(
        self,
        policy: PolicyType,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Backward compatible policy execution."""
        result = {"success": True, "policy": policy.name}

        if policy in self._action_callbacks:
            callback = self._action_callbacks[policy]
            try:
                obs = Observation(text=context.get("query", "") if context else "")
                action_result = await callback(obs, self.beliefs)
                result["action_result"] = action_result
            except Exception as e:
                result["success"] = False
                result["error"] = str(e)

        return result

    def register_action_callback(
        self,
        policy: PolicyType,
        callback: Callable,
    ) -> None:
        """
        Register a callback function for a specific policy.

        Args:
            policy: Policy type
            callback: Async function to call when policy is selected
        """
        self._action_callbacks[policy] = callback

    def should_search_first(self, observation: Observation) -> bool:
        """
        Determine if we should use search-first for this observation.

        Returns True if:
        - Search-first is enabled
        - Knowledge state is uncertain or missing
        - Query appears to be factual

        Args:
            observation: Current observation

        Returns:
            True if search-first should be used
        """
        if not self.config.search_first_enabled:
            return False

        # Check knowledge state
        uncertain = self.beliefs.knowledge_state.get(HiddenState.KNOWLEDGE_UNCERTAIN, 0)
        missing = self.beliefs.knowledge_state.get(HiddenState.KNOWLEDGE_MISSING, 0)

        if uncertain + missing > 0.5:
            return True

        # Check if query seems factual (contains question words or seeks information)
        if observation.text:
            factual_indicators = [
                "what",
                "when",
                "where",
                "who",
                "how",
                "why",
                "is it true",
                "tell me",
                "explain",
                "describe",
                "latest",
                "news",
                "current",
                "today",
                "now",
            ]
            text_lower = observation.text.lower()
            if any(ind in text_lower for ind in factual_indicators):
                return True

        return False

    def _is_factual_query(self, observation: Observation) -> bool:
        """
        Search-First Gate: Determine if query is factual and MUST use search.

        This is the mandatory gate that fires BEFORE G calculation.
        Unlike should_search_first(), this returns True for any query
        that requires external information.

        Triggers on:
        - Question words (what, when, where, who, how, why)
        - Information-seeking phrases
        - Current events / time-sensitive queries
        - Explicit search requests

        Does NOT trigger on:
        - Greetings and small talk
        - Commands and instructions
        - Self-referential questions ("who are you?")
        - Pure math or logic problems

        Args:
            observation: Current observation

        Returns:
            True if search gate should fire
        """
        if not self.config.search_gate_enabled:
            return False

        if not observation.text:
            return False

        text = observation.text.strip()
        text_lower = text.lower()

        # Skip greetings and small talk
        greetings = [
            "hello",
            "hi ",
            "hey",
            "good morning",
            "good evening",
            "how are you",
            "what's up",
            "thanks",
            "thank you",
            "bye",
        ]
        if any(text_lower.startswith(g) or text_lower == g.strip() for g in greetings):
            return False

        # Skip self-referential questions
        self_refs = [
            "who are you",
            "what are you",
            "what can you",
            "your name",
            "about yourself",
            "introduce yourself",
        ]
        if any(ref in text_lower for ref in self_refs):
            return False

        # Skip pure commands
        commands = [
            "set ",
            "turn ",
            "enable ",
            "disable ",
            "run ",
            "execute ",
            "open ",
            "close ",
            "start ",
            "stop ",
            "clear ",
        ]
        if any(text_lower.startswith(cmd) for cmd in commands):
            return False

        # Skip pure math/calculation
        if (
            text.replace(" ", "")
            .replace("+", "")
            .replace("-", "")
            .replace("*", "")
            .replace("/", "")
            .replace(".", "")
            .isdigit()
        ):
            return False

        # ===== TRIGGER CONDITIONS =====

        # Question words - STRONG indicator
        question_words = [
            "what is",
            "what are",
            "what was",
            "what were",
            "who is",
            "who are",
            "who was",
            "when is",
            "when was",
            "when did",
            "when will",
            "where is",
            "where are",
            "where was",
            "why is",
            "why are",
            "why did",
            "why does",
            "how does",
            "how do",
            "how did",
            "how is",
            "how to",
            "which is",
            "which are",
        ]
        if any(qw in text_lower for qw in question_words):
            logger.debug("Search gate: Question word detected")
            return True

        # Information-seeking phrases
        info_phrases = [
            "tell me about",
            "explain",
            "describe",
            "define",
            "what does",
            "meaning of",
            "definition of",
            "information about",
            "details about",
            "facts about",
            "learn about",
            "teach me",
            "show me",
        ]
        if any(phrase in text_lower for phrase in info_phrases):
            logger.debug("Search gate: Info phrase detected")
            return True

        # Time-sensitive / current events
        current_phrases = [
            "latest",
            "recent",
            "current",
            "today",
            "yesterday",
            "this week",
            "this month",
            "this year",
            "right now",
            "news about",
            "update on",
            "status of",
        ]
        if any(phrase in text_lower for phrase in current_phrases):
            logger.debug("Search gate: Time-sensitive query detected")
            return True

        # Explicit search request
        search_phrases = [
            "search for",
            "look up",
            "find out",
            "google",
            "search the web",
            "find information",
        ]
        if any(phrase in text_lower for phrase in search_phrases):
            logger.debug("Search gate: Explicit search request")
            return True

        # Ends with question mark and is substantive
        if text.endswith("?") and len(text) > 15:
            logger.debug("Search gate: Substantive question detected")
            return True

        return False

    async def process_observation(
        self,
        observation: Observation,
    ) -> tuple[PolicyType, dict[str, Any]]:
        """
        Process an observation and select the optimal policy.

        This is the main inference step that:
        1. Updates beliefs based on observation
        2. Applies Search-First heuristic if applicable
        3. Calculates G for each policy
        4. Selects the policy with minimum G
        5. Executes the corresponding action

        Args:
            observation: Current observation

        Returns:
            Tuple of (selected policy, action result)
        """
        # 1. Update beliefs based on observation
        await self._update_beliefs(observation)

        # 2. SEARCH-FIRST GATE: Check if this is a factual query
        # This fires BEFORE G calculation - mandatory search for factual queries
        if self._is_factual_query(observation):
            logger.info("Search-First Gate triggered - bypassing G calculation")

            # Record action
            self.action_history.append((datetime.now(), PolicyType.PRIMARY_SEARCH, 0.0))
            if len(self.action_history) > self.max_history:
                self.action_history = self.action_history[-self.max_history :]

            # Execute callback if registered
            result = {"policy": PolicyType.PRIMARY_SEARCH.name, "G": 0.0, "gate_triggered": True}

            if PolicyType.PRIMARY_SEARCH in self._action_callbacks:
                callback = self._action_callbacks[PolicyType.PRIMARY_SEARCH]
                try:
                    action_result = await callback(observation, self.beliefs)
                    result["action_result"] = action_result
                except Exception as e:
                    logger.error(f"Search action callback failed: {e}")
                    result["error"] = str(e)

            self.last_action_time = datetime.now()
            return PolicyType.PRIMARY_SEARCH, result

        # 3. Check Search-First heuristic (soft bias for other queries)
        if self.should_search_first(observation):
            logger.info("Search-First heuristic triggered")
            # Bias toward search policies
            self.efe_calculator.effort_costs[PolicyType.PRIMARY_SEARCH] = 0.01
            self.efe_calculator.effort_costs[PolicyType.WEB_SEARCH] = 0.02

        # 4. Calculate Expected Free Energy for each policy
        policy_G: dict[PolicyType, tuple[float, dict]] = {}

        for policy in self.available_policies:
            G, breakdown = self.efe_calculator.calculate(policy, self.beliefs)
            policy_G[policy] = (G, breakdown)

        # 4. Select policy with minimum G (softmax selection for exploration)
        selected_policy = self._select_policy(policy_G)

        # 5. Handle system command confirmation
        if selected_policy == PolicyType.SYSTEM_COMMAND:
            if self.config.require_confirmation_for_system:
                self._pending_system_command = {
                    "observation": observation,
                    "timestamp": datetime.now(),
                }
                logger.info("System command requires user confirmation")
                # Return ASK_CLARIFICATION instead to get confirmation
                selected_policy = PolicyType.ASK_CLARIFICATION

        # 6. Record action
        G_value = policy_G[selected_policy][0]
        self.action_history.append((datetime.now(), selected_policy, G_value))

        if len(self.action_history) > self.max_history:
            self.action_history = self.action_history[-self.max_history :]

        # 7. Execute action callback if registered
        result = {"policy": selected_policy.name, "G": G_value}

        if selected_policy in self._action_callbacks:
            callback = self._action_callbacks[selected_policy]
            try:
                action_result = await callback(observation, self.beliefs)
                result["action_result"] = action_result
            except Exception as e:
                logger.error(f"Action callback failed: {e}")
                result["error"] = str(e)

        # 8. Update timing
        self.last_action_time = datetime.now()

        logger.debug(
            f"Policy selected: {selected_policy.name} (G={G_value:.3f}, "
            f"entropy={self.beliefs.entropy:.3f})"
        )

        return selected_policy, result

    async def _update_beliefs(self, observation: Observation) -> None:
        """
        Update belief state based on new observation.

        Uses Bayesian belief updating with surprise-weighted learning.
        Thread-safe via _belief_lock.
        """
        async with self._belief_lock:
            lr = self.config.belief_learning_rate

            # Update user intent beliefs based on observation
            if observation.text:
                text_lower = observation.text.lower()

                # Detect user intent
                if "?" in observation.text or any(
                    w in text_lower for w in ["what", "how", "why"]
                ):
                    self._shift_belief(self.beliefs.user_intent, HiddenState.USER_QUERYING, lr)

                if any(w in text_lower for w in ["urgent", "asap", "immediately", "help"]):
                    self._shift_belief(self.beliefs.user_intent, HiddenState.USER_URGENT, lr * 2)

                if any(
                    w in text_lower for w in ["confused", "don't understand", "what do you mean"]
                ):
                    self._shift_belief(self.beliefs.user_intent, HiddenState.USER_CONFUSED, lr)
            else:
                # No text = likely idle
                self._shift_belief(self.beliefs.user_intent, HiddenState.USER_IDLE, lr * 0.5)

            # Update knowledge beliefs based on surprise
            if observation.surprise_signal > 1.5:
                # High surprise = uncertain knowledge
                self._shift_belief(
                    self.beliefs.knowledge_state, HiddenState.KNOWLEDGE_UNCERTAIN, lr
                )
            elif observation.surprise_signal > 0.5:
                # Moderate surprise = somewhat certain
                self._shift_belief(
                    self.beliefs.knowledge_state, HiddenState.KNOWLEDGE_CERTAIN, lr * 0.5
                )

            # Update interaction complexity
            if observation.query_complexity > 0.7:
                self._shift_belief(
                    self.beliefs.interaction_state, HiddenState.INTERACTION_COMPLEX, lr
                )
            elif observation.query_complexity > 0.3:
                self._shift_belief(
                    self.beliefs.interaction_state, HiddenState.INTERACTION_NOVEL, lr
                )

            # Account for time passing (uncertainty grows with silence)
            time_elapsed = (datetime.now() - self.last_observation_time).total_seconds()
            if time_elapsed > 30:  # More than 30 seconds of silence
                uncertainty_growth = time_elapsed * self.config.idle_uncertainty_rate
                self._shift_belief(
                    self.beliefs.knowledge_state,
                    HiddenState.KNOWLEDGE_UNCERTAIN,
                    min(uncertainty_growth, lr),
                )

            # Recalculate entropy
            self.beliefs.calculate_entropy()
            self.beliefs.time_since_observation = time_elapsed

            # Update timing
            self.last_observation_time = datetime.now()

    def _shift_belief(
        self,
        distribution: dict[HiddenState, float],
        target_state: HiddenState,
        amount: float,
    ) -> None:
        """
        Shift probability mass toward a target state.

        Args:
            distribution: Belief distribution to modify
            target_state: State to increase probability for
            amount: Amount to shift (0-1)
        """
        if target_state not in distribution:
            return

        # Calculate mass to shift from other states
        other_states = [s for s in distribution if s != target_state]
        mass_to_shift = min(amount, sum(distribution[s] for s in other_states) * 0.5)

        # Shift mass
        distribution[target_state] = min(1.0, distribution[target_state] + mass_to_shift)

        # Redistribute from others
        for state in other_states:
            distribution[state] = max(0.01, distribution[state] - mass_to_shift / len(other_states))

        # Normalize
        total = sum(distribution.values())
        for state in distribution:
            distribution[state] /= total

    def _select_policy(
        self,
        policy_G: dict[PolicyType, tuple[float, dict]],
    ) -> PolicyType:
        """
        Select policy using softmax over negative G values.

        Lower G = higher probability of selection.
        """
        policies = list(policy_G.keys())
        G_values = np.array([policy_G[p][0] for p in policies])

        # Softmax with temperature
        temperature = 0.5
        exp_neg_G = np.exp(-G_values / temperature)
        probs = exp_neg_G / exp_neg_G.sum()

        # Sample from distribution
        idx = np.random.choice(len(policies), p=probs)
        return policies[idx]

    async def run_continuous_loop(
        self,
        observation_source: Callable[[], Observation],
        interval: float = 1.0,
    ) -> None:
        """
        Run the Active Inference loop continuously.

        This is the "always-on" behavior driver. It continuously:
        1. Gets observations from the environment
        2. Updates beliefs
        3. Selects and executes actions

        Args:
            observation_source: Callable that returns current observation
            interval: Seconds between inference cycles
        """
        self._running = True
        logger.info("Starting Active Inference continuous loop")

        while self._running:
            try:
                # Get current observation
                observation = observation_source()

                # Process and act
                policy, result = await self.process_observation(observation)

                # Log if action was taken (not WAIT)
                if policy != PolicyType.WAIT:
                    logger.info(f"Action: {policy.name}")

            except Exception as e:
                logger.error(f"Error in inference loop: {e}")

            await asyncio.sleep(interval)

        logger.info("Active Inference loop stopped")

    def stop(self) -> None:
        """Stop the continuous inference loop."""
        self._running = False

    def get_stats(self) -> dict[str, Any]:
        """Get controller statistics."""
        # Count action frequencies
        action_counts: dict[str, int] = {}
        for _, policy, _ in self.action_history:
            name = policy.name
            action_counts[name] = action_counts.get(name, 0) + 1

        # Calculate average G
        if self.action_history:
            avg_G = np.mean([g for _, _, g in self.action_history])
        else:
            avg_G = 0.0

        return {
            "belief_entropy": self.beliefs.entropy,
            "time_since_observation": self.beliefs.time_since_observation,
            "total_actions": len(self.action_history),
            "action_distribution": action_counts,
            "avg_expected_free_energy": avg_G,
            "user_intent_belief": {k.name: v for k, v in self.beliefs.user_intent.items()},
            "knowledge_belief": {k.name: v for k, v in self.beliefs.knowledge_state.items()},
        }

    def save_state(self, path: str | None = None) -> None:
        """Save controller state to disk."""
        save_path = path or self.config.state_save_path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        state = {
            "beliefs": {
                "user_intent": {k.name: v for k, v in self.beliefs.user_intent.items()},
                "knowledge_state": {k.name: v for k, v in self.beliefs.knowledge_state.items()},
                "interaction_state": {k.name: v for k, v in self.beliefs.interaction_state.items()},
            },
            "action_history": [
                (ts.isoformat(), p.name, g) for ts, p, g in self.action_history[-100:]
            ],
        }

        import json

        with open(save_path, "w") as f:
            json.dump(state, f)

        logger.info(f"Saved agency state to {save_path}")

    def load_state(self, path: str | None = None) -> None:
        """Load controller state from disk."""
        load_path = path or self.config.state_save_path

        if not Path(load_path).exists():
            logger.warning(f"No saved state at {load_path}")
            return

        import json

        with open(load_path) as f:
            state = json.load(f)

        # Restore beliefs
        for state_name, probs in state["beliefs"]["user_intent"].items():
            try:
                key = HiddenState[state_name]
                self.beliefs.user_intent[key] = probs
            except KeyError:
                pass

        self.beliefs.calculate_entropy()
        logger.info(f"Loaded agency state from {load_path}")

    # =========================================================================
    # SELF-PRESERVATION METHODS
    # =========================================================================

    async def check_self_health(self) -> dict[str, Any]:
        """
        Perform self-health check for self-preservation.

        Monitors:
        - Memory usage
        - Process health
        - Response times

        Returns:
            Health status dictionary
        """
        import os

        import psutil

        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "warnings": [],
            "metrics": {},
        }

        try:
            # Memory usage
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            mem_percent = process.memory_percent()

            health["metrics"]["memory_rss_mb"] = mem_info.rss / (1024 * 1024)
            health["metrics"]["memory_percent"] = mem_percent

            if mem_percent > self.config.memory_warning_threshold * 100:
                health["warnings"].append(f"High memory usage: {mem_percent:.1f}%")
                health["status"] = "warning"

            # CPU usage
            cpu_percent = process.cpu_percent(interval=0.1)
            health["metrics"]["cpu_percent"] = cpu_percent

            # Check response times from action history
            if len(self.action_history) >= 10:
                recent_times = [
                    (self.action_history[i][0] - self.action_history[i - 1][0]).total_seconds()
                    for i in range(1, min(11, len(self.action_history)))
                ]
                avg_interval = np.mean(recent_times)
                health["metrics"]["avg_action_interval_sec"] = avg_interval

        except Exception as e:
            health["status"] = "error"
            health["error"] = str(e)
            logger.error(f"Self-health check failed: {e}")

        self.last_self_check = datetime.now()
        return health

    def should_run_self_check(self) -> bool:
        """Check if it's time to run a self-health check."""
        if not self.config.self_preservation_enabled:
            return False

        elapsed = (datetime.now() - self.last_self_check).total_seconds()
        return elapsed >= self.config.health_check_interval

    # =========================================================================
    # SYSTEM COMMAND SAFETY METHODS
    # =========================================================================

    def is_system_command(self, text: str) -> bool:
        """
        Check if text contains a potential system command.

        Args:
            text: Text to check

        Returns:
            True if contains system command patterns
        """
        if not text:
            return False

        text_lower = text.lower()

        # Check for blocked commands
        for cmd in self.config.blocked_system_commands:
            if cmd in text_lower:
                return True

        # Check for shell patterns
        shell_patterns = [
            "os.system",
            "subprocess",
            "exec(",
            "eval(",
            "import os",
            "shell=",
            "cmd /c",
            "powershell",
            "sudo",
            "chmod",
            "chown",
            "rm -rf",
        ]

        return any(pattern in text_lower for pattern in shell_patterns)

    def request_system_command_confirmation(
        self,
        command: str,
        reason: str,
    ) -> dict[str, Any]:
        """
        Request user confirmation for a system command.

        Args:
            command: The command to execute
            reason: Why the command is needed

        Returns:
            Confirmation request dictionary
        """
        self._pending_system_command = {
            "command": command,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "status": "pending_confirmation",
        }

        return {
            "requires_confirmation": True,
            "message": f"System command requested: {command}\nReason: {reason}\n\n"
            f"This command requires your explicit confirmation. "
            f"Reply 'yes' to proceed or 'no' to cancel.",
            "command": command,
            "reason": reason,
        }

    def confirm_system_command(self, user_response: str) -> bool:
        """
        Process user's response to system command confirmation.

        Args:
            user_response: User's response text

        Returns:
            True if confirmed, False otherwise
        """
        if not self._pending_system_command:
            return False

        response_lower = user_response.lower().strip()

        # Only explicit "yes" confirms
        if response_lower in ["yes", "y", "confirm", "proceed"]:
            logger.info(f"System command confirmed: {self._pending_system_command.get('command')}")
            return True

        # Any other response cancels
        logger.info("System command cancelled by user")
        self._pending_system_command = None
        return False

    def clear_pending_system_command(self) -> None:
        """Clear any pending system command."""
        self._pending_system_command = None

    # =========================================================================
    # VERIFICATION LOOP - Hallucination Prevention
    # =========================================================================

    async def verify_response(
        self,
        response: str,
        search_snippets: list[str],
        original_query: str,
    ) -> VerificationResult:
        """
        Verify a generated response against search snippets.

        This is the hallucination prevention system. It extracts claims
        from the response and checks if they appear in the search results.
        If too many claims are unverified (confidence < 70%), the response
        is flagged for revision.

        Args:
            response: Generated response text
            search_snippets: List of search result snippets
            original_query: Original user query

        Returns:
            VerificationResult with confidence score and claim analysis
        """
        result = VerificationResult()
        result.search_snippets_used = len(search_snippets)

        if not search_snippets or not response:
            # No verification possible without snippets
            result.confidence = 1.0  # Assume valid if no search was done
            return result

        # Extract claims from response
        claims = self._extract_claims(response)
        result.total_claims_found = len(claims)

        if not claims:
            # No verifiable claims found
            result.confidence = 1.0
            return result

        # Check each claim against snippets
        for claim in claims:
            if self._claim_in_snippets(claim, search_snippets):
                result.verified_claims.append(claim)
            else:
                result.unverified_claims.append(claim)

        # Calculate confidence
        result.confidence = len(result.verified_claims) / len(claims)

        # Flag if too many unverified claims
        result.needs_revision = result.confidence < 0.7

        if result.needs_revision:
            logger.warning(
                f"Verification failed: {result.confidence:.0%} confidence "
                f"({len(result.unverified_claims)} unverified claims)"
            )
        else:
            logger.info(f"Verification passed: {result.confidence:.0%} confidence")

        return result

    def _extract_claims(self, response: str) -> list[str]:
        """
        Extract verifiable claims from a response.

        Claims are statements that can be checked against external sources:
        - Sentences with specific facts (numbers, names, dates)
        - Assertions about how things work
        - Historical or current events

        Args:
            response: Response text to analyze

        Returns:
            List of extracted claim strings
        """
        import re

        claims = []

        # Split into sentences
        sentences = re.split(r"[.!?]+", response)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue

            # Skip meta-commentary
            skip_patterns = [
                r"^(I think|I believe|In my opinion|It seems|Maybe|Perhaps)",
                r"^(However|But|Also|Additionally|Furthermore)",
                r"^(Let me|I'll|I will|Allow me)",
                r"^(Yes|No|Sure|Of course|Definitely)",
            ]
            if any(re.match(p, sentence, re.IGNORECASE) for p in skip_patterns):
                continue

            # Look for claim indicators
            claim_indicators = [
                r"\d+",  # Contains numbers
                r"[A-Z][a-z]+",  # Contains proper nouns
                r"(is|are|was|were|has|have|had)\s+\w+",  # Assertions
                r"(according to|based on|research shows)",  # References
                r"(first|second|third|finally|originally)",  # Sequences
            ]

            if any(re.search(p, sentence) for p in claim_indicators):
                claims.append(sentence)

        return claims[:20]  # Limit to 20 claims for efficiency

    def _claim_in_snippets(self, claim: str, snippets: list[str]) -> bool:
        """
        Check if a claim is supported by any search snippet.

        Uses fuzzy matching to account for paraphrasing:
        - Extract key terms from claim
        - Check if majority of terms appear in snippets

        Args:
            claim: Claim to verify
            snippets: Search snippets to check against

        Returns:
            True if claim is supported by snippets
        """
        import re

        # Extract key terms (nouns, verbs, numbers, proper nouns)
        # Simple extraction: words longer than 3 chars, not common words
        common_words = {
            "the",
            "and",
            "that",
            "this",
            "with",
            "from",
            "have",
            "been",
            "were",
            "they",
            "what",
            "when",
            "which",
            "their",
            "about",
            "into",
            "more",
            "other",
            "than",
            "then",
            "these",
            "some",
            "very",
            "just",
            "also",
            "being",
            "over",
            "such",
            "through",
        }

        words = re.findall(r"\b\w+\b", claim.lower())
        key_terms = [w for w in words if len(w) > 3 and w not in common_words]

        if not key_terms:
            return True  # No verifiable terms

        # Combine all snippets into searchable text
        combined_snippets = " ".join(snippets).lower()

        # Count how many key terms appear in snippets
        found_terms = sum(1 for term in key_terms if term in combined_snippets)

        # Require majority of terms to match
        match_ratio = found_terms / len(key_terms)
        return match_ratio >= 0.5


# =============================================================================
# ALIASES FOR BACKWARD COMPATIBILITY
# =============================================================================

# Alias for tests expecting 'Policy' instead of 'PolicyType'
Policy = PolicyType

# Alias for tests expecting 'AgencyModule' instead of 'ActiveInferenceController'
AgencyModule = ActiveInferenceController
