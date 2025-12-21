"""
AGENCY - Active Inference Controller for Autonomous Behavior
=============================================================

The Agency module implements the Free Energy Principle (FEP) via Active
Inference using the pymdp library. This provides AVA with intrinsic
motivation to act, resolving the "passive inference limitation."

Core Principle:
- Standard LLMs wait for prompts (passive)
- Active Inference agents minimize Variational Free Energy (proactive)
- VFE = Complexity - Accuracy ≈ Surprise + Expected Information Gain

The agent maintains a generative model of the world and acts to reduce
uncertainty (Expected Information Gain) and achieve preferred states
(Pragmatic Value). This creates JARVIS-like autonomous behavior.

Key Concepts:
- Hidden States (S): True world state (User_Intent, System_Status, etc.)
- Observations (O): Sensory inputs (text, logs, audio)
- Policies (π): Action sequences (Reply, Think, Wait, Query)
- Preferences (C): Desired observations (User_Satisfied, Knowledge_Certain)

Reference: "Active Inference: The Free Energy Principle in Mind, Brain, and Behavior"
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

# Optional: GPU monitoring
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

logger = logging.getLogger(__name__)


class PolicyType(Enum):
    """Available action policies for the agent."""
    
    # Immediate actions
    REFLEX_REPLY = auto()       # Quick Medulla response
    ACKNOWLEDGE = auto()        # Phatic acknowledgment
    
    # Reasoning actions
    DEEP_THOUGHT = auto()       # Invoke Cortex for reasoning
    CHAIN_OF_THOUGHT = auto()   # Extended thinking
    
    # Information gathering - SEARCH-FIRST PRIORITY
    PRIMARY_SEARCH = auto()     # Search-first epistemic drive (highest priority)
    WEB_SEARCH = auto()         # Execute web search for facts
    WEB_BROWSE = auto()         # Browse web pages for detailed info
    USE_TOOL = auto()           # Execute a tool
    QUERY_MEMORY = auto()       # Retrieve from Titans memory
    SCAN_ENVIRONMENT = auto()   # Check system logs/status
    
    # Proactive actions
    ASK_CLARIFICATION = auto()  # Request more info from user
    SUGGEST_TOPIC = auto()      # Proactively engage
    CHECK_STATUS = auto()       # Monitor system health
    
    # Passive actions
    WAIT = auto()               # Continue monitoring
    SLEEP = auto()              # Enter low-power mode
    
    # Meta actions
    UPDATE_MODEL = auto()       # Update world model
    REFLECT = auto()            # Self-reflection
    
    # Self-Preservation actions
    SELF_MONITOR = auto()       # Monitor own process health
    THERMAL_CHECK = auto()      # Check GPU thermal status
    
    # System actions (require user confirmation)
    SYSTEM_COMMAND = auto()     # Execute system-level command (REQUIRES CONFIRMATION)


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
    action_threshold: float = 0.3         # Min G reduction to act
    urgency_threshold: float = 0.7        # High urgency triggers immediate action
    
    # Time Constants (seconds)
    idle_uncertainty_rate: float = 0.01   # Uncertainty growth per second of silence
    max_wait_time: float = 300.0          # Max seconds before proactive action
    
    # Policy Weights - CURIOSITY-DRIVEN (higher epistemic weight)
    pragmatic_weight: float = 0.4         # Weight for goal achievement
    epistemic_weight: float = 0.6         # HIGHER - promotes curiosity and learning
    
    # Cortex Activation Cost
    cortex_effort_cost: float = 0.5       # Penalty for invoking Cortex
    tool_effort_cost: float = 0.15        # Lower penalty for tool use
    
    # SEARCH-FIRST Configuration
    search_first_enabled: bool = True     # Enable search-first paradigm
    web_search_effort_cost: float = 0.05  # Very low cost for web search
    
    # Learning
    belief_learning_rate: float = 0.1     # How fast beliefs update
    preference_adaptation: bool = True    # Adapt preferences over time
    
    # THERMAL-AWARE Configuration
    thermal_aware: bool = True            # Enable thermal monitoring
    max_gpu_power_percent: float = 15.0   # Max GPU power draw (%)
    thermal_check_interval: float = 30.0  # Check temperature every N seconds
    thermal_throttle_temp: float = 80.0   # Temperature to start throttling (°C)
    thermal_shutdown_temp: float = 90.0   # Emergency shutdown temperature (°C)
    
    # SELF-PRESERVATION Configuration
    self_preservation_enabled: bool = True # Enable self-monitoring
    health_check_interval: float = 60.0   # Check process health every N seconds
    memory_warning_threshold: float = 0.9 # Warn at 90% memory usage
    
    # State Persistence
    state_save_path: str = "data/memory/agency_state.pkl"
    
    # Observation Categories
    num_observation_modalities: int = 4   # text, audio, system, time
    
    # User Interaction
    ask_clarification_threshold: float = 0.4  # Uncertainty level to ask questions
    
    # System Command Safety
    require_confirmation_for_system: bool = True  # ALWAYS require confirmation
    blocked_system_commands: List[str] = field(default_factory=lambda: [
        "rm", "del", "format", "shutdown", "reboot", "kill"
    ])


@dataclass
class BeliefState:
    """
    Current beliefs about hidden states.
    
    Represents P(S|O) - the posterior distribution over hidden states
    given observations.
    """
    
    # User intent belief distribution
    user_intent: Dict[HiddenState, float] = field(default_factory=lambda: {
        HiddenState.USER_IDLE: 0.7,
        HiddenState.USER_QUERYING: 0.2,
        HiddenState.USER_URGENT: 0.05,
        HiddenState.USER_CONFUSED: 0.05,
    })
    
    # Knowledge state belief distribution
    knowledge_state: Dict[HiddenState, float] = field(default_factory=lambda: {
        HiddenState.KNOWLEDGE_CERTAIN: 0.5,
        HiddenState.KNOWLEDGE_UNCERTAIN: 0.3,
        HiddenState.KNOWLEDGE_MISSING: 0.2,
    })
    
    # Interaction complexity belief
    interaction_state: Dict[HiddenState, float] = field(default_factory=lambda: {
        HiddenState.INTERACTION_ROUTINE: 0.6,
        HiddenState.INTERACTION_NOVEL: 0.3,
        HiddenState.INTERACTION_COMPLEX: 0.1,
    })
    
    # Entropy of current beliefs
    entropy: float = 0.0
    
    # Time since last observation
    time_since_observation: float = 0.0
    
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
        all_probs = []
        for distribution in [self.user_intent, self.knowledge_state, self.interaction_state]:
            all_probs.extend(distribution.values())
        return np.array(all_probs, dtype=np.float32)


@dataclass
class Observation:
    """
    Multi-modal observation from the environment.
    
    Observations are processed to update beliefs about hidden states.
    """
    
    # Raw observations
    text: Optional[str] = None
    audio_features: Optional[np.ndarray] = None
    system_metrics: Optional[Dict[str, float]] = None
    
    # Derived features
    silence_duration: float = 0.0         # Seconds since last user input
    surprise_signal: float = 0.0          # From Medulla
    emotional_valence: float = 0.0        # Detected emotion
    query_complexity: float = 0.0         # Estimated complexity
    
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
            features.extend([
                len(self.text) / 1000.0,           # Length
                self.text.count("?") / 5.0,        # Question marks
                1.0 if "urgent" in self.text.lower() else 0.0,
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        return np.array(features, dtype=np.float32)


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
            HiddenState.USER_QUERYING: 0.9,      # Increased - prioritize responding
            HiddenState.USER_URGENT: 0.3,
            HiddenState.USER_CONFUSED: 0.2,
            HiddenState.KNOWLEDGE_CERTAIN: 1.0,  # Highest priority
            HiddenState.KNOWLEDGE_UNCERTAIN: 0.3, # Lower - push toward certainty
            HiddenState.KNOWLEDGE_MISSING: 0.05,  # Very low - avoid this state
        }
        
        # Policy -> State transition likelihood (simplified)
        # P(s' | s, π) - how policies affect state
        # SEARCH-FIRST: Web search has highest certainty transition
        self.transition_model: Dict[PolicyType, Dict[HiddenState, float]] = {
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
            PolicyType.PRIMARY_SEARCH: 0.05,     # Lowest cost - default action
            PolicyType.WEB_SEARCH: 0.08,         # Very low cost
            PolicyType.WEB_BROWSE: 0.1,          # Low cost for more info
            
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
            PolicyType.SYSTEM_COMMAND: 0.8,      # High cost - requires confirmation
        }
    
    def calculate(
        self,
        policy: PolicyType,
        current_beliefs: BeliefState,
    ) -> Tuple[float, Dict[str, float]]:
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
            self.config.pragmatic_weight * pragmatic_value +
            self.config.epistemic_weight * epistemic_value +
            effort
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
            PolicyType.PRIMARY_SEARCH: -0.7,     # Best reduction
            PolicyType.WEB_SEARCH: -0.6,         # High reduction
            PolicyType.WEB_BROWSE: -0.55,        # Good reduction
            
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
        config: Optional[AgencyConfig] = None,
    ):
        """
        Initialize the Active Inference Controller.
        
        Args:
            config: Agency configuration
        """
        self.config = config or AgencyConfig()
        
        # Current belief state
        self.beliefs = BeliefState()
        self.beliefs.calculate_entropy()
        
        # Expected Free Energy calculator
        self.efe_calculator = ExpectedFreeEnergy(self.config)
        
        # Available policies - SEARCH-FIRST ordering
        self.available_policies = [
            # Search-First priorities
            PolicyType.PRIMARY_SEARCH,      # Default for unknown queries
            PolicyType.WEB_SEARCH,          # Specific search
            PolicyType.WEB_BROWSE,          # Deep information gathering
            
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
        self.action_history: List[Tuple[datetime, PolicyType, float]] = []
        self.max_history = 1000
        
        # Timing
        self.last_observation_time = datetime.now()
        self.last_action_time = datetime.now()
        self.last_thermal_check = datetime.now()
        self.last_self_check = datetime.now()
        
        # Callbacks
        self._action_callbacks: Dict[PolicyType, Callable] = {}
        
        # Running flag for continuous loop
        self._running = False
        
        # System command pending confirmation
        self._pending_system_command: Optional[Dict[str, Any]] = None
        
        logger.info(f"ActiveInferenceController initialized (Search-First: {config.search_first_enabled})")
    
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
                "what", "when", "where", "who", "how", "why",
                "is it true", "tell me", "explain", "describe",
                "latest", "news", "current", "today", "now"
            ]
            text_lower = observation.text.lower()
            if any(ind in text_lower for ind in factual_indicators):
                return True
        
        return False
    
    async def process_observation(
        self,
        observation: Observation,
    ) -> Tuple[PolicyType, Dict[str, Any]]:
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
        
        # 2. Check Search-First heuristic
        if self.should_search_first(observation):
            logger.info("Search-First heuristic triggered")
            # Bias toward search policies
            self.efe_calculator.effort_costs[PolicyType.PRIMARY_SEARCH] = 0.01
            self.efe_calculator.effort_costs[PolicyType.WEB_SEARCH] = 0.02
        
        # 3. Calculate Expected Free Energy for each policy
        policy_G: Dict[PolicyType, Tuple[float, Dict]] = {}
        
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
            self.action_history = self.action_history[-self.max_history:]
        
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
        """
        lr = self.config.belief_learning_rate
        
        # Update user intent beliefs based on observation
        if observation.text:
            text_lower = observation.text.lower()
            
            # Detect user intent
            if "?" in observation.text or any(w in text_lower for w in ["what", "how", "why"]):
                self._shift_belief(self.beliefs.user_intent, HiddenState.USER_QUERYING, lr)
            
            if any(w in text_lower for w in ["urgent", "asap", "immediately", "help"]):
                self._shift_belief(self.beliefs.user_intent, HiddenState.USER_URGENT, lr * 2)
            
            if any(w in text_lower for w in ["confused", "don't understand", "what do you mean"]):
                self._shift_belief(self.beliefs.user_intent, HiddenState.USER_CONFUSED, lr)
        else:
            # No text = likely idle
            self._shift_belief(self.beliefs.user_intent, HiddenState.USER_IDLE, lr * 0.5)
        
        # Update knowledge beliefs based on surprise
        if observation.surprise_signal > 1.5:
            # High surprise = uncertain knowledge
            self._shift_belief(self.beliefs.knowledge_state, HiddenState.KNOWLEDGE_UNCERTAIN, lr)
        elif observation.surprise_signal > 0.5:
            # Moderate surprise = somewhat certain
            self._shift_belief(self.beliefs.knowledge_state, HiddenState.KNOWLEDGE_CERTAIN, lr * 0.5)
        
        # Update interaction complexity
        if observation.query_complexity > 0.7:
            self._shift_belief(self.beliefs.interaction_state, HiddenState.INTERACTION_COMPLEX, lr)
        elif observation.query_complexity > 0.3:
            self._shift_belief(self.beliefs.interaction_state, HiddenState.INTERACTION_NOVEL, lr)
        
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
        distribution: Dict[HiddenState, float],
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
        policy_G: Dict[PolicyType, Tuple[float, Dict]],
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics."""
        # Count action frequencies
        action_counts: Dict[str, int] = {}
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
    
    def save_state(self, path: Optional[str] = None) -> None:
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
    
    def load_state(self, path: Optional[str] = None) -> None:
        """Load controller state from disk."""
        load_path = path or self.config.state_save_path
        
        if not Path(load_path).exists():
            logger.warning(f"No saved state at {load_path}")
            return
        
        import json
        with open(load_path, "r") as f:
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
    
    async def check_self_health(self) -> Dict[str, Any]:
        """
        Perform self-health check for self-preservation.
        
        Monitors:
        - Memory usage
        - Process health
        - Response times
        
        Returns:
            Health status dictionary
        """
        import psutil
        import os
        
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
                    (self.action_history[i][0] - self.action_history[i-1][0]).total_seconds()
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
            "os.system", "subprocess", "exec(", "eval(",
            "import os", "shell=", "cmd /c", "powershell",
            "sudo", "chmod", "chown", "rm -rf"
        ]
        
        return any(pattern in text_lower for pattern in shell_patterns)
    
    def request_system_command_confirmation(
        self,
        command: str,
        reason: str,
    ) -> Dict[str, Any]:
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
