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

logger = logging.getLogger(__name__)


class PolicyType(Enum):
    """Available action policies for the agent."""
    
    # Immediate actions
    REFLEX_REPLY = auto()       # Quick Medulla response
    ACKNOWLEDGE = auto()        # Phatic acknowledgment
    
    # Reasoning actions
    DEEP_THOUGHT = auto()       # Invoke Cortex for reasoning
    CHAIN_OF_THOUGHT = auto()   # Extended thinking
    
    # Information gathering
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
    """
    
    # Free Energy Thresholds
    action_threshold: float = 0.3         # Min G reduction to act
    urgency_threshold: float = 0.7        # High urgency triggers immediate action
    
    # Time Constants (seconds)
    idle_uncertainty_rate: float = 0.01   # Uncertainty growth per second of silence
    max_wait_time: float = 300.0          # Max seconds before proactive action
    
    # Policy Weights (Pragmatic vs Epistemic)
    pragmatic_weight: float = 0.6         # Weight for goal achievement
    epistemic_weight: float = 0.4         # Weight for information gain
    
    # Cortex Activation Cost
    cortex_effort_cost: float = 0.5       # Penalty for invoking Cortex
    tool_effort_cost: float = 0.2         # Penalty for tool use
    
    # Learning
    belief_learning_rate: float = 0.1     # How fast beliefs update
    preference_adaptation: bool = True     # Adapt preferences over time
    
    # State Persistence
    state_save_path: str = "data/memory/agency_state.pkl"
    
    # Observation Categories
    num_observation_modalities: int = 4   # text, audio, system, time


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
    """
    
    def __init__(self, config: AgencyConfig):
        self.config = config
        
        # Define preferred observations (C matrix)
        # Higher values = more preferred
        self.preferences = {
            HiddenState.USER_IDLE: 0.5,
            HiddenState.USER_QUERYING: 0.8,
            HiddenState.USER_URGENT: 0.3,
            HiddenState.USER_CONFUSED: 0.2,
            HiddenState.KNOWLEDGE_CERTAIN: 1.0,
            HiddenState.KNOWLEDGE_UNCERTAIN: 0.4,
            HiddenState.KNOWLEDGE_MISSING: 0.1,
        }
        
        # Policy -> State transition likelihood (simplified)
        # P(s' | s, π) - how policies affect state
        self.transition_model: Dict[PolicyType, Dict[HiddenState, float]] = {
            PolicyType.DEEP_THOUGHT: {
                HiddenState.KNOWLEDGE_CERTAIN: 0.7,
                HiddenState.KNOWLEDGE_UNCERTAIN: 0.2,
            },
            PolicyType.USE_TOOL: {
                HiddenState.KNOWLEDGE_CERTAIN: 0.6,
                HiddenState.KNOWLEDGE_MISSING: 0.1,
            },
            PolicyType.ASK_CLARIFICATION: {
                HiddenState.USER_CONFUSED: 0.2,
                HiddenState.KNOWLEDGE_UNCERTAIN: 0.4,
            },
            PolicyType.WAIT: {
                # Waiting increases uncertainty
                HiddenState.KNOWLEDGE_UNCERTAIN: 0.6,
            },
        }
        
        # Policy effort costs
        self.effort_costs = {
            PolicyType.DEEP_THOUGHT: config.cortex_effort_cost,
            PolicyType.USE_TOOL: config.tool_effort_cost,
            PolicyType.CHAIN_OF_THOUGHT: config.cortex_effort_cost * 0.5,
            PolicyType.REFLEX_REPLY: 0.05,
            PolicyType.WAIT: 0.0,
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
        """
        # Policies that gather information reduce ambiguity
        information_gathering = {
            PolicyType.ASK_CLARIFICATION: -0.5,  # Reduces uncertainty
            PolicyType.QUERY_MEMORY: -0.3,
            PolicyType.SCAN_ENVIRONMENT: -0.2,
            PolicyType.DEEP_THOUGHT: -0.4,
            PolicyType.WAIT: 0.2,  # Increases uncertainty
            PolicyType.SLEEP: 0.3,
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
        
        # Available policies
        self.available_policies = [
            PolicyType.REFLEX_REPLY,
            PolicyType.DEEP_THOUGHT,
            PolicyType.USE_TOOL,
            PolicyType.ASK_CLARIFICATION,
            PolicyType.WAIT,
            PolicyType.QUERY_MEMORY,
            PolicyType.SCAN_ENVIRONMENT,
        ]
        
        # Action history
        self.action_history: List[Tuple[datetime, PolicyType, float]] = []
        self.max_history = 1000
        
        # Timing
        self.last_observation_time = datetime.now()
        self.last_action_time = datetime.now()
        
        # Callbacks
        self._action_callbacks: Dict[PolicyType, Callable] = {}
        
        # Running flag for continuous loop
        self._running = False
        
        logger.info(f"ActiveInferenceController initialized")
    
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
    
    async def process_observation(
        self,
        observation: Observation,
    ) -> Tuple[PolicyType, Dict[str, Any]]:
        """
        Process an observation and select the optimal policy.
        
        This is the main inference step that:
        1. Updates beliefs based on observation
        2. Calculates G for each policy
        3. Selects the policy with minimum G
        4. Executes the corresponding action
        
        Args:
            observation: Current observation
            
        Returns:
            Tuple of (selected policy, action result)
        """
        # 1. Update beliefs based on observation
        await self._update_beliefs(observation)
        
        # 2. Calculate Expected Free Energy for each policy
        policy_G: Dict[PolicyType, Tuple[float, Dict]] = {}
        
        for policy in self.available_policies:
            G, breakdown = self.efe_calculator.calculate(policy, self.beliefs)
            policy_G[policy] = (G, breakdown)
        
        # 3. Select policy with minimum G (softmax selection for exploration)
        selected_policy = self._select_policy(policy_G)
        
        # 4. Record action
        G_value = policy_G[selected_policy][0]
        self.action_history.append((datetime.now(), selected_policy, G_value))
        
        if len(self.action_history) > self.max_history:
            self.action_history = self.action_history[-self.max_history:]
        
        # 5. Execute action callback if registered
        result = {"policy": selected_policy.name, "G": G_value}
        
        if selected_policy in self._action_callbacks:
            callback = self._action_callbacks[selected_policy]
            try:
                action_result = await callback(observation, self.beliefs)
                result["action_result"] = action_result
            except Exception as e:
                logger.error(f"Action callback failed: {e}")
                result["error"] = str(e)
        
        # 6. Update timing
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
