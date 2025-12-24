"""
ENTROPIX: The Metacognitive Module
==================================

Measures the "texture of thought" via Entropy (H) and Varentropy (V)
to determine the cognitive state of the LLM during inference.

Based on Entropix research (2024-2025):
- Low H, Low V = FLOW (confident, rote knowledge)
- Low H, High V = HESITATION (mostly sure, competing alternatives)
- High H, Low V = CONFUSION (uniformly uncertain, hallucination risk)
- High H, High V = CREATIVE (many valid paths, divergent thinking)

This module enables AVA to:
1. Detect when it's confused and should use tools
2. Trigger chain-of-thought when hesitating
3. Adjust sampling parameters dynamically
4. Provide a "surprise" signal to the Titans Neural Memory

Reference: Entropix sampling, xjdr/entropix (2024)
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class CognitiveStateLabel(Enum):
    """Labels for cognitive states based on entropy/varentropy quadrants."""

    FLOW = "flow"  # Low H, Low V - Confident, fast execution
    HESITATION = "hesitation"  # Low H, High V - Mostly sure, some conflict
    CONFUSION = "confusion"  # High H, Low V - Uniformly uncertain
    CREATIVE = "creative"  # High H, High V - Divergent, exploratory
    NEUTRAL = "neutral"  # Default/unknown state


@dataclass
class CognitiveState:
    """
    Current cognitive state of the model.

    This represents the model's "mental state" during inference,
    derived from analysis of the token probability distribution.
    """

    label: CognitiveStateLabel = CognitiveStateLabel.NEUTRAL
    entropy: float = 0.0  # Shannon entropy H(X)
    varentropy: float = 0.0  # Variance of entropy V(X)
    confidence: float = 1.0  # Derived confidence (inverse of normalized entropy)

    # Action recommendations
    should_use_tools: bool = False
    should_use_cot: bool = False
    should_increase_temp: bool = False
    recommended_temp: float = 0.7

    # Raw data for debugging
    top_tokens: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "label": self.label.value,
            "entropy": self.entropy,
            "varentropy": self.varentropy,
            "confidence": self.confidence,
            "should_use_tools": self.should_use_tools,
            "should_use_cot": self.should_use_cot,
            "should_increase_temp": self.should_increase_temp,
            "recommended_temp": self.recommended_temp,
        }

    def __repr__(self) -> str:
        return (
            f"CognitiveState({self.label.value}, "
            f"H={self.entropy:.3f}, V={self.varentropy:.3f}, "
            f"conf={self.confidence:.2f})"
        )


@dataclass
class EntropixConfig:
    """Configuration for Entropix thresholds and behavior."""

    # Entropy thresholds (based on empirical tuning)
    low_entropy_threshold: float = 0.5  # Below this = very confident
    high_entropy_threshold: float = 3.0  # Above this = confused/creative

    # Varentropy thresholds
    low_varentropy_threshold: float = 0.5  # Below this = uniform distribution
    high_varentropy_threshold: float = 2.0  # Above this = competing peaks

    # Temperature adjustments
    base_temperature: float = 0.7
    confusion_temperature: float = 0.3  # Lower temp when confused
    creative_temperature: float = 1.2  # Higher temp for creativity

    # Tool/CoT triggers
    confusion_triggers_tools: bool = True
    hesitation_triggers_cot: bool = True

    # Surprise signal scaling (for Titans memory)
    surprise_scale: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "low_entropy_threshold": self.low_entropy_threshold,
            "high_entropy_threshold": self.high_entropy_threshold,
            "low_varentropy_threshold": self.low_varentropy_threshold,
            "high_varentropy_threshold": self.high_varentropy_threshold,
            "base_temperature": self.base_temperature,
            "confusion_temperature": self.confusion_temperature,
            "creative_temperature": self.creative_temperature,
        }


class Entropix:
    """
    The Entropix Controller - Metacognitive Awareness.

    This class analyzes the probability distribution of LLM outputs
    to determine the model's cognitive state and recommend actions.

    Key formulas:
    - Entropy: H(X) = -∑ p(x) * log(p(x))
    - Varentropy: V(X) = ∑ p(x) * (log(p(x)) + H(X))²

    Usage:
        entropix = Entropix()
        state = entropix.diagnose(logprobs)
        if state.should_use_tools:
            # Trigger tool execution
        elif state.should_use_cot:
            # Enable chain-of-thought
    """

    def __init__(self, config: EntropixConfig | None = None):
        """
        Initialize Entropix with configuration.

        Args:
            config: Entropix configuration (uses defaults if None)
        """
        self.config = config or EntropixConfig()

        # Statistics tracking
        self.history: list[CognitiveState] = []
        self.max_history = 100

    def calculate_metrics(
        self,
        logprobs: list[dict[str, Any]],
    ) -> tuple[float, float]:
        """
        Calculate Entropy (H) and Varentropy (V) from log probabilities.

        The logprobs should be a list of dictionaries containing:
        - "token": The token string
        - "logprob": The log probability (natural log)

        Ollama format: [{"token": "Hello", "logprob": -0.5}, ...]

        Args:
            logprobs: List of token log probability dictionaries

        Returns:
            Tuple of (entropy, varentropy)
        """
        if not logprobs:
            return 0.0, 0.0

        # Extract log probabilities
        try:
            lps = []
            for lp in logprobs:
                if isinstance(lp, dict):
                    log_prob = lp.get("logprob", lp.get("log_prob", 0.0))
                else:
                    log_prob = float(lp)
                lps.append(log_prob)

            if not lps:
                return 0.0, 0.0

            # Convert to probabilities (exp of log probs)
            probs = np.array([math.exp(lp) for lp in lps])

            # Normalize (Ollama returns top_k, not full vocab)
            total = np.sum(probs)
            if total <= 0:
                return 0.0, 0.0
            probs = probs / total

            # Filter out zeros to avoid log(0)
            probs = probs[probs > 0]
            if len(probs) == 0:
                return 0.0, 0.0

            # Calculate Entropy: H(X) = -∑ p(x) * log(p(x))
            log_probs = np.log(probs)
            entropy = -np.sum(probs * log_probs)

            # Calculate Varentropy: V(X) = ∑ p(x) * (log(p(x)) + H)²
            varentropy = np.sum(probs * (log_probs + entropy) ** 2)

            return float(entropy), float(varentropy)

        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Error calculating entropy metrics: {e}")
            return 0.0, 0.0

    def _classify_state(
        self,
        entropy: float,
        varentropy: float,
    ) -> CognitiveStateLabel:
        """
        Classify cognitive state based on entropy/varentropy quadrant.

        The Entropix Quadrants:

                    Low Varentropy    High Varentropy
                    ─────────────────────────────────
        Low H    │    FLOW          │   HESITATION   │
                 │  (confident)     │  (conflicted)  │
                    ─────────────────────────────────
        High H   │   CONFUSION      │    CREATIVE    │
                 │  (uncertain)     │  (exploratory) │
                    ─────────────────────────────────

        Args:
            entropy: Shannon entropy value
            varentropy: Variance of entropy value

        Returns:
            CognitiveStateLabel enum value
        """
        cfg = self.config

        if entropy < cfg.low_entropy_threshold:
            # Low entropy = confident
            return CognitiveStateLabel.FLOW

        elif entropy < cfg.high_entropy_threshold:
            # Medium entropy - check varentropy
            if varentropy > cfg.high_varentropy_threshold:
                return CognitiveStateLabel.HESITATION
            else:
                return CognitiveStateLabel.NEUTRAL

        else:
            # High entropy
            if varentropy > cfg.high_varentropy_threshold:
                return CognitiveStateLabel.CREATIVE
            else:
                return CognitiveStateLabel.CONFUSION

    def diagnose(
        self,
        logprobs: list[dict[str, Any]],
    ) -> CognitiveState:
        """
        Diagnose the cognitive state from log probabilities.

        This is the main entry point for metacognitive analysis.

        Args:
            logprobs: Log probabilities from LLM (Ollama format)

        Returns:
            CognitiveState with classification and recommendations
        """
        # Calculate metrics
        entropy, varentropy = self.calculate_metrics(logprobs)

        # Classify state
        label = self._classify_state(entropy, varentropy)

        # Calculate confidence (inverse of normalized entropy)
        # Assuming max entropy around 5.0 for typical vocab
        confidence = max(0.0, min(1.0, 1.0 - (entropy / 5.0)))

        # Determine action recommendations
        cfg = self.config

        should_use_tools = cfg.confusion_triggers_tools and label == CognitiveStateLabel.CONFUSION

        should_use_cot = cfg.hesitation_triggers_cot and label in [
            CognitiveStateLabel.HESITATION,
            CognitiveStateLabel.CONFUSION,
        ]

        should_increase_temp = label == CognitiveStateLabel.CREATIVE

        # Recommend temperature
        if label == CognitiveStateLabel.CONFUSION:
            recommended_temp = cfg.confusion_temperature
        elif label == CognitiveStateLabel.CREATIVE:
            recommended_temp = cfg.creative_temperature
        else:
            recommended_temp = cfg.base_temperature

        # Create state object
        state = CognitiveState(
            label=label,
            entropy=entropy,
            varentropy=varentropy,
            confidence=confidence,
            should_use_tools=should_use_tools,
            should_use_cot=should_use_cot,
            should_increase_temp=should_increase_temp,
            recommended_temp=recommended_temp,
            top_tokens=logprobs[:5] if logprobs else [],
        )

        # Track history
        self.history.append(state)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

        logger.debug(f"Entropix diagnosis: {state}")

        return state

    def get_surprise_signal(
        self,
        state: CognitiveState | None = None,
    ) -> float:
        """
        Get a surprise signal for the Titans Neural Memory.

        Higher entropy = higher surprise = stronger memory update.
        This connects the metacognitive analysis to the memory system.

        Args:
            state: CognitiveState to extract signal from (uses latest if None)

        Returns:
            Surprise value (0.0 to ~5.0, scaled)
        """
        if state is None:
            state = self.history[-1] if self.history else None

        if state is None:
            return 0.0

        # Scale entropy by config factor
        surprise = state.entropy * self.config.surprise_scale

        return surprise

    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about recent cognitive states.

        Returns:
            Dictionary with state distribution and averages
        """
        if not self.history:
            return {
                "total_diagnoses": 0,
                "avg_entropy": 0.0,
                "avg_varentropy": 0.0,
                "state_distribution": {},
            }

        # Count state distribution
        state_counts: dict[str, int] = {}
        for state in self.history:
            label = state.label.value
            state_counts[label] = state_counts.get(label, 0) + 1

        # Calculate averages
        avg_entropy = np.mean([s.entropy for s in self.history])
        avg_varentropy = np.mean([s.varentropy for s in self.history])
        avg_confidence = np.mean([s.confidence for s in self.history])

        return {
            "total_diagnoses": len(self.history),
            "avg_entropy": float(avg_entropy),
            "avg_varentropy": float(avg_varentropy),
            "avg_confidence": float(avg_confidence),
            "state_distribution": state_counts,
            "tool_trigger_rate": sum(1 for s in self.history if s.should_use_tools)
            / len(self.history),
            "cot_trigger_rate": sum(1 for s in self.history if s.should_use_cot)
            / len(self.history),
        }

    def reset_history(self):
        """Clear the cognitive state history."""
        self.history = []


# =============================================================================
# Utility Functions
# =============================================================================


def diagnose_from_ollama_response(
    response: dict[str, Any],
    config: EntropixConfig | None = None,
) -> CognitiveState:
    """
    Convenience function to diagnose from an Ollama API response.

    Args:
        response: Full Ollama generate/chat response
        config: Optional Entropix configuration

    Returns:
        CognitiveState diagnosis
    """
    entropix = Entropix(config)

    # Extract logprobs from Ollama response
    # Ollama format: response["logprobs"] contains list of token info
    logprobs = response.get("logprobs", [])

    if not logprobs:
        logger.warning("No logprobs in Ollama response - returning neutral state")
        return CognitiveState()

    # Handle nested structure if present
    if isinstance(logprobs, dict) and "top_logprobs" in logprobs:
        logprobs = logprobs["top_logprobs"]
    elif isinstance(logprobs, list) and logprobs:
        # May be list of lists (per-token top_logprobs)
        if isinstance(logprobs[0], list):
            # Flatten to first token's top candidates
            logprobs = logprobs[0]

    return entropix.diagnose(logprobs)


def create_entropix_controller(
    entropy_threshold: float = 3.0,
    varentropy_threshold: float = 2.0,
    enable_tools: bool = True,
    enable_cot: bool = True,
) -> Entropix:
    """
    Factory function to create a configured Entropix controller.

    Args:
        entropy_threshold: High entropy threshold
        varentropy_threshold: High varentropy threshold
        enable_tools: Whether confusion triggers tool use
        enable_cot: Whether hesitation triggers CoT

    Returns:
        Configured Entropix instance
    """
    config = EntropixConfig(
        high_entropy_threshold=entropy_threshold,
        high_varentropy_threshold=varentropy_threshold,
        confusion_triggers_tools=enable_tools,
        hesitation_triggers_cot=enable_cot,
    )

    return Entropix(config)
