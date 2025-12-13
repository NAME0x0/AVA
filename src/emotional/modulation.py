"""
Emotional Modulation Functions for AVA

Standalone functions for calculating how emotions modulate various
aspects of AVA's behavior. These can be used independently or through
the EmotionalEngine.
"""

from typing import Dict
from .models import EmotionVector


def get_learning_rate_modifier(emotions: EmotionVector) -> float:
    """
    Calculate learning rate modifier from emotion vector.

    Args:
        emotions: Current emotion vector

    Returns:
        Multiplier for base learning rate (0.5 to 1.5)
    """
    # Positive contributions to learning
    positive_factor = (
        emotions.hope * 0.3 +
        emotions.joy * 0.3 +
        emotions.ambition * 0.4
    )

    # Fear inhibits learning
    fear_penalty = emotions.fear * 0.3

    # Surprise/novelty temporarily boosts learning
    surprise_bonus = emotions.surprise * 0.2

    # Base modifier around 1.0
    modifier = 0.7 + positive_factor - fear_penalty + surprise_bonus

    return max(0.5, min(1.5, modifier))


def get_response_modulation(emotions: EmotionVector) -> Dict[str, float]:
    """
    Calculate response generation modulation factors.

    Args:
        emotions: Current emotion vector

    Returns:
        Dictionary of modulation factors for response generation
    """
    valence = emotions.compute_valence()
    arousal = emotions.compute_arousal()

    return {
        # Temperature affects creativity/randomness
        "temperature": max(0.3, min(1.5,
            0.8 + (emotions.joy * 0.2) +
            (emotions.surprise * 0.1) -
            (emotions.fear * 0.2)
        )),

        # Verbosity affects response length
        "verbosity": max(0.5, min(1.5,
            1.0 + (emotions.ambition * 0.2) -
            (emotions.fear * 0.15)
        )),

        # Confidence affects assertiveness
        "confidence": max(0.3, min(1.0,
            0.7 + (emotions.hope * 0.2) +
            (emotions.joy * 0.1) -
            (emotions.fear * 0.3)
        )),

        # Warmth affects friendliness
        "warmth": max(0.2, min(1.0,
            0.5 + (valence * 0.3) +
            (emotions.joy * 0.2)
        )),

        # Energy affects enthusiasm
        "energy": max(0.3, min(1.0,
            0.5 + arousal
        )),

        # Caution affects hedging/qualifications
        "caution": max(0.0, min(1.0,
            emotions.fear * 0.6 -
            (emotions.hope * 0.2) -
            (emotions.ambition * 0.1)
        )),
    }


def get_tool_selection_bias(emotions: EmotionVector) -> Dict[str, float]:
    """
    Calculate tool selection biases from emotions.

    Args:
        emotions: Current emotion vector

    Returns:
        Dictionary of biases affecting tool choice
    """
    return {
        # Preference for safe/known tools
        "safe_preference": max(0.0, min(1.0,
            0.5 + (emotions.fear * 0.4) -
            (emotions.ambition * 0.2) -
            (emotions.hope * 0.1)
        )),

        # Willingness to try new tools
        "exploration": max(0.0, min(1.0,
            0.3 + (emotions.hope * 0.3) +
            (emotions.surprise * 0.3) +
            (emotions.ambition * 0.2) -
            (emotions.fear * 0.3)
        )),

        # Preference for complex vs simple tools
        "complexity_preference": max(0.0, min(1.0,
            0.3 + (emotions.ambition * 0.4) +
            (emotions.joy * 0.1) -
            (emotions.fear * 0.3)
        )),

        # Risk tolerance in tool usage
        "risk_tolerance": max(0.0, min(1.0,
            0.4 + (emotions.hope * 0.2) +
            (emotions.ambition * 0.2) -
            (emotions.fear * 0.4)
        )),
    }


def get_thinking_modulation(emotions: EmotionVector) -> Dict[str, float]:
    """
    Calculate thinking/reasoning modulation factors.

    Args:
        emotions: Current emotion vector

    Returns:
        Dictionary of factors affecting test-time compute
    """
    return {
        # Budget multiplier for thinking tokens
        "budget_multiplier": max(0.5, min(1.5,
            1.0 + (emotions.ambition * 0.3) +
            (emotions.fear * 0.2) -  # Fear increases deliberation
            (emotions.joy * 0.1)     # High joy can be impulsive
        )),

        # Depth of self-reflection
        "reflection_depth": max(0.3, min(1.0,
            0.5 + (emotions.fear * 0.2) +
            (emotions.ambition * 0.2) -
            (emotions.surprise * 0.1)
        )),

        # Tendency to second-guess
        "self_doubt": max(0.0, min(1.0,
            emotions.fear * 0.4 -
            (emotions.hope * 0.2) -
            (emotions.joy * 0.1)
        )),
    }


def get_articulation_modifier(emotions: EmotionVector) -> float:
    """
    Calculate modifier for articulation clarity.

    High negative emotions can impair articulation (like being flustered).

    Args:
        emotions: Current emotion vector

    Returns:
        Multiplier for articulation clarity (0.7 to 1.1)
    """
    # Extreme emotions impair articulation
    arousal = emotions.compute_arousal()
    fear_penalty = emotions.fear * 0.1
    surprise_penalty = emotions.surprise * 0.1

    # Joy and calm help articulation
    joy_bonus = emotions.joy * 0.05

    modifier = 1.0 - fear_penalty - surprise_penalty + joy_bonus

    # High arousal can impair articulation
    if arousal > 0.7:
        modifier -= (arousal - 0.7) * 0.2

    return max(0.7, min(1.1, modifier))


def compute_emotional_response_to_outcome(
    outcome: str,
    quality_score: float,
    was_challenging: bool = False,
    user_feedback: str | None = None,
) -> Dict[str, float]:
    """
    Compute emotional deltas based on interaction outcome.

    Args:
        outcome: "success", "partial", or "failure"
        quality_score: Quality of the interaction (0.0 to 1.0)
        was_challenging: Whether the task was challenging
        user_feedback: Optional user feedback ("positive", "negative", "neutral")

    Returns:
        Dictionary of emotion deltas to apply
    """
    deltas: Dict[str, float] = {}

    # Base deltas from outcome
    if outcome == "success":
        deltas = {
            "hope": 0.05 + (quality_score * 0.05),
            "joy": 0.1 + (quality_score * 0.1),
            "fear": -0.05,
            "ambition": 0.02,
        }
        if was_challenging:
            deltas["joy"] += 0.1
            deltas["ambition"] += 0.05

    elif outcome == "partial":
        deltas = {
            "hope": 0.0,
            "joy": 0.02,
            "ambition": 0.02,
        }

    elif outcome == "failure":
        deltas = {
            "hope": -0.05,
            "joy": -0.1,
            "fear": 0.1,
            "ambition": -0.02,
        }

    # Modify based on user feedback
    if user_feedback == "positive":
        deltas["hope"] = deltas.get("hope", 0) + 0.1
        deltas["joy"] = deltas.get("joy", 0) + 0.15
        deltas["fear"] = deltas.get("fear", 0) - 0.1
        deltas["ambition"] = deltas.get("ambition", 0) + 0.05

    elif user_feedback == "negative":
        deltas["hope"] = deltas.get("hope", 0) - 0.05
        deltas["joy"] = deltas.get("joy", 0) - 0.1
        deltas["fear"] = deltas.get("fear", 0) + 0.1

    return deltas
