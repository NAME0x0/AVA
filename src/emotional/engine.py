"""
Emotional Engine for AVA

Central system for processing emotional triggers, managing emotional state,
and providing emotional modulation factors for other systems.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .models import (
    EmotionVector,
    EmotionalState,
    EmotionalTrigger,
    TriggerType,
    create_trigger,
)

logger = logging.getLogger(__name__)


class EmotionalEngine:
    """
    Manages AVA's emotional state and processes emotional events.

    The engine:
    - Processes triggers that cause emotional changes
    - Applies time-based decay toward baseline emotions
    - Provides modulation factors for learning, responses, and tool selection
    - Maintains emotional history for trend analysis
    """

    def __init__(
        self,
        data_path: str = "data/emotional",
        stage_volatility: float = 1.0,
    ):
        """
        Initialize the emotional engine.

        Args:
            data_path: Path for persisting emotional state
            stage_volatility: Developmental stage volatility multiplier
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)

        self.stage_volatility = stage_volatility
        self.state = self._load_state()

        logger.info(
            f"EmotionalEngine initialized: "
            f"Dominant={self.state.current.get_dominant_emotion().value}, "
            f"Valence={self.state.current.compute_valence():.2f}"
        )

    def _load_state(self) -> EmotionalState:
        """Load emotional state from disk or create new."""
        state_file = self.data_path / "state.json"
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    data = json.load(f)
                    return EmotionalState.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load emotional state: {e}")
                return EmotionalState()
        return EmotionalState()

    def save_state(self):
        """Save emotional state to disk."""
        state_file = self.data_path / "state.json"
        with open(state_file, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)

    def set_stage_volatility(self, volatility: float):
        """
        Update the stage volatility multiplier.

        Called by DevelopmentTracker when stage changes.
        Higher volatility = bigger emotional swings.
        """
        self.stage_volatility = volatility

    def process_trigger(self, trigger: EmotionalTrigger) -> EmotionVector:
        """
        Process an emotional trigger and update state.

        Args:
            trigger: The emotional trigger to process

        Returns:
            The updated emotion vector
        """
        # Calculate effective deltas with intensity and volatility
        effective_multiplier = trigger.intensity
        if trigger.apply_stage_modifier:
            effective_multiplier *= self.stage_volatility

        # Apply deltas to current emotions
        for emotion, delta in trigger.emotion_deltas.items():
            current_value = getattr(self.state.current, emotion, None)
            if current_value is not None:
                new_value = current_value + (delta * effective_multiplier)
                setattr(self.state.current, emotion, new_value)

        # Normalize to keep within bounds
        self.state.current.normalize()

        # Update mood and record to history
        self.state.update_mood()
        self.state.record_to_history()
        self.state.last_updated = datetime.now()

        self.save_state()

        logger.debug(
            f"Processed trigger {trigger.trigger_type.value}: "
            f"New dominant={self.state.current.get_dominant_emotion().value}"
        )

        return self.state.current

    def apply_decay(self, elapsed_seconds: float):
        """
        Apply time-based decay toward baseline emotions.

        Should be called at the start of each interaction to simulate
        emotional return to baseline over time.

        Args:
            elapsed_seconds: Seconds since last interaction
        """
        if elapsed_seconds <= 0:
            return

        elapsed_minutes = elapsed_seconds / 60

        # Apply decay for each emotion
        for emotion, decay_rate in self.state.decay_rates.items():
            current_value = getattr(self.state.current, emotion)
            baseline_value = getattr(self.state.baseline, emotion)

            # Exponential decay toward baseline
            decay_amount = decay_rate * elapsed_minutes
            difference = current_value - baseline_value

            if abs(difference) > 0.01:
                # Move toward baseline
                if difference > 0:
                    new_value = max(baseline_value, current_value - decay_amount)
                else:
                    new_value = min(baseline_value, current_value + decay_amount)
                setattr(self.state.current, emotion, new_value)

        self.state.current.normalize()
        self.state.last_updated = datetime.now()

    def trigger_success(self, intensity: float = 1.0, context: Optional[Dict] = None):
        """Convenience method to trigger success emotion."""
        trigger = create_trigger(TriggerType.SUCCESS, intensity, context=context)
        return self.process_trigger(trigger)

    def trigger_failure(self, intensity: float = 1.0, context: Optional[Dict] = None):
        """Convenience method to trigger failure emotion."""
        trigger = create_trigger(TriggerType.FAILURE, intensity, context=context)
        return self.process_trigger(trigger)

    def trigger_praise(self, intensity: float = 1.0, context: Optional[Dict] = None):
        """Convenience method to trigger praise emotion."""
        trigger = create_trigger(TriggerType.PRAISE, intensity, context=context)
        return self.process_trigger(trigger)

    def trigger_novelty(self, intensity: float = 1.0, context: Optional[Dict] = None):
        """Convenience method to trigger novelty/surprise emotion."""
        trigger = create_trigger(TriggerType.NOVELTY, intensity, context=context)
        return self.process_trigger(trigger)

    def trigger_challenge(self, intensity: float = 1.0, context: Optional[Dict] = None):
        """Convenience method to trigger challenge emotion."""
        trigger = create_trigger(TriggerType.CHALLENGE, intensity, context=context)
        return self.process_trigger(trigger)

    def trigger_completion(self, intensity: float = 1.0, context: Optional[Dict] = None):
        """Convenience method to trigger completion/milestone emotion."""
        trigger = create_trigger(TriggerType.COMPLETION, intensity, context=context)
        return self.process_trigger(trigger)

    def get_current_emotions(self) -> EmotionVector:
        """Get the current emotion vector."""
        return self.state.current

    def get_emotional_state(self) -> EmotionalState:
        """Get the complete emotional state."""
        return self.state

    def get_learning_rate_modifier(self) -> float:
        """
        Calculate learning rate modifier based on emotional state.

        Returns a multiplier (typically 0.5 to 1.5) for the base learning rate.

        - High hope/joy/ambition -> increased learning
        - High fear -> decreased learning (protective)
        - High surprise -> temporary boost (novelty bonus)
        """
        emotions = self.state.current

        # Positive contributions
        positive_factor = (
            emotions.hope * 0.3 +
            emotions.joy * 0.3 +
            emotions.ambition * 0.4
        )

        # Negative contributions
        fear_penalty = emotions.fear * 0.3

        # Surprise bonus (novelty enhances learning)
        surprise_bonus = emotions.surprise * 0.2

        # Calculate final modifier (centered around 1.0)
        modifier = 0.7 + positive_factor - fear_penalty + surprise_bonus

        # Clamp to reasonable range
        return max(0.5, min(1.5, modifier))

    def get_response_modulation(self) -> Dict[str, float]:
        """
        Get modulation factors for response generation.

        Returns factors that affect:
        - temperature: Randomness/creativity
        - verbosity: Response length tendency
        - confidence: Assertiveness of statements
        - warmth: Friendliness of tone
        """
        emotions = self.state.current
        valence = emotions.compute_valence()
        arousal = emotions.compute_arousal()

        return {
            # High joy/surprise -> more creative, high fear -> more conservative
            "temperature": 0.8 + (emotions.joy * 0.2) + (emotions.surprise * 0.1) - (emotions.fear * 0.2),

            # High ambition -> longer responses, high fear -> shorter
            "verbosity": 1.0 + (emotions.ambition * 0.2) - (emotions.fear * 0.15),

            # High hope/joy -> more confident, high fear -> less confident
            "confidence": 0.7 + (emotions.hope * 0.2) + (emotions.joy * 0.1) - (emotions.fear * 0.3),

            # Positive valence -> warmer tone
            "warmth": 0.5 + (valence * 0.3) + (emotions.joy * 0.2),

            # Overall arousal affects energy level
            "energy": 0.5 + arousal,
        }

    def get_tool_selection_bias(self) -> Dict[str, float]:
        """
        Get biases for tool selection based on emotional state.

        Returns factors that influence which tools AVA prefers:
        - safe_preference: Tendency toward safer tools
        - exploration: Willingness to try new/unknown tools
        - complexity: Preference for complex vs simple tools
        """
        emotions = self.state.current

        # Fear -> prefer safe tools
        safe_preference = 0.5 + (emotions.fear * 0.4) - (emotions.ambition * 0.2)

        # Hope + Surprise -> exploration
        exploration = 0.3 + (emotions.hope * 0.3) + (emotions.surprise * 0.3) - (emotions.fear * 0.2)

        # Ambition -> complex tools
        complexity = 0.3 + (emotions.ambition * 0.4) + (emotions.joy * 0.1) - (emotions.fear * 0.2)

        return {
            "safe_preference": max(0.0, min(1.0, safe_preference)),
            "exploration": max(0.0, min(1.0, exploration)),
            "complexity_preference": max(0.0, min(1.0, complexity)),
        }

    def should_accelerate_learning(self) -> bool:
        """
        Check if emotional state suggests accelerated learning.

        Returns True if emotions indicate a good time for intensive learning
        (high positive emotions, low fear).
        """
        emotions = self.state.current
        positive_score = (emotions.hope + emotions.joy + emotions.ambition) / 3
        return positive_score > 0.6 and emotions.fear < 0.4

    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of current emotional status."""
        emotions = self.state.current
        derived = emotions.get_derived_emotions()

        return {
            "primary_emotions": emotions.to_dict(),
            "derived_emotions": {k.value: v for k, v in derived.items()},
            "dominant_emotion": emotions.get_dominant_emotion().value,
            "valence": emotions.compute_valence(),
            "arousal": emotions.compute_arousal(),
            "mood": {
                "valence": self.state.mood_valence,
                "arousal": self.state.mood_arousal,
            },
            "learning_rate_modifier": self.get_learning_rate_modifier(),
            "response_modulation": self.get_response_modulation(),
            "tool_bias": self.get_tool_selection_bias(),
            "accelerate_learning": self.should_accelerate_learning(),
            "stage_volatility": self.stage_volatility,
        }

    def reset_to_baseline(self):
        """Reset current emotions to baseline."""
        self.state.current = self.state.baseline.copy()
        self.state.last_updated = datetime.now()
        self.save_state()
        logger.info("Emotional state reset to baseline")
