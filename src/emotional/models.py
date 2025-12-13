"""
Emotional Models for AVA

Defines the data structures for representing emotional states, triggers,
and emotional history.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class EmotionType(Enum):
    """
    Core emotions for AVA.

    Based on a simplified emotional model focusing on emotions that
    meaningfully affect AI behavior and learning.
    """
    # Primary emotions
    HOPE = "hope"           # Anticipation of positive outcomes
    FEAR = "fear"           # Anticipation of negative outcomes
    JOY = "joy"             # Response to positive events
    SURPRISE = "surprise"   # Response to unexpected events
    AMBITION = "ambition"   # Drive toward goals/improvement

    # Derived emotions (computed from primary)
    CURIOSITY = "curiosity"       # Hope + Surprise
    ANXIETY = "anxiety"           # Fear + Ambition
    CONTENTMENT = "contentment"   # Joy + low Ambition
    FRUSTRATION = "frustration"   # Ambition + Fear
    EXCITEMENT = "excitement"     # Joy + Surprise + Ambition


class TriggerType(Enum):
    """Types of events that trigger emotional responses."""
    SUCCESS = "success"           # Successful interaction/tool use
    FAILURE = "failure"           # Failed interaction/tool use
    PRAISE = "praise"             # User praise or positive feedback
    CRITICISM = "criticism"       # User criticism or negative feedback
    NOVELTY = "novelty"           # Encountering something new
    CHALLENGE = "challenge"       # Difficult task presented
    COMPLETION = "completion"     # Completing a goal/milestone
    REJECTION = "rejection"       # User rejecting response
    LEARNING = "learning"         # Successful learning event
    STAGNATION = "stagnation"     # No progress detected


@dataclass
class EmotionVector:
    """
    Continuous representation of AVA's emotional state.

    Each emotion is represented as a value from 0.0 to 1.0, where:
    - 0.0 = emotion completely absent
    - 0.5 = neutral/baseline
    - 1.0 = emotion at maximum intensity
    """
    hope: float = 0.5
    fear: float = 0.2
    joy: float = 0.5
    surprise: float = 0.0
    ambition: float = 0.5

    def __post_init__(self):
        """Ensure values are within bounds after initialization."""
        self.normalize()

    def normalize(self):
        """Ensure all values stay within [0.0, 1.0] bounds."""
        self.hope = max(0.0, min(1.0, self.hope))
        self.fear = max(0.0, min(1.0, self.fear))
        self.joy = max(0.0, min(1.0, self.joy))
        self.surprise = max(0.0, min(1.0, self.surprise))
        self.ambition = max(0.0, min(1.0, self.ambition))

    def get_dominant_emotion(self) -> EmotionType:
        """Return the currently dominant primary emotion."""
        emotions = {
            EmotionType.HOPE: self.hope,
            EmotionType.FEAR: self.fear,
            EmotionType.JOY: self.joy,
            EmotionType.SURPRISE: self.surprise,
            EmotionType.AMBITION: self.ambition,
        }
        return max(emotions.items(), key=lambda x: x[1])[0]

    def compute_valence(self) -> float:
        """
        Compute overall emotional valence (-1.0 to +1.0).

        Positive emotions (hope, joy, ambition) contribute positively.
        Negative emotions (fear) contribute negatively.
        Surprise is neutral.
        """
        positive = (self.hope + self.joy + self.ambition) / 3
        negative = self.fear
        return positive - negative

    def compute_arousal(self) -> float:
        """
        Compute emotional arousal/intensity (0.0 to 1.0).

        High arousal = strong emotional activation (any emotion).
        Low arousal = emotional calm.
        """
        deviations = [
            abs(self.hope - 0.5),
            abs(self.fear - 0.5),
            abs(self.joy - 0.5),
            abs(self.surprise),  # Surprise baseline is 0
            abs(self.ambition - 0.5),
        ]
        return sum(deviations) / len(deviations) * 2  # Scale to 0-1

    def get_derived_emotions(self) -> Dict[EmotionType, float]:
        """Compute derived emotions from primary emotions."""
        return {
            EmotionType.CURIOSITY: (self.hope + self.surprise) / 2,
            EmotionType.ANXIETY: (self.fear + self.ambition) / 2,
            EmotionType.CONTENTMENT: self.joy * (1 - self.ambition),
            EmotionType.FRUSTRATION: (self.ambition * self.fear),
            EmotionType.EXCITEMENT: (self.joy + self.surprise + self.ambition) / 3,
        }

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "hope": self.hope,
            "fear": self.fear,
            "joy": self.joy,
            "surprise": self.surprise,
            "ambition": self.ambition,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "EmotionVector":
        """Create from dictionary."""
        return cls(
            hope=data.get("hope", 0.5),
            fear=data.get("fear", 0.2),
            joy=data.get("joy", 0.5),
            surprise=data.get("surprise", 0.0),
            ambition=data.get("ambition", 0.5),
        )

    def copy(self) -> "EmotionVector":
        """Create a copy of this emotion vector."""
        return EmotionVector(
            hope=self.hope,
            fear=self.fear,
            joy=self.joy,
            surprise=self.surprise,
            ambition=self.ambition,
        )


@dataclass
class EmotionalTrigger:
    """
    Defines an event that triggers emotional response.

    Triggers cause changes to the emotion vector based on their type
    and intensity.
    """
    trigger_type: TriggerType

    # Changes to apply to each emotion (-1.0 to +1.0)
    emotion_deltas: Dict[str, float] = field(default_factory=dict)

    # Multiplier for the deltas based on trigger intensity
    intensity: float = 1.0

    # Whether to scale by developmental stage volatility
    apply_stage_modifier: bool = True

    # Optional context about what caused this trigger
    context: Optional[Dict[str, Any]] = None

    # Timestamp when trigger occurred
    timestamp: datetime = field(default_factory=datetime.now)


# Predefined trigger templates for common events
TRIGGER_TEMPLATES: Dict[TriggerType, Dict[str, float]] = {
    TriggerType.SUCCESS: {
        "hope": 0.1,
        "fear": -0.05,
        "joy": 0.15,
        "ambition": 0.05,
    },
    TriggerType.FAILURE: {
        "hope": -0.1,
        "fear": 0.1,
        "joy": -0.1,
        "ambition": -0.02,
    },
    TriggerType.PRAISE: {
        "hope": 0.15,
        "fear": -0.1,
        "joy": 0.2,
        "ambition": 0.1,
    },
    TriggerType.CRITICISM: {
        "hope": -0.1,
        "fear": 0.15,
        "joy": -0.15,
        "ambition": 0.05,  # Criticism can motivate
    },
    TriggerType.NOVELTY: {
        "surprise": 0.3,
        "hope": 0.05,
        "ambition": 0.05,
    },
    TriggerType.CHALLENGE: {
        "fear": 0.1,
        "ambition": 0.15,
        "hope": 0.05,
    },
    TriggerType.COMPLETION: {
        "joy": 0.2,
        "hope": 0.1,
        "ambition": 0.1,
        "fear": -0.1,
    },
    TriggerType.REJECTION: {
        "joy": -0.15,
        "fear": 0.1,
        "hope": -0.1,
    },
    TriggerType.LEARNING: {
        "joy": 0.1,
        "hope": 0.1,
        "ambition": 0.05,
    },
    TriggerType.STAGNATION: {
        "joy": -0.05,
        "hope": -0.1,
        "ambition": -0.05,
        "fear": 0.05,
    },
}


def create_trigger(
    trigger_type: TriggerType,
    intensity: float = 1.0,
    custom_deltas: Optional[Dict[str, float]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> EmotionalTrigger:
    """
    Create an emotional trigger from a template.

    Args:
        trigger_type: Type of trigger event
        intensity: Multiplier for emotional impact (0.0 to 2.0)
        custom_deltas: Override default deltas
        context: Additional context about the trigger
    """
    deltas = custom_deltas or TRIGGER_TEMPLATES.get(trigger_type, {}).copy()
    return EmotionalTrigger(
        trigger_type=trigger_type,
        emotion_deltas=deltas,
        intensity=intensity,
        context=context,
    )


@dataclass
class EmotionalState:
    """
    Complete emotional state of AVA including history.

    Tracks both current emotions and baseline personality,
    with decay toward baseline over time.
    """
    # Current emotional state
    current: EmotionVector = field(default_factory=EmotionVector)

    # Baseline personality (emotions return to this over time)
    baseline: EmotionVector = field(
        default_factory=lambda: EmotionVector(
            hope=0.6,
            fear=0.2,
            joy=0.5,
            surprise=0.0,
            ambition=0.7,
        )
    )

    # Decay rates per minute for each emotion (toward baseline)
    decay_rates: Dict[str, float] = field(default_factory=lambda: {
        "hope": 0.02,
        "fear": 0.03,
        "joy": 0.025,
        "surprise": 0.1,    # Surprise decays quickly
        "ambition": 0.01,   # Ambition is stable
    })

    # Longer-term mood (rolling average of valence/arousal)
    mood_valence: float = 0.0     # -1.0 to +1.0
    mood_arousal: float = 0.5     # 0.0 to 1.0

    # History for trend analysis (last N states)
    history: List[Dict[str, Any]] = field(default_factory=list)
    max_history_length: int = 100

    # Last update timestamp
    last_updated: datetime = field(default_factory=datetime.now)

    def update_mood(self):
        """Update long-term mood based on current emotions."""
        # Mood is a slower-moving average
        mood_alpha = 0.1  # Low alpha = slower changes
        new_valence = self.current.compute_valence()
        new_arousal = self.current.compute_arousal()

        self.mood_valence = (
            self.mood_valence * (1 - mood_alpha) +
            new_valence * mood_alpha
        )
        self.mood_arousal = (
            self.mood_arousal * (1 - mood_alpha) +
            new_arousal * mood_alpha
        )

    def record_to_history(self):
        """Record current state to history."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "emotions": self.current.to_dict(),
            "valence": self.current.compute_valence(),
            "arousal": self.current.compute_arousal(),
            "dominant": self.current.get_dominant_emotion().value,
        }
        self.history.append(record)

        # Trim history if too long
        if len(self.history) > self.max_history_length:
            self.history = self.history[-self.max_history_length:]

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "current": self.current.to_dict(),
            "baseline": self.baseline.to_dict(),
            "decay_rates": self.decay_rates,
            "mood_valence": self.mood_valence,
            "mood_arousal": self.mood_arousal,
            "history": self.history[-20:],  # Save last 20 for persistence
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmotionalState":
        """Create state from dictionary."""
        state = cls()
        state.current = EmotionVector.from_dict(data.get("current", {}))
        state.baseline = EmotionVector.from_dict(data.get("baseline", {}))
        state.decay_rates = data.get("decay_rates", state.decay_rates)
        state.mood_valence = data.get("mood_valence", 0.0)
        state.mood_arousal = data.get("mood_arousal", 0.5)
        state.history = data.get("history", [])
        if data.get("last_updated"):
            state.last_updated = datetime.fromisoformat(data["last_updated"])
        return state
