"""
Development Tracker for AVA

Central system for tracking AVA's developmental progression using
a hybrid approach combining time passage, interaction count, and
competency milestones.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .stages import (
    DevelopmentalStage,
    StageProperties,
    STAGE_PROPERTIES,
    get_stage_properties,
    get_next_stage,
)
from .milestones import MilestoneChecker, MILESTONES

logger = logging.getLogger(__name__)


@dataclass
class DevelopmentalState:
    """
    Complete developmental state of AVA.

    This state is persisted to disk and loaded on startup to maintain
    continuity of development across sessions.
    """
    current_stage: DevelopmentalStage = DevelopmentalStage.INFANT
    birth_timestamp: datetime = field(default_factory=datetime.now)

    # Interaction metrics
    total_interactions: int = 0
    successful_interactions: int = 0
    failed_interactions: int = 0

    # Quality metrics
    total_quality_score: float = 0.0
    avg_quality_score: float = 0.0

    # Competency scores (0.0 to 1.0)
    competencies: Dict[str, float] = field(default_factory=lambda: {
        "communication": 0.1,
        "reasoning": 0.1,
        "emotional": 0.1,
        "tool_use": 0.0,
        "memory": 0.1,
        "social": 0.1,
        "meta_cognitive": 0.0,
    })

    # Stage transition history
    stage_history: List[Dict[str, Any]] = field(default_factory=list)

    # Vocabulary tracking
    unique_words_used: int = 0

    # Tool usage stats
    tool_attempts: int = 0
    tool_successes: int = 0

    def get_age_days(self, time_acceleration: float = 1.0) -> float:
        """Calculate current age in days with time acceleration."""
        elapsed = datetime.now() - self.birth_timestamp
        return elapsed.total_seconds() / 86400 * time_acceleration

    def get_age_human_equivalent(self, time_acceleration: float = 1.0) -> str:
        """Get human-readable age equivalent."""
        days = self.get_age_days(time_acceleration)
        if days < 30:
            return f"{int(days)} days old (infant)"
        elif days < 365:
            months = int(days / 30)
            return f"{months} months old (toddler/child)"
        else:
            years = days / 365
            return f"{years:.1f} years old"

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "current_stage": self.current_stage.value,
            "birth_timestamp": self.birth_timestamp.isoformat(),
            "total_interactions": self.total_interactions,
            "successful_interactions": self.successful_interactions,
            "failed_interactions": self.failed_interactions,
            "total_quality_score": self.total_quality_score,
            "avg_quality_score": self.avg_quality_score,
            "competencies": self.competencies,
            "stage_history": self.stage_history,
            "unique_words_used": self.unique_words_used,
            "tool_attempts": self.tool_attempts,
            "tool_successes": self.tool_successes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DevelopmentalState":
        """Create state from dictionary."""
        state = cls()
        state.current_stage = DevelopmentalStage(data.get("current_stage", 0))
        state.birth_timestamp = datetime.fromisoformat(
            data.get("birth_timestamp", datetime.now().isoformat())
        )
        state.total_interactions = data.get("total_interactions", 0)
        state.successful_interactions = data.get("successful_interactions", 0)
        state.failed_interactions = data.get("failed_interactions", 0)
        state.total_quality_score = data.get("total_quality_score", 0.0)
        state.avg_quality_score = data.get("avg_quality_score", 0.0)
        state.competencies = data.get("competencies", state.competencies)
        state.stage_history = data.get("stage_history", [])
        state.unique_words_used = data.get("unique_words_used", 0)
        state.tool_attempts = data.get("tool_attempts", 0)
        state.tool_successes = data.get("tool_successes", 0)
        return state


class DevelopmentTracker:
    """
    Tracks and manages AVA's developmental progression.

    Uses a hybrid maturation approach combining:
    - Time passage (with configurable acceleration)
    - Interaction count and quality
    - Competency milestones

    The maturation score determines when stage transitions occur.
    """

    def __init__(
        self,
        data_path: str = "data/developmental",
        time_acceleration: float = 5.0,
        maturation_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the development tracker.

        Args:
            data_path: Path for persisting developmental state
            time_acceleration: Multiplier for simulated time (5.0 = 5x faster)
            maturation_weights: Weights for hybrid maturation calculation
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)

        self.time_acceleration = time_acceleration
        self.maturation_weights = maturation_weights or {
            "time": 0.3,
            "interactions": 0.4,
            "milestones": 0.3,
        }

        self.milestone_checker = MilestoneChecker(str(self.data_path))
        self.state = self._load_state()

        logger.info(
            f"DevelopmentTracker initialized: Stage={self.state.current_stage.name}, "
            f"Age={self.state.get_age_human_equivalent(self.time_acceleration)}"
        )

    def _load_state(self) -> DevelopmentalState:
        """Load developmental state from disk or create new."""
        state_file = self.data_path / "state.json"
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    data = json.load(f)
                    return DevelopmentalState.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load state, creating new: {e}")
                return DevelopmentalState()
        return DevelopmentalState()

    def save_state(self):
        """Save developmental state to disk."""
        state_file = self.data_path / "state.json"
        with open(state_file, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)

    def get_state(self) -> DevelopmentalState:
        """Get the current developmental state."""
        return self.state

    def get_stage_properties(self) -> StageProperties:
        """Get properties for the current developmental stage."""
        return get_stage_properties(self.state.current_stage)

    def calculate_maturation_score(self) -> float:
        """
        Calculate the hybrid maturation score (0.0 to 1.0).

        Combines:
        - Time-based aging progress
        - Interaction-based progress
        - Milestone-based progress
        """
        props = self.get_stage_properties()
        next_stage = get_next_stage(self.state.current_stage)

        if next_stage is None:
            return 1.0  # Already at MATURE

        next_props = get_stage_properties(next_stage)

        # Time score: progress toward next stage's min age
        current_age = self.state.get_age_days(self.time_acceleration)
        time_range = next_props.min_age_days - props.min_age_days
        time_progress = (current_age - props.min_age_days) / time_range if time_range > 0 else 1.0
        time_score = min(1.0, max(0.0, time_progress))

        # Interaction score: progress toward next stage's min interactions
        interaction_range = next_props.min_interactions - props.min_interactions
        interaction_progress = (
            (self.state.total_interactions - props.min_interactions) / interaction_range
            if interaction_range > 0 else 1.0
        )
        interaction_score = min(1.0, max(0.0, interaction_progress))

        # Milestone score: percentage of required milestones achieved
        required = props.required_milestones
        if required:
            achieved = self.milestone_checker.get_achieved_milestones()
            achieved_required = [m for m in required if m in achieved]
            milestone_score = len(achieved_required) / len(required)
        else:
            milestone_score = 1.0

        # Weighted combination
        maturation_score = (
            self.maturation_weights["time"] * time_score +
            self.maturation_weights["interactions"] * interaction_score +
            self.maturation_weights["milestones"] * milestone_score
        )

        return maturation_score

    def check_stage_transition(self) -> Optional[DevelopmentalStage]:
        """
        Check if AVA should transition to the next developmental stage.

        Returns the new stage if transitioning, None otherwise.
        """
        next_stage = get_next_stage(self.state.current_stage)
        if next_stage is None:
            return None  # Already MATURE

        maturation_score = self.calculate_maturation_score()

        # Require minimum 80% maturation AND all milestones for transition
        props = self.get_stage_properties()
        required_milestones = props.required_milestones
        achieved = self.milestone_checker.get_achieved_milestones()
        milestones_complete = all(m in achieved for m in required_milestones)

        if maturation_score >= 0.8 and milestones_complete:
            self._transition_to_stage(next_stage)
            return next_stage

        return None

    def _transition_to_stage(self, new_stage: DevelopmentalStage):
        """Execute a stage transition."""
        old_stage = self.state.current_stage
        transition_record = {
            "from_stage": old_stage.value,
            "to_stage": new_stage.value,
            "timestamp": datetime.now().isoformat(),
            "age_days": self.state.get_age_days(self.time_acceleration),
            "total_interactions": self.state.total_interactions,
            "maturation_score": self.calculate_maturation_score(),
        }

        self.state.stage_history.append(transition_record)
        self.state.current_stage = new_stage
        self.save_state()

        logger.info(
            f"Stage transition: {old_stage.name} -> {new_stage.name} "
            f"(Age: {self.state.get_age_human_equivalent(self.time_acceleration)})"
        )

    def record_interaction(
        self,
        outcome: str,  # "success", "partial", "failure"
        quality_score: float = 0.5,  # 0.0 to 1.0
        words_used: Optional[List[str]] = None,
        tool_used: Optional[str] = None,
        tool_success: Optional[bool] = None,
    ):
        """
        Record an interaction and update developmental metrics.

        Args:
            outcome: Interaction outcome
            quality_score: Quality rating of the interaction
            words_used: List of words used in the response
            tool_used: Name of tool used (if any)
            tool_success: Whether tool use was successful
        """
        self.state.total_interactions += 1

        if outcome == "success":
            self.state.successful_interactions += 1
        elif outcome == "failure":
            self.state.failed_interactions += 1

        # Update quality tracking
        self.state.total_quality_score += quality_score
        self.state.avg_quality_score = (
            self.state.total_quality_score / self.state.total_interactions
        )

        # Track vocabulary
        if words_used:
            # This is a simple approximation; real implementation would track unique words
            self.state.unique_words_used = max(
                self.state.unique_words_used,
                len(set(words_used))
            )

        # Track tool usage
        if tool_used:
            self.state.tool_attempts += 1
            if tool_success:
                self.state.tool_successes += 1

        self.save_state()

    def update_competency(self, competency: str, delta: float):
        """
        Update a competency score.

        Args:
            competency: Name of competency to update
            delta: Amount to add (can be negative)
        """
        if competency in self.state.competencies:
            self.state.competencies[competency] = max(
                0.0,
                min(1.0, self.state.competencies[competency] + delta)
            )
            self.save_state()

    def record_milestone(
        self,
        milestone_id: str,
        evidence: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record progress toward a milestone.

        Returns True if the milestone was just achieved.
        """
        return self.milestone_checker.record_occurrence(milestone_id, evidence)

    def get_status_summary(self) -> Dict[str, Any]:
        """Get a comprehensive status summary."""
        props = self.get_stage_properties()
        milestone_summary = self.milestone_checker.get_progress_summary()

        return {
            "stage": {
                "current": self.state.current_stage.name,
                "value": self.state.current_stage.value,
                "properties": {
                    "articulation_clarity": props.articulation_clarity,
                    "vocabulary_range": props.vocabulary_range,
                    "reasoning_depth": props.reasoning_depth,
                    "emotional_stability": props.emotional_stability,
                    "tool_safety_level": props.tool_safety_level,
                    "learning_rate_multiplier": props.learning_rate_multiplier,
                },
            },
            "age": {
                "days": self.state.get_age_days(self.time_acceleration),
                "human_equivalent": self.state.get_age_human_equivalent(self.time_acceleration),
                "time_acceleration": self.time_acceleration,
            },
            "interactions": {
                "total": self.state.total_interactions,
                "successful": self.state.successful_interactions,
                "failed": self.state.failed_interactions,
                "success_rate": (
                    self.state.successful_interactions / self.state.total_interactions
                    if self.state.total_interactions > 0 else 0
                ),
                "avg_quality": self.state.avg_quality_score,
            },
            "maturation": {
                "score": self.calculate_maturation_score(),
                "next_stage": get_next_stage(self.state.current_stage).name
                if get_next_stage(self.state.current_stage) else "MATURE (capped)",
            },
            "competencies": self.state.competencies,
            "milestones": milestone_summary,
            "tools": {
                "attempts": self.state.tool_attempts,
                "successes": self.state.tool_successes,
                "success_rate": (
                    self.state.tool_successes / self.state.tool_attempts
                    if self.state.tool_attempts > 0 else 0
                ),
            },
        }

    def reset(self, preserve_history: bool = False):
        """
        Reset developmental state to initial (INFANT).

        Args:
            preserve_history: If True, keep stage transition history
        """
        history = self.state.stage_history if preserve_history else []
        self.state = DevelopmentalState()
        self.state.stage_history = history
        self.milestone_checker = MilestoneChecker(str(self.data_path))
        self.save_state()
        logger.info("Developmental state reset to INFANT")
