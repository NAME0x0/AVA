"""
Continual Learning for AVA

Implements online learning from interactions, collecting high-quality
samples for future fine-tuning, and tracking learning progress.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SampleQuality(Enum):
    """Quality rating for learning samples."""
    EXCELLENT = "excellent"    # User explicitly praised
    GOOD = "good"              # Successful interaction
    NEUTRAL = "neutral"        # Normal interaction
    POOR = "poor"              # User corrected AVA
    BAD = "bad"                # User complained/frustrated


@dataclass
class LearningSample:
    """A single learning sample from an interaction."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    
    # The interaction
    user_input: str = ""
    ava_response: str = ""
    
    # Context
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    developmental_stage: str = "INFANT"
    emotional_state: Dict[str, float] = field(default_factory=dict)
    tools_used: List[str] = field(default_factory=list)
    thinking_trace: str = ""
    
    # Quality assessment
    quality: SampleQuality = SampleQuality.NEUTRAL
    user_feedback: Optional[str] = None
    correction: Optional[str] = None  # If user provided a correction
    
    # Learning metadata
    learning_weight: float = 1.0  # How much to weight this sample
    has_been_used: bool = False   # Whether used in fine-tuning
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "user_input": self.user_input,
            "ava_response": self.ava_response,
            "conversation_history": self.conversation_history,
            "developmental_stage": self.developmental_stage,
            "emotional_state": self.emotional_state,
            "tools_used": self.tools_used,
            "thinking_trace": self.thinking_trace,
            "quality": self.quality.value,
            "user_feedback": self.user_feedback,
            "correction": self.correction,
            "learning_weight": self.learning_weight,
            "has_been_used": self.has_been_used,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearningSample":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            user_input=data["user_input"],
            ava_response=data["ava_response"],
            conversation_history=data.get("conversation_history", []),
            developmental_stage=data.get("developmental_stage", "INFANT"),
            emotional_state=data.get("emotional_state", {}),
            tools_used=data.get("tools_used", []),
            thinking_trace=data.get("thinking_trace", ""),
            quality=SampleQuality(data.get("quality", "neutral")),
            user_feedback=data.get("user_feedback"),
            correction=data.get("correction"),
            learning_weight=data.get("learning_weight", 1.0),
            has_been_used=data.get("has_been_used", False),
        )
    
    def to_training_format(self) -> Dict[str, str]:
        """
        Convert to format suitable for fine-tuning.
        
        Returns a prompt/completion pair.
        """
        # Build the prompt with context
        prompt_parts = []
        
        # Add conversation history
        for msg in self.conversation_history[-3:]:  # Last 3 turns
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"[{role}]: {content}")
        
        # Add current user input
        prompt_parts.append(f"[user]: {self.user_input}")
        
        prompt = "\n".join(prompt_parts)
        
        # Completion is AVA's response (or correction if provided)
        completion = self.correction if self.correction else self.ava_response
        
        return {
            "prompt": prompt,
            "completion": f"[assistant]: {completion}",
        }


@dataclass
class LearningEvent:
    """Records a learning event (not necessarily a sample)."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = ""  # "interaction", "correction", "praise", "stage_transition", etc.
    description: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "description": self.description,
            "metrics": self.metrics,
        }


class ContinualLearner:
    """
    Manages continual learning from interactions.
    
    Responsibilities:
    - Collect learning samples from interactions
    - Assess sample quality based on feedback
    - Store samples for future fine-tuning
    - Track learning progress and metrics
    """
    
    def __init__(
        self,
        samples_dir: str = "data/learning/samples",
        min_quality_for_training: SampleQuality = SampleQuality.NEUTRAL,
    ):
        """
        Initialize the continual learner.
        
        Args:
            samples_dir: Directory to store learning samples
            min_quality_for_training: Minimum quality to include in training
        """
        self.samples_dir = Path(samples_dir)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        
        self.min_quality_for_training = min_quality_for_training
        
        # In-memory sample buffer
        self.sample_buffer: List[LearningSample] = []
        self.max_buffer_size = 100
        
        # Learning events log
        self.events: List[LearningEvent] = []
        
        # Statistics
        self.stats = {
            "total_samples": 0,
            "excellent_samples": 0,
            "good_samples": 0,
            "neutral_samples": 0,
            "poor_samples": 0,
            "bad_samples": 0,
            "corrections_received": 0,
            "samples_used_in_training": 0,
        }
        
        self._load_stats()
    
    def record_interaction(
        self,
        user_input: str,
        ava_response: str,
        conversation_history: List[Dict[str, str]] = None,
        developmental_stage: str = "INFANT",
        emotional_state: Dict[str, float] = None,
        tools_used: List[str] = None,
        thinking_trace: str = "",
    ) -> LearningSample:
        """
        Record an interaction as a potential learning sample.
        
        Args:
            user_input: What the user said
            ava_response: AVA's response
            conversation_history: Previous conversation turns
            developmental_stage: Current stage
            emotional_state: Current emotions
            tools_used: Tools that were used
            thinking_trace: The thinking process
            
        Returns:
            The created learning sample
        """
        sample = LearningSample(
            user_input=user_input,
            ava_response=ava_response,
            conversation_history=conversation_history or [],
            developmental_stage=developmental_stage,
            emotional_state=emotional_state or {},
            tools_used=tools_used or [],
            thinking_trace=thinking_trace,
        )
        
        self.sample_buffer.append(sample)
        self.stats["total_samples"] += 1
        self.stats["neutral_samples"] += 1
        
        # Flush buffer if too large
        if len(self.sample_buffer) >= self.max_buffer_size:
            self._flush_buffer()
        
        # Log the event
        self._log_event("interaction", f"Recorded interaction sample {sample.id}")
        
        return sample
    
    def update_sample_quality(
        self,
        sample_id: str,
        quality: SampleQuality,
        user_feedback: Optional[str] = None,
        correction: Optional[str] = None,
    ) -> bool:
        """
        Update the quality rating of a sample.
        
        Args:
            sample_id: The sample to update
            quality: New quality rating
            user_feedback: Optional feedback text
            correction: Optional correction from user
            
        Returns:
            True if sample was found and updated
        """
        # Check buffer first
        for sample in self.sample_buffer:
            if sample.id == sample_id:
                old_quality = sample.quality
                sample.quality = quality
                sample.user_feedback = user_feedback
                sample.correction = correction
                
                # Update learning weight based on quality
                sample.learning_weight = self._calculate_weight(quality, correction)
                
                # Update stats
                self._update_quality_stats(old_quality, quality)
                
                if correction:
                    self.stats["corrections_received"] += 1
                    self._log_event(
                        "correction",
                        f"User provided correction for sample {sample_id}",
                    )
                
                return True
        
        # Check persisted samples
        return self._update_persisted_sample(sample_id, quality, user_feedback, correction)
    
    def _calculate_weight(
        self,
        quality: SampleQuality,
        correction: Optional[str],
    ) -> float:
        """Calculate learning weight based on quality."""
        base_weights = {
            SampleQuality.EXCELLENT: 2.0,
            SampleQuality.GOOD: 1.5,
            SampleQuality.NEUTRAL: 1.0,
            SampleQuality.POOR: 0.5,  # Still learn what not to do
            SampleQuality.BAD: 0.3,
        }
        
        weight = base_weights.get(quality, 1.0)
        
        # Corrections are highly valuable for learning
        if correction:
            weight *= 2.0
        
        return weight
    
    def _update_quality_stats(
        self,
        old_quality: SampleQuality,
        new_quality: SampleQuality,
    ):
        """Update quality statistics."""
        quality_keys = {
            SampleQuality.EXCELLENT: "excellent_samples",
            SampleQuality.GOOD: "good_samples",
            SampleQuality.NEUTRAL: "neutral_samples",
            SampleQuality.POOR: "poor_samples",
            SampleQuality.BAD: "bad_samples",
        }
        
        # Decrement old
        if old_quality in quality_keys:
            self.stats[quality_keys[old_quality]] -= 1
        
        # Increment new
        if new_quality in quality_keys:
            self.stats[quality_keys[new_quality]] += 1
    
    def get_training_samples(
        self,
        max_samples: int = 1000,
        include_used: bool = False,
    ) -> List[LearningSample]:
        """
        Get samples suitable for training.
        
        Args:
            max_samples: Maximum number of samples to return
            include_used: Whether to include previously used samples
            
        Returns:
            List of training samples
        """
        samples = []
        
        # Flush buffer first to ensure all samples are persisted
        self._flush_buffer()
        
        # Collect samples from files
        for sample_file in self.samples_dir.glob("*.json"):
            try:
                with open(sample_file, "r") as f:
                    data = json.load(f)
                    sample = LearningSample.from_dict(data)
                    
                    # Filter by quality
                    if self._meets_quality_threshold(sample.quality):
                        # Filter by used status
                        if include_used or not sample.has_been_used:
                            samples.append(sample)
                            
            except Exception as e:
                logger.warning(f"Failed to load sample {sample_file}: {e}")
        
        # Sort by learning weight (highest first) and recency
        samples.sort(
            key=lambda s: (s.learning_weight, s.timestamp.timestamp()),
            reverse=True,
        )
        
        return samples[:max_samples]
    
    def _meets_quality_threshold(self, quality: SampleQuality) -> bool:
        """Check if quality meets minimum threshold."""
        quality_order = [
            SampleQuality.BAD,
            SampleQuality.POOR,
            SampleQuality.NEUTRAL,
            SampleQuality.GOOD,
            SampleQuality.EXCELLENT,
        ]
        
        return quality_order.index(quality) >= quality_order.index(
            self.min_quality_for_training
        )
    
    def mark_samples_used(self, sample_ids: List[str]):
        """Mark samples as used in training."""
        for sample_id in sample_ids:
            self._mark_sample_used(sample_id)
        
        self.stats["samples_used_in_training"] += len(sample_ids)
        self._save_stats()
    
    def _mark_sample_used(self, sample_id: str):
        """Mark a single sample as used."""
        sample_file = self.samples_dir / f"{sample_id}.json"
        
        if sample_file.exists():
            try:
                with open(sample_file, "r") as f:
                    data = json.load(f)
                
                data["has_been_used"] = True
                
                with open(sample_file, "w") as f:
                    json.dump(data, f, indent=2)
                    
            except Exception as e:
                logger.warning(f"Failed to mark sample {sample_id} as used: {e}")
    
    def get_sample_count(self, unused_only: bool = False) -> int:
        """Get count of available samples."""
        count = 0
        
        # Count buffer
        for sample in self.sample_buffer:
            if not unused_only or not sample.has_been_used:
                if self._meets_quality_threshold(sample.quality):
                    count += 1
        
        # Count persisted
        for sample_file in self.samples_dir.glob("*.json"):
            try:
                with open(sample_file, "r") as f:
                    data = json.load(f)
                    if self._meets_quality_threshold(SampleQuality(data.get("quality", "neutral"))):
                        if not unused_only or not data.get("has_been_used", False):
                            count += 1
            except Exception:
                pass
        
        return count
    
    def _flush_buffer(self):
        """Flush sample buffer to disk."""
        for sample in self.sample_buffer:
            sample_file = self.samples_dir / f"{sample.id}.json"
            
            try:
                with open(sample_file, "w") as f:
                    json.dump(sample.to_dict(), f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save sample {sample.id}: {e}")
        
        self.sample_buffer.clear()
        self._save_stats()
    
    def _update_persisted_sample(
        self,
        sample_id: str,
        quality: SampleQuality,
        user_feedback: Optional[str],
        correction: Optional[str],
    ) -> bool:
        """Update a persisted sample's quality."""
        sample_file = self.samples_dir / f"{sample_id}.json"
        
        if not sample_file.exists():
            return False
        
        try:
            with open(sample_file, "r") as f:
                data = json.load(f)
            
            old_quality = SampleQuality(data.get("quality", "neutral"))
            
            data["quality"] = quality.value
            data["user_feedback"] = user_feedback
            data["correction"] = correction
            data["learning_weight"] = self._calculate_weight(quality, correction)
            
            with open(sample_file, "w") as f:
                json.dump(data, f, indent=2)
            
            self._update_quality_stats(old_quality, quality)
            
            if correction:
                self.stats["corrections_received"] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update sample {sample_id}: {e}")
            return False
    
    def _log_event(self, event_type: str, description: str, metrics: Dict[str, float] = None):
        """Log a learning event."""
        event = LearningEvent(
            event_type=event_type,
            description=description,
            metrics=metrics or {},
        )
        self.events.append(event)
        
        # Keep events bounded
        if len(self.events) > 1000:
            self.events = self.events[-500:]
    
    def _load_stats(self):
        """Load statistics from disk."""
        stats_file = self.samples_dir / "stats.json"
        
        if stats_file.exists():
            try:
                with open(stats_file, "r") as f:
                    saved_stats = json.load(f)
                    self.stats.update(saved_stats)
            except Exception as e:
                logger.warning(f"Failed to load stats: {e}")
    
    def _save_stats(self):
        """Save statistics to disk."""
        stats_file = self.samples_dir / "stats.json"
        
        try:
            with open(stats_file, "w") as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save stats: {e}")
    
    def get_learning_metrics(self) -> Dict[str, Any]:
        """Get current learning metrics."""
        return {
            **self.stats,
            "buffer_size": len(self.sample_buffer),
            "available_for_training": self.get_sample_count(unused_only=True),
            "recent_events": [e.to_dict() for e in self.events[-10:]],
        }
