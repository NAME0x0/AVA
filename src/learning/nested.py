"""
Nested Learning Contexts for AVA

Implements meta-learning through nested contexts that allow AVA
to learn different things at different scopes:
- Session-level: Learning within a conversation
- Topic-level: Learning about specific topics
- Task-level: Learning how to do specific tasks
- Global-level: Overall behavioral learning

Also implements Fast/Slow weight separation based on:
Reference: "Nested Learning: A New ML Paradigm for Continual Learning" (Google, 2025)
"""

import collections
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class LearningScope(Enum):
    """Scope levels for nested learning."""

    GLOBAL = "global"  # Overall behavioral patterns
    TOPIC = "topic"  # Topic-specific knowledge
    TASK = "task"  # Task-specific procedures
    SESSION = "session"  # Current session only
    EPHEMERAL = "ephemeral"  # Single interaction


@dataclass
class LearningContext:
    """A context for learning at a specific scope."""

    id: str = ""
    name: str = ""
    scope: LearningScope = LearningScope.SESSION
    created_at: datetime = field(default_factory=datetime.now)

    # Context data
    parent_id: str | None = None
    topic: str = ""
    task: str = ""

    # Learning state within this context
    observations: list[str] = field(default_factory=list)
    insights: list[str] = field(default_factory=list)
    patterns: dict[str, Any] = field(default_factory=dict)

    # Performance in this context
    successes: int = 0
    failures: int = 0
    corrections: int = 0

    # Active state
    is_active: bool = True
    closed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "scope": self.scope.value,
            "created_at": self.created_at.isoformat(),
            "parent_id": self.parent_id,
            "topic": self.topic,
            "task": self.task,
            "observations": self.observations,
            "insights": self.insights,
            "patterns": self.patterns,
            "successes": self.successes,
            "failures": self.failures,
            "corrections": self.corrections,
            "is_active": self.is_active,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LearningContext":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            scope=LearningScope(data["scope"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            parent_id=data.get("parent_id"),
            topic=data.get("topic", ""),
            task=data.get("task", ""),
            observations=data.get("observations", []),
            insights=data.get("insights", []),
            patterns=data.get("patterns", {}),
            successes=data.get("successes", 0),
            failures=data.get("failures", 0),
            corrections=data.get("corrections", 0),
            is_active=data.get("is_active", True),
            closed_at=datetime.fromisoformat(data["closed_at"]) if data.get("closed_at") else None,
        )

    @property
    def success_rate(self) -> float:
        """Calculate success rate in this context."""
        total = self.successes + self.failures
        if total == 0:
            return 0.5  # Neutral
        return self.successes / total

    def add_observation(self, observation: str):
        """Add an observation to this context."""
        self.observations.append(observation)

        # Keep bounded
        if len(self.observations) > 100:
            self.observations = self.observations[-50:]

    def add_insight(self, insight: str):
        """Add a derived insight."""
        if insight not in self.insights:
            self.insights.append(insight)

    def record_outcome(self, success: bool, was_corrected: bool = False):
        """Record an outcome in this context."""
        if success:
            self.successes += 1
        else:
            self.failures += 1

        if was_corrected:
            self.corrections += 1


class NestedLearningContext:
    """
    Manages nested learning contexts at different scopes.

    This enables meta-learning by tracking what works in different
    contexts and applying those learnings appropriately.
    """

    def __init__(
        self,
        data_dir: str = "data/learning",
    ):
        """
        Initialize nested learning context manager.

        Args:
            data_dir: Directory for persisting context data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Active context stack
        self.context_stack: list[LearningContext] = []

        # All contexts (indexed by id)
        self.contexts: dict[str, LearningContext] = {}

        # Global context (always present)
        self.global_context = self._get_or_create_global_context()

        # Current session context
        self.session_context: LearningContext | None = None

        self._load_contexts()

    def _get_or_create_global_context(self) -> LearningContext:
        """Get or create the global learning context."""
        global_file = self.data_dir / "global_context.json"

        if global_file.exists():
            try:
                with open(global_file) as f:
                    data = json.load(f)
                    return LearningContext.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load global context: {e}")

        # Create new global context
        import uuid

        context = LearningContext(
            id=str(uuid.uuid4())[:8],
            name="global",
            scope=LearningScope.GLOBAL,
            topic="general",
        )

        self._save_context(context)
        return context

    def start_session(self, session_id: str | None = None) -> LearningContext:
        """
        Start a new session context.

        Args:
            session_id: Optional session identifier

        Returns:
            The new session context
        """
        import uuid

        # Close previous session if any
        if self.session_context and self.session_context.is_active:
            self.close_context(self.session_context.id)

        # Create new session context
        context = LearningContext(
            id=session_id or str(uuid.uuid4())[:8],
            name=f"session_{datetime.now().strftime('%Y%m%d_%H%M')}",
            scope=LearningScope.SESSION,
            parent_id=self.global_context.id,
        )

        self.session_context = context
        self.contexts[context.id] = context
        self.context_stack.append(context)

        logger.debug(f"Started session context: {context.id}")

        return context

    def enter_topic_context(
        self,
        topic: str,
        parent_context: LearningContext | None = None,
    ) -> LearningContext:
        """
        Enter a topic-specific learning context.

        Args:
            topic: The topic to focus on
            parent_context: Parent context (defaults to session or global)

        Returns:
            The topic context
        """
        import uuid

        # Determine parent
        if parent_context is None:
            parent_context = self.session_context or self.global_context

        # Check if we already have this topic context
        for ctx in self.contexts.values():
            if ctx.scope == LearningScope.TOPIC and ctx.topic == topic and ctx.is_active:
                self.context_stack.append(ctx)
                return ctx

        # Create new topic context
        context = LearningContext(
            id=str(uuid.uuid4())[:8],
            name=f"topic_{topic}",
            scope=LearningScope.TOPIC,
            parent_id=parent_context.id,
            topic=topic,
        )

        self.contexts[context.id] = context
        self.context_stack.append(context)

        logger.debug(f"Entered topic context: {topic}")

        return context

    def enter_task_context(
        self,
        task: str,
        parent_context: LearningContext | None = None,
    ) -> LearningContext:
        """
        Enter a task-specific learning context.

        Args:
            task: The task to focus on
            parent_context: Parent context

        Returns:
            The task context
        """
        import uuid

        # Determine parent
        if parent_context is None:
            parent_context = self._get_current_context()

        # Create task context
        context = LearningContext(
            id=str(uuid.uuid4())[:8],
            name=f"task_{task[:20]}",
            scope=LearningScope.TASK,
            parent_id=parent_context.id,
            task=task,
        )

        self.contexts[context.id] = context
        self.context_stack.append(context)

        logger.debug(f"Entered task context: {task[:30]}...")

        return context

    def exit_context(self) -> LearningContext | None:
        """
        Exit the current context and return to parent.

        Returns:
            The exited context, or None if at root
        """
        if len(self.context_stack) <= 1:
            return None

        exited = self.context_stack.pop()
        self._save_context(exited)

        logger.debug(f"Exited context: {exited.name}")

        return exited

    def close_context(self, context_id: str):
        """
        Close a context (mark as inactive).

        Args:
            context_id: ID of context to close
        """
        if context_id in self.contexts:
            context = self.contexts[context_id]
            context.is_active = False
            context.closed_at = datetime.now()
            self._save_context(context)

            # Remove from stack if present
            self.context_stack = [c for c in self.context_stack if c.id != context_id]

    def _get_current_context(self) -> LearningContext:
        """Get the current active context."""
        if self.context_stack:
            return self.context_stack[-1]
        return self.session_context or self.global_context

    def record_observation(self, observation: str, context: LearningContext | None = None):
        """
        Record an observation in the current or specified context.

        Args:
            observation: What was observed
            context: Specific context (defaults to current)
        """
        target = context or self._get_current_context()
        target.add_observation(observation)

        # Also propagate to parent contexts (with less weight)
        self._propagate_to_parents(target, "observation", observation)

    def record_insight(self, insight: str, context: LearningContext | None = None):
        """
        Record an insight (derived learning) in the context.

        Args:
            insight: The insight to record
            context: Specific context
        """
        target = context or self._get_current_context()
        target.add_insight(insight)

        # Insights propagate more strongly to parents
        self._propagate_to_parents(target, "insight", insight)

    def record_outcome(
        self,
        success: bool,
        was_corrected: bool = False,
        context: LearningContext | None = None,
    ):
        """
        Record an outcome in the context.

        Args:
            success: Whether the outcome was successful
            was_corrected: Whether user provided a correction
            context: Specific context
        """
        target = context or self._get_current_context()
        target.record_outcome(success, was_corrected)

        # Propagate outcomes to all parent contexts
        parent_id = target.parent_id
        while parent_id and parent_id in self.contexts:
            parent = self.contexts[parent_id]
            parent.record_outcome(success, was_corrected)
            parent_id = parent.parent_id

        # Also update global
        if target.id != self.global_context.id:
            self.global_context.record_outcome(success, was_corrected)

    def _propagate_to_parents(
        self,
        context: LearningContext,
        data_type: str,
        data: str,
    ):
        """Propagate learning data to parent contexts."""
        parent_id = context.parent_id

        while parent_id and parent_id in self.contexts:
            parent = self.contexts[parent_id]

            if data_type == "observation":
                # Summarize for parent
                parent.add_observation(f"[{context.name}] {data[:100]}")
            elif data_type == "insight":
                parent.add_insight(data)

            parent_id = parent.parent_id

    def get_relevant_context(
        self, topic: str | None = None, task: str | None = None
    ) -> dict[str, Any]:
        """
        Get relevant context for the current situation.

        Aggregates insights and patterns from relevant contexts.

        Args:
            topic: Optional topic filter
            task: Optional task filter

        Returns:
            Aggregated context information
        """
        relevant_insights = []
        relevant_patterns = {}
        relevant_observations = []

        # Start with global context
        relevant_insights.extend(self.global_context.insights[-10:])
        relevant_patterns.update(self.global_context.patterns)

        # Add session context
        if self.session_context:
            relevant_insights.extend(self.session_context.insights[-5:])
            relevant_patterns.update(self.session_context.patterns)
            relevant_observations.extend(self.session_context.observations[-10:])

        # Find topic-specific contexts
        if topic:
            for ctx in self.contexts.values():
                if ctx.scope == LearningScope.TOPIC and topic.lower() in ctx.topic.lower():
                    relevant_insights.extend(ctx.insights[-5:])
                    relevant_patterns.update(ctx.patterns)

        # Find task-specific contexts
        if task:
            for ctx in self.contexts.values():
                if ctx.scope == LearningScope.TASK and task.lower() in ctx.task.lower():
                    relevant_insights.extend(ctx.insights[-3:])
                    relevant_patterns.update(ctx.patterns)

        return {
            "insights": list(set(relevant_insights)),
            "patterns": relevant_patterns,
            "recent_observations": relevant_observations[-5:],
            "global_success_rate": self.global_context.success_rate,
            "session_success_rate": (
                self.session_context.success_rate if self.session_context else None
            ),
        }

    def get_learning_summary(self) -> dict[str, Any]:
        """Get a summary of all learning contexts."""
        active_contexts = [c for c in self.contexts.values() if c.is_active]

        return {
            "total_contexts": len(self.contexts),
            "active_contexts": len(active_contexts),
            "global_insights": len(self.global_context.insights),
            "global_success_rate": self.global_context.success_rate,
            "session_active": self.session_context is not None,
            "context_stack_depth": len(self.context_stack),
            "contexts_by_scope": {
                scope.value: sum(1 for c in self.contexts.values() if c.scope == scope)
                for scope in LearningScope
            },
        }

    def _load_contexts(self):
        """Load persisted contexts."""
        contexts_file = self.data_dir / "contexts.json"

        if contexts_file.exists():
            try:
                with open(contexts_file) as f:
                    data = json.load(f)
                    for ctx_data in data.get("contexts", []):
                        ctx = LearningContext.from_dict(ctx_data)
                        self.contexts[ctx.id] = ctx
            except Exception as e:
                logger.warning(f"Failed to load contexts: {e}")

    def _save_context(self, context: LearningContext):
        """Save a single context."""
        # Save global context separately
        if context.scope == LearningScope.GLOBAL:
            global_file = self.data_dir / "global_context.json"
            with open(global_file, "w") as f:
                json.dump(context.to_dict(), f, indent=2)

        # Save all contexts
        self._save_all_contexts()

    def _save_all_contexts(self):
        """Save all contexts to disk."""
        contexts_file = self.data_dir / "contexts.json"

        try:
            data = {
                "contexts": [c.to_dict() for c in self.contexts.values()],
                "saved_at": datetime.now().isoformat(),
            }

            with open(contexts_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save contexts: {e}")

    def end_session(self):
        """End the current session and save all contexts."""
        if self.session_context:
            self.close_context(self.session_context.id)
            self.session_context = None

        # Save global context with session learnings
        self._save_context(self.global_context)
        self._save_all_contexts()


# =============================================================================
# FAST/SLOW WEIGHT SEPARATION - Nested Learning Paradigm
# =============================================================================
# Reference: "Nested Learning: A New ML Paradigm for Continual Learning" (Google, 2025)
#
# Key concepts:
# - Fast weights: Adapt quickly to current task (1-10 updates)
# - Slow weights: Consolidate over many interactions (100-1000 updates)
# - Replay buffer: Store high-quality samples for consolidation
# - Drift penalty: Prevent catastrophic forgetting during fast updates
# =============================================================================


@dataclass
class ReplayBufferSample:
    """A sample stored in the replay buffer."""

    id: str = ""
    input_text: str = ""
    output_text: str = ""
    quality_score: float = 0.5
    emotional_context: dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: datetime | None = None
    task_context: str = ""
    topic_context: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "quality_score": self.quality_score,
            "emotional_context": self.emotional_context,
            "created_at": self.created_at.isoformat(),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "task_context": self.task_context,
            "topic_context": self.topic_context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReplayBufferSample":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            input_text=data["input_text"],
            output_text=data["output_text"],
            quality_score=data.get("quality_score", 0.5),
            emotional_context=data.get("emotional_context", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            access_count=data.get("access_count", 0),
            last_accessed=(
                datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None
            ),
            task_context=data.get("task_context", ""),
            topic_context=data.get("topic_context", ""),
        )


@dataclass
class FastSlowConfig:
    """Configuration for Fast/Slow weight separation."""

    # Fast weight settings
    fast_learning_rate: float = 1e-3
    fast_update_interval: int = 1  # Update every N interactions

    # Slow weight settings
    slow_learning_rate: float = 1e-5
    slow_update_interval: int = 100  # Consolidation every N interactions

    # Replay buffer settings
    replay_buffer_size: int = 5000
    replay_batch_size: int = 32
    min_quality_for_replay: float = 0.5

    # Drift prevention
    drift_penalty_weight: float = 0.1
    max_drift_per_step: float = 0.05

    # Consolidation settings
    consolidation_epochs: int = 3
    consolidation_mix_ratio: float = 0.5  # 50% new, 50% replay

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fast_learning_rate": self.fast_learning_rate,
            "fast_update_interval": self.fast_update_interval,
            "slow_learning_rate": self.slow_learning_rate,
            "slow_update_interval": self.slow_update_interval,
            "replay_buffer_size": self.replay_buffer_size,
            "replay_batch_size": self.replay_batch_size,
            "min_quality_for_replay": self.min_quality_for_replay,
            "drift_penalty_weight": self.drift_penalty_weight,
            "max_drift_per_step": self.max_drift_per_step,
            "consolidation_epochs": self.consolidation_epochs,
            "consolidation_mix_ratio": self.consolidation_mix_ratio,
        }


class ReplayBuffer:
    """
    Experience replay buffer for continual learning.

    Stores high-quality interaction samples and provides
    prioritized sampling for training.
    """

    def __init__(
        self,
        max_size: int = 5000,
        min_quality: float = 0.5,
        data_dir: str = "data/learning/replay",
    ):
        """
        Initialize replay buffer.

        Args:
            max_size: Maximum number of samples
            min_quality: Minimum quality to store
            data_dir: Directory for persistence
        """
        self.max_size = max_size
        self.min_quality = min_quality
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Storage
        self.buffer: collections.deque = collections.deque(maxlen=max_size)

        # Indexing for efficient retrieval
        self._by_quality: dict[str, float] = {}  # id -> quality
        self._by_topic: dict[str, list[str]] = collections.defaultdict(list)  # topic -> [ids]
        self._by_task: dict[str, list[str]] = collections.defaultdict(list)  # task -> [ids]

        # Statistics
        self.total_added = 0
        self.total_sampled = 0

        self._load()

    def add(
        self,
        input_text: str,
        output_text: str,
        quality_score: float,
        emotional_context: dict[str, float] | None = None,
        task_context: str = "",
        topic_context: str = "",
    ) -> str | None:
        """
        Add a sample to the replay buffer.

        Args:
            input_text: User input
            output_text: Model output
            quality_score: Quality rating (0-1)
            emotional_context: Emotional state during interaction
            task_context: Task context if any
            topic_context: Topic context if any

        Returns:
            Sample ID if added, None if rejected
        """
        # Quality filter
        if quality_score < self.min_quality:
            return None

        import uuid

        sample_id = str(uuid.uuid4())[:12]

        sample = ReplayBufferSample(
            id=sample_id,
            input_text=input_text,
            output_text=output_text,
            quality_score=quality_score,
            emotional_context=emotional_context or {},
            task_context=task_context,
            topic_context=topic_context,
        )

        self.buffer.append(sample)

        # Update indices
        self._by_quality[sample_id] = quality_score
        if topic_context:
            self._by_topic[topic_context].append(sample_id)
        if task_context:
            self._by_task[task_context].append(sample_id)

        self.total_added += 1

        # Prune indices if buffer wrapped
        self._prune_indices()

        return sample_id

    def sample(
        self,
        batch_size: int = 32,
        strategy: str = "quality_weighted",
        topic_filter: str | None = None,
        task_filter: str | None = None,
    ) -> list[ReplayBufferSample]:
        """
        Sample from the replay buffer.

        Args:
            batch_size: Number of samples to retrieve
            strategy: Sampling strategy (uniform, quality_weighted, recency_weighted)
            topic_filter: Optional topic filter
            task_filter: Optional task filter

        Returns:
            List of samples
        """
        if len(self.buffer) == 0:
            return []

        # Apply filters
        candidates = list(self.buffer)

        if topic_filter:
            topic_ids = set(self._by_topic.get(topic_filter, []))
            candidates = [s for s in candidates if s.id in topic_ids]

        if task_filter:
            task_ids = set(self._by_task.get(task_filter, []))
            candidates = [s for s in candidates if s.id in task_ids]

        if not candidates:
            candidates = list(self.buffer)

        # Compute sampling weights
        if strategy == "quality_weighted":
            weights = np.array([s.quality_score for s in candidates])
        elif strategy == "recency_weighted":
            now = datetime.now()
            weights = np.array([1.0 / (1.0 + (now - s.created_at).days) for s in candidates])
        else:  # uniform
            weights = np.ones(len(candidates))

        # Normalize weights
        weights = weights / weights.sum()

        # Sample
        batch_size = min(batch_size, len(candidates))
        indices = np.random.choice(
            len(candidates),
            size=batch_size,
            replace=False,
            p=weights,
        )

        samples = [candidates[i] for i in indices]

        # Update access counts
        for sample in samples:
            sample.access_count += 1
            sample.last_accessed = datetime.now()

        self.total_sampled += len(samples)

        return samples

    def get_statistics(self) -> dict[str, Any]:
        """Get buffer statistics."""
        if len(self.buffer) == 0:
            return {
                "size": 0,
                "total_added": self.total_added,
                "total_sampled": self.total_sampled,
            }

        qualities = [s.quality_score for s in self.buffer]

        return {
            "size": len(self.buffer),
            "capacity": self.max_size,
            "utilization": len(self.buffer) / self.max_size,
            "total_added": self.total_added,
            "total_sampled": self.total_sampled,
            "avg_quality": float(np.mean(qualities)),
            "min_quality": float(np.min(qualities)),
            "max_quality": float(np.max(qualities)),
            "unique_topics": len(self._by_topic),
            "unique_tasks": len(self._by_task),
        }

    def _prune_indices(self):
        """Remove stale entries from indices."""
        current_ids = {s.id for s in self.buffer}

        # Prune quality index
        self._by_quality = {k: v for k, v in self._by_quality.items() if k in current_ids}

        # Prune topic index
        for topic in list(self._by_topic.keys()):
            self._by_topic[topic] = [id for id in self._by_topic[topic] if id in current_ids]
            if not self._by_topic[topic]:
                del self._by_topic[topic]

        # Prune task index
        for task in list(self._by_task.keys()):
            self._by_task[task] = [id for id in self._by_task[task] if id in current_ids]
            if not self._by_task[task]:
                del self._by_task[task]

    def save(self):
        """Save buffer to disk."""
        buffer_file = self.data_dir / "replay_buffer.json"

        try:
            data = {
                "samples": [s.to_dict() for s in self.buffer],
                "total_added": self.total_added,
                "total_sampled": self.total_sampled,
                "saved_at": datetime.now().isoformat(),
            }

            with open(buffer_file, "w") as f:
                json.dump(data, f)

            logger.debug(f"Saved replay buffer with {len(self.buffer)} samples")
        except Exception as e:
            logger.warning(f"Failed to save replay buffer: {e}")

    def _load(self):
        """Load buffer from disk."""
        buffer_file = self.data_dir / "replay_buffer.json"

        if not buffer_file.exists():
            return

        try:
            with open(buffer_file) as f:
                data = json.load(f)

            for sample_data in data.get("samples", []):
                sample = ReplayBufferSample.from_dict(sample_data)
                self.buffer.append(sample)

                # Rebuild indices
                self._by_quality[sample.id] = sample.quality_score
                if sample.topic_context:
                    self._by_topic[sample.topic_context].append(sample.id)
                if sample.task_context:
                    self._by_task[sample.task_context].append(sample.id)

            self.total_added = data.get("total_added", len(self.buffer))
            self.total_sampled = data.get("total_sampled", 0)

            logger.debug(f"Loaded replay buffer with {len(self.buffer)} samples")
        except Exception as e:
            logger.warning(f"Failed to load replay buffer: {e}")


class FastSlowWeightManager:
    """
    Manages Fast/Slow weight separation for continual learning.

    Based on Nested Learning (Google, 2025):
    - Fast weights adapt to current task quickly
    - Slow weights consolidate knowledge over time
    - Replay buffer prevents catastrophic forgetting
    """

    def __init__(
        self,
        config: FastSlowConfig | None = None,
        data_dir: str = "data/learning",
    ):
        """
        Initialize Fast/Slow weight manager.

        Args:
            config: Configuration settings
            data_dir: Directory for persistence
        """
        self.config = config or FastSlowConfig()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            max_size=self.config.replay_buffer_size,
            min_quality=self.config.min_quality_for_replay,
            data_dir=str(self.data_dir / "replay"),
        )

        # Consolidated parameter state (for drift penalty)
        self._consolidated_state: dict[str, np.ndarray] | None = None

        # Tracking
        self.interaction_count = 0
        self.fast_updates = 0
        self.slow_updates = 0
        self.last_consolidation: datetime | None = None

        # Pending samples for next slow update
        self._pending_samples: list[ReplayBufferSample] = []

        self._load_state()

    def record_interaction(
        self,
        input_text: str,
        output_text: str,
        quality_score: float,
        emotional_context: dict[str, float] | None = None,
        task_context: str = "",
        topic_context: str = "",
    ) -> dict[str, Any]:
        """
        Record an interaction for learning.

        Args:
            input_text: User input
            output_text: Model output
            quality_score: Quality rating
            emotional_context: Emotional state
            task_context: Current task
            topic_context: Current topic

        Returns:
            Learning action taken (fast_update, slow_update, or none)
        """
        self.interaction_count += 1

        result = {
            "interaction_count": self.interaction_count,
            "action": "none",
            "added_to_replay": False,
            "should_fast_update": False,
            "should_slow_update": False,
        }

        # Add to replay buffer if quality is good
        sample_id = self.replay_buffer.add(
            input_text=input_text,
            output_text=output_text,
            quality_score=quality_score,
            emotional_context=emotional_context,
            task_context=task_context,
            topic_context=topic_context,
        )

        result["added_to_replay"] = sample_id is not None

        # Track for slow update
        if sample_id:
            sample = ReplayBufferSample(
                id=sample_id,
                input_text=input_text,
                output_text=output_text,
                quality_score=quality_score,
                emotional_context=emotional_context or {},
                task_context=task_context,
                topic_context=topic_context,
            )
            self._pending_samples.append(sample)

        # Check if fast update needed
        if self.interaction_count % self.config.fast_update_interval == 0:
            result["should_fast_update"] = True
            result["action"] = "fast_update"
            self.fast_updates += 1

        # Check if slow update (consolidation) needed
        if self.interaction_count % self.config.slow_update_interval == 0:
            result["should_slow_update"] = True
            result["action"] = "slow_update"
            self.slow_updates += 1
            self.last_consolidation = datetime.now()

        return result

    def get_fast_update_batch(self) -> list[ReplayBufferSample]:
        """
        Get batch for fast weight update.

        Returns recent high-quality samples.
        """
        # Use pending samples (most recent interactions)
        batch = self._pending_samples[-self.config.replay_batch_size :]
        return batch

    def get_slow_update_batch(
        self,
        topic_filter: str | None = None,
    ) -> list[ReplayBufferSample]:
        """
        Get batch for slow weight update (consolidation).

        Mixes recent samples with replay buffer samples.
        """
        batch_size = self.config.replay_batch_size
        new_portion = int(batch_size * self.config.consolidation_mix_ratio)
        replay_portion = batch_size - new_portion

        # Recent samples
        recent = self._pending_samples[-new_portion:] if self._pending_samples else []

        # Replay samples (quality-weighted)
        replay = self.replay_buffer.sample(
            batch_size=replay_portion,
            strategy="quality_weighted",
            topic_filter=topic_filter,
        )

        # Clear pending after consolidation
        self._pending_samples = []

        return recent + replay

    def compute_drift_penalty(
        self,
        current_params: dict[str, np.ndarray],
    ) -> float:
        """
        Compute penalty for drifting from consolidated state.

        This prevents catastrophic forgetting during fast updates.

        Args:
            current_params: Current parameter values

        Returns:
            Drift penalty value
        """
        if self._consolidated_state is None:
            return 0.0

        total_drift = 0.0
        num_params = 0

        for name, value in current_params.items():
            if name in self._consolidated_state:
                drift = np.sum((value - self._consolidated_state[name]) ** 2)
                total_drift += drift
                num_params += 1

        if num_params == 0:
            return 0.0

        avg_drift = total_drift / num_params
        penalty = self.config.drift_penalty_weight * avg_drift

        return float(penalty)

    def update_consolidated_state(
        self,
        params: dict[str, np.ndarray],
    ):
        """
        Update the consolidated parameter state.

        Called after slow (consolidation) updates.

        Args:
            params: New consolidated parameters
        """
        self._consolidated_state = {
            name: value.copy() if isinstance(value, np.ndarray) else np.array(value)
            for name, value in params.items()
        }

        self._save_state()

    def get_statistics(self) -> dict[str, Any]:
        """Get manager statistics."""
        return {
            "interaction_count": self.interaction_count,
            "fast_updates": self.fast_updates,
            "slow_updates": self.slow_updates,
            "pending_samples": len(self._pending_samples),
            "has_consolidated_state": self._consolidated_state is not None,
            "last_consolidation": (
                self.last_consolidation.isoformat() if self.last_consolidation else None
            ),
            "replay_buffer": self.replay_buffer.get_statistics(),
            "config": self.config.to_dict(),
        }

    def _save_state(self):
        """Save state to disk."""
        state_file = self.data_dir / "fast_slow_state.json"

        try:
            data = {
                "interaction_count": self.interaction_count,
                "fast_updates": self.fast_updates,
                "slow_updates": self.slow_updates,
                "last_consolidation": (
                    self.last_consolidation.isoformat() if self.last_consolidation else None
                ),
                "saved_at": datetime.now().isoformat(),
            }

            with open(state_file, "w") as f:
                json.dump(data, f, indent=2)

            # Save consolidated state separately (can be large)
            if self._consolidated_state:
                consolidated_file = self.data_dir / "consolidated_params.npz"
                np.savez(consolidated_file, **self._consolidated_state)

            # Save replay buffer
            self.replay_buffer.save()

        except Exception as e:
            logger.warning(f"Failed to save fast/slow state: {e}")

    def _load_state(self):
        """Load state from disk."""
        state_file = self.data_dir / "fast_slow_state.json"

        if state_file.exists():
            try:
                with open(state_file) as f:
                    data = json.load(f)

                self.interaction_count = data.get("interaction_count", 0)
                self.fast_updates = data.get("fast_updates", 0)
                self.slow_updates = data.get("slow_updates", 0)

                if data.get("last_consolidation"):
                    self.last_consolidation = datetime.fromisoformat(data["last_consolidation"])

            except Exception as e:
                logger.warning(f"Failed to load fast/slow state: {e}")

        # Load consolidated params
        consolidated_file = self.data_dir / "consolidated_params.npz"
        if consolidated_file.exists():
            try:
                loaded = np.load(consolidated_file)
                self._consolidated_state = {key: loaded[key] for key in loaded.files}
            except Exception as e:
                logger.warning(f"Failed to load consolidated params: {e}")


def create_nested_learning_system(
    data_dir: str = "data/learning",
    fast_slow_config: FastSlowConfig | None = None,
) -> tuple[NestedLearningContext, FastSlowWeightManager]:
    """
    Factory function to create the complete nested learning system.

    Creates both the context manager and fast/slow weight manager.

    Args:
        data_dir: Base directory for data
        fast_slow_config: Configuration for Fast/Slow weights

    Returns:
        Tuple of (NestedLearningContext, FastSlowWeightManager)
    """
    context_manager = NestedLearningContext(data_dir=data_dir)
    weight_manager = FastSlowWeightManager(
        config=fast_slow_config,
        data_dir=data_dir,
    )

    return context_manager, weight_manager
