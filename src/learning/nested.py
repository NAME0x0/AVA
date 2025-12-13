"""
Nested Learning Contexts for AVA

Implements meta-learning through nested contexts that allow AVA
to learn different things at different scopes:
- Session-level: Learning within a conversation
- Topic-level: Learning about specific topics
- Task-level: Learning how to do specific tasks
- Global-level: Overall behavioral learning
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LearningScope(Enum):
    """Scope levels for nested learning."""
    GLOBAL = "global"         # Overall behavioral patterns
    TOPIC = "topic"           # Topic-specific knowledge
    TASK = "task"             # Task-specific procedures
    SESSION = "session"       # Current session only
    EPHEMERAL = "ephemeral"   # Single interaction


@dataclass
class LearningContext:
    """A context for learning at a specific scope."""
    
    id: str = ""
    name: str = ""
    scope: LearningScope = LearningScope.SESSION
    created_at: datetime = field(default_factory=datetime.now)
    
    # Context data
    parent_id: Optional[str] = None
    topic: str = ""
    task: str = ""
    
    # Learning state within this context
    observations: List[str] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    patterns: Dict[str, Any] = field(default_factory=dict)
    
    # Performance in this context
    successes: int = 0
    failures: int = 0
    corrections: int = 0
    
    # Active state
    is_active: bool = True
    closed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "LearningContext":
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
        self.context_stack: List[LearningContext] = []
        
        # All contexts (indexed by id)
        self.contexts: Dict[str, LearningContext] = {}
        
        # Global context (always present)
        self.global_context = self._get_or_create_global_context()
        
        # Current session context
        self.session_context: Optional[LearningContext] = None
        
        self._load_contexts()
    
    def _get_or_create_global_context(self) -> LearningContext:
        """Get or create the global learning context."""
        global_file = self.data_dir / "global_context.json"
        
        if global_file.exists():
            try:
                with open(global_file, "r") as f:
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
    
    def start_session(self, session_id: Optional[str] = None) -> LearningContext:
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
        parent_context: Optional[LearningContext] = None,
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
        parent_context: Optional[LearningContext] = None,
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
    
    def exit_context(self) -> Optional[LearningContext]:
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
    
    def record_observation(self, observation: str, context: Optional[LearningContext] = None):
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
    
    def record_insight(self, insight: str, context: Optional[LearningContext] = None):
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
        context: Optional[LearningContext] = None,
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
    
    def get_relevant_context(self, topic: Optional[str] = None, task: Optional[str] = None) -> Dict[str, Any]:
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
            "session_success_rate": self.session_context.success_rate if self.session_context else None,
        }
    
    def get_learning_summary(self) -> Dict[str, Any]:
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
                with open(contexts_file, "r") as f:
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
