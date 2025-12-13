"""
Memory Models for AVA

Defines data structures for different types of memories including
episodic (event-based) and semantic (factual knowledge) memories.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class MemoryType(Enum):
    """Types of memory storage."""
    EPISODIC = "episodic"    # Event-based memories (interactions)
    SEMANTIC = "semantic"    # Factual knowledge


@dataclass
class MemoryItem:
    """
    Base memory item structure.

    All memories share these common attributes for storage,
    retrieval, and consolidation.
    """
    # Unique identifier
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Memory classification
    memory_type: MemoryType = MemoryType.EPISODIC

    # Core content
    content: str = ""
    summary: str = ""  # Brief summary for quick retrieval

    # Metadata
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Temporal information
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    # Emotional tagging
    emotional_valence: float = 0.0    # -1.0 to +1.0
    emotional_intensity: float = 0.0  # 0.0 to 1.0
    associated_emotions: List[str] = field(default_factory=list)

    # Memory strength (for consolidation)
    strength: float = 1.0        # Decays over time, strengthens with access
    importance_score: float = 0.5  # Base importance rating

    # Developmental context
    stage_when_formed: int = 0   # DevelopmentalStage value when created

    # Embedding for semantic search (computed on demand)
    embedding: Optional[List[float]] = None

    def access(self):
        """Record an access to this memory, strengthening it."""
        self.last_accessed = datetime.now()
        self.access_count += 1
        # Strengthen memory when accessed
        self.strength = min(1.0, self.strength + 0.1)

    def decay(self, amount: float):
        """Apply decay to memory strength."""
        self.strength = max(0.0, self.strength - amount)

    def compute_relevance_score(
        self,
        recency_weight: float = 0.3,
        strength_weight: float = 0.3,
        importance_weight: float = 0.2,
        access_weight: float = 0.2,
    ) -> float:
        """
        Compute overall relevance score for retrieval ranking.

        Combines multiple factors into a single score (0.0 to 1.0).
        """
        # Recency score (exponential decay based on time)
        days_ago = (datetime.now() - self.last_accessed).days
        recency_score = max(0.0, 1.0 - (days_ago * 0.01))

        # Access frequency score (log scale)
        import math
        access_score = min(1.0, math.log(self.access_count + 1) / 5)

        return (
            recency_weight * recency_score +
            strength_weight * self.strength +
            importance_weight * self.importance_score +
            access_weight * access_score
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "summary": self.summary,
            "context": self.context,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "emotional_valence": self.emotional_valence,
            "emotional_intensity": self.emotional_intensity,
            "associated_emotions": self.associated_emotions,
            "strength": self.strength,
            "importance_score": self.importance_score,
            "stage_when_formed": self.stage_when_formed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        """Create from dictionary."""
        item = cls()
        item.memory_id = data.get("memory_id", item.memory_id)
        item.memory_type = MemoryType(data.get("memory_type", "episodic"))
        item.content = data.get("content", "")
        item.summary = data.get("summary", "")
        item.context = data.get("context", {})
        item.tags = data.get("tags", [])
        if data.get("created_at"):
            item.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("last_accessed"):
            item.last_accessed = datetime.fromisoformat(data["last_accessed"])
        item.access_count = data.get("access_count", 0)
        item.emotional_valence = data.get("emotional_valence", 0.0)
        item.emotional_intensity = data.get("emotional_intensity", 0.0)
        item.associated_emotions = data.get("associated_emotions", [])
        item.strength = data.get("strength", 1.0)
        item.importance_score = data.get("importance_score", 0.5)
        item.stage_when_formed = data.get("stage_when_formed", 0)
        return item


@dataclass
class EpisodicMemory(MemoryItem):
    """
    Episodic memory - records specific interactions/events.

    Stores the complete context of an interaction including
    user input, AVA's response, tool usage, and outcome.
    """
    memory_type: MemoryType = MemoryType.EPISODIC

    # Interaction details
    user_input: str = ""
    ava_response: str = ""

    # Tool usage during this episode
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    # Outcome assessment
    outcome: str = "unknown"  # "success", "partial", "failure", "unknown"
    quality_score: float = 0.5

    # Conversation context
    turn_number: int = 0
    conversation_id: Optional[str] = None

    # Learning signals
    user_feedback: Optional[str] = None  # "positive", "negative", "neutral"
    was_corrected: bool = False
    correction_content: Optional[str] = None

    # Links to other memories
    related_memories: List[str] = field(default_factory=list)  # Memory IDs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = super().to_dict()
        data.update({
            "user_input": self.user_input,
            "ava_response": self.ava_response,
            "tool_calls": self.tool_calls,
            "outcome": self.outcome,
            "quality_score": self.quality_score,
            "turn_number": self.turn_number,
            "conversation_id": self.conversation_id,
            "user_feedback": self.user_feedback,
            "was_corrected": self.was_corrected,
            "correction_content": self.correction_content,
            "related_memories": self.related_memories,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodicMemory":
        """Create from dictionary."""
        mem = cls()
        # Load base attributes
        base = MemoryItem.from_dict(data)
        for key, value in base.__dict__.items():
            setattr(mem, key, value)

        # Load episodic-specific attributes
        mem.memory_type = MemoryType.EPISODIC
        mem.user_input = data.get("user_input", "")
        mem.ava_response = data.get("ava_response", "")
        mem.tool_calls = data.get("tool_calls", [])
        mem.outcome = data.get("outcome", "unknown")
        mem.quality_score = data.get("quality_score", 0.5)
        mem.turn_number = data.get("turn_number", 0)
        mem.conversation_id = data.get("conversation_id")
        mem.user_feedback = data.get("user_feedback")
        mem.was_corrected = data.get("was_corrected", False)
        mem.correction_content = data.get("correction_content")
        mem.related_memories = data.get("related_memories", [])
        return mem


@dataclass
class SemanticMemory(MemoryItem):
    """
    Semantic memory - stores factual knowledge.

    Represents learned facts and concepts that can be used
    across different contexts.
    """
    memory_type: MemoryType = MemoryType.SEMANTIC

    # Knowledge structure (subject-predicate-object triple)
    subject: str = ""
    predicate: str = ""
    object: str = ""

    # Categorization
    domain: str = ""  # e.g., "science", "history", "user_preference"
    category: str = ""

    # Confidence and sourcing
    confidence: float = 0.5  # How confident AVA is in this knowledge
    source: str = ""         # Where this knowledge came from
    source_type: str = ""    # "interaction", "inference", "external"

    # Verification
    times_confirmed: int = 0
    times_contradicted: int = 0

    def update_confidence(self, confirmed: bool):
        """Update confidence based on confirmation or contradiction."""
        if confirmed:
            self.times_confirmed += 1
            self.confidence = min(1.0, self.confidence + 0.1)
        else:
            self.times_contradicted += 1
            self.confidence = max(0.0, self.confidence - 0.15)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = super().to_dict()
        data.update({
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "domain": self.domain,
            "category": self.category,
            "confidence": self.confidence,
            "source": self.source,
            "source_type": self.source_type,
            "times_confirmed": self.times_confirmed,
            "times_contradicted": self.times_contradicted,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticMemory":
        """Create from dictionary."""
        mem = cls()
        # Load base attributes
        base = MemoryItem.from_dict(data)
        for key, value in base.__dict__.items():
            setattr(mem, key, value)

        # Load semantic-specific attributes
        mem.memory_type = MemoryType.SEMANTIC
        mem.subject = data.get("subject", "")
        mem.predicate = data.get("predicate", "")
        mem.object = data.get("object", "")
        mem.domain = data.get("domain", "")
        mem.category = data.get("category", "")
        mem.confidence = data.get("confidence", 0.5)
        mem.source = data.get("source", "")
        mem.source_type = data.get("source_type", "")
        mem.times_confirmed = data.get("times_confirmed", 0)
        mem.times_contradicted = data.get("times_contradicted", 0)
        return mem


def create_episodic_memory(
    user_input: str,
    ava_response: str,
    outcome: str = "unknown",
    quality_score: float = 0.5,
    emotional_valence: float = 0.0,
    stage: int = 0,
    tool_calls: Optional[List[Dict]] = None,
    tags: Optional[List[str]] = None,
) -> EpisodicMemory:
    """
    Factory function to create an episodic memory.

    Args:
        user_input: The user's input
        ava_response: AVA's response
        outcome: Interaction outcome
        quality_score: Quality rating
        emotional_valence: Emotional context
        stage: Developmental stage when formed
        tool_calls: List of tool calls made
        tags: Tags for categorization
    """
    # Create summary from user input
    summary = user_input[:100] + "..." if len(user_input) > 100 else user_input

    memory = EpisodicMemory(
        content=f"User: {user_input}\nAVA: {ava_response}",
        summary=summary,
        user_input=user_input,
        ava_response=ava_response,
        outcome=outcome,
        quality_score=quality_score,
        emotional_valence=emotional_valence,
        stage_when_formed=stage,
        tool_calls=tool_calls or [],
        tags=tags or [],
    )

    # Set importance based on outcome and quality
    if outcome == "success" and quality_score > 0.7:
        memory.importance_score = 0.7
    elif outcome == "failure":
        memory.importance_score = 0.6  # Failures are important to learn from
    else:
        memory.importance_score = 0.5

    return memory


def create_semantic_memory(
    content: str,
    subject: str,
    predicate: str,
    obj: str,
    domain: str = "",
    source: str = "interaction",
    confidence: float = 0.5,
    stage: int = 0,
) -> SemanticMemory:
    """
    Factory function to create a semantic memory.

    Args:
        content: Full content/context
        subject: Subject of the fact
        predicate: Relationship/predicate
        obj: Object of the fact
        domain: Knowledge domain
        source: Where this came from
        confidence: Initial confidence
        stage: Developmental stage when formed
    """
    summary = f"{subject} {predicate} {obj}"

    return SemanticMemory(
        content=content,
        summary=summary,
        subject=subject,
        predicate=predicate,
        object=obj,
        domain=domain,
        source=source,
        source_type="interaction" if "interaction" in source.lower() else "inference",
        confidence=confidence,
        stage_when_formed=stage,
        importance_score=0.6,
    )
