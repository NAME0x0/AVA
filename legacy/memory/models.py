"""
Memory Models for AVA

Defines data structures for different types of memories including
episodic (event-based) and semantic (factual knowledge) memories.

Also implements Titans-style Neural Memory for test-time learning.
Reference: "Titans: Learning to Memorize at Test Time" (arXiv:2501.00663)
"""

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Try to import torch for TitansMemory, fallback to numpy-only mode
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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


# =============================================================================
# TITANS NEURAL MEMORY - Test-Time Learning
# =============================================================================
# Reference: "Titans: Learning to Memorize at Test Time" (arXiv:2501.00663)
#
# Key concepts:
# - Neural Memory MLP that stores information in its weights
# - Surprise signal (reconstruction error) determines what to memorize
# - Momentum buffer aggregates recent surprises
# - Forgetting gate decays old, low-importance information
# =============================================================================


@dataclass
class SurpriseMetrics:
    """Metrics from the surprise computation."""
    surprise_value: float = 0.0
    reconstruction_error: float = 0.0
    momentum_value: float = 0.0
    forget_gate_value: float = 0.0
    memory_updated: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "surprise_value": self.surprise_value,
            "reconstruction_error": self.reconstruction_error,
            "momentum_value": self.momentum_value,
            "forget_gate_value": self.forget_gate_value,
            "memory_updated": self.memory_updated,
            "timestamp": self.timestamp.isoformat(),
        }


class TitansMemoryNumpy:
    """
    NumPy-based Titans Memory for environments without PyTorch.
    
    This is a simplified version that uses numpy arrays for
    the memory MLP and gradient-based updates.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        memory_dim: int = 2048,
        learning_rate: float = 0.01,
        momentum_decay: float = 0.99,
        surprise_threshold: float = 0.5,
    ):
        """
        Initialize NumPy-based Titans memory.
        
        Args:
            input_dim: Dimension of input embeddings
            memory_dim: Dimension of internal memory
            learning_rate: Learning rate for weight updates
            momentum_decay: Decay factor for momentum buffer
            surprise_threshold: Threshold for triggering updates
        """
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.learning_rate = learning_rate
        self.momentum_decay = momentum_decay
        self.surprise_threshold = surprise_threshold
        
        # Initialize memory MLP weights (He initialization)
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / memory_dim)
        
        self.W1 = np.random.randn(input_dim, memory_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(memory_dim, dtype=np.float32)
        self.W2 = np.random.randn(memory_dim, input_dim).astype(np.float32) * scale2
        self.b2 = np.zeros(input_dim, dtype=np.float32)
        
        # Surprise scale (learnable in full version)
        self.surprise_scale = 0.1
        
        # Momentum buffer for aggregating surprises
        self.momentum_buffer = np.zeros(memory_dim, dtype=np.float32)
        
        # Forget gate weights
        self.W_forget = np.random.randn(input_dim, 1).astype(np.float32) * 0.1
        self.b_forget = np.zeros(1, dtype=np.float32)
        
        # Statistics tracking
        self.total_updates = 0
        self.recent_surprises: List[float] = []
        
    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation function."""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def _gelu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of GELU for backpropagation."""
        cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        pdf = np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
        return cdf + x * pdf
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Forward pass through memory MLP.
        
        Args:
            x: Input embedding (input_dim,) or (batch, input_dim)
            
        Returns:
            Tuple of (output, cache) for backward pass
        """
        # Ensure 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Layer 1
        z1 = x @ self.W1 + self.b1
        a1 = self._gelu(z1)
        
        # Layer 2
        z2 = a1 @ self.W2 + self.b2
        output = z2  # No activation on output
        
        cache = {
            "x": x,
            "z1": z1,
            "a1": a1,
            "z2": z2,
        }
        
        return output, cache
    
    def compute_surprise(
        self,
        hidden_state: np.ndarray,
    ) -> Tuple[float, np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute surprise signal based on reconstruction error.
        
        The surprise measures how novel/unexpected the input is
        relative to what the memory can reconstruct.
        
        Args:
            hidden_state: Current hidden state embedding
            
        Returns:
            Tuple of (surprise_value, memory_output, cache)
        """
        # Forward pass
        memory_output, cache = self.forward(hidden_state)
        
        # Compute reconstruction error (MSE)
        reconstruction_error = np.mean((hidden_state - memory_output) ** 2)
        
        # Scale to get surprise signal
        surprise = float(reconstruction_error * self.surprise_scale)
        
        # Track for statistics
        self.recent_surprises.append(surprise)
        if len(self.recent_surprises) > 100:
            self.recent_surprises = self.recent_surprises[-100:]
        
        return surprise, memory_output.squeeze(), cache
    
    def compute_forget_gate(self, hidden_state: np.ndarray) -> float:
        """
        Compute forget gate value to determine decay strength.
        
        Args:
            hidden_state: Current hidden state
            
        Returns:
            Forget probability (0 to 1)
        """
        if hidden_state.ndim == 1:
            hidden_state = hidden_state.reshape(1, -1)
        
        logit = hidden_state @ self.W_forget + self.b_forget
        forget_prob = float(self._sigmoid(logit).squeeze())
        
        return forget_prob
    
    def update(
        self,
        hidden_state: np.ndarray,
        force_update: bool = False,
    ) -> SurpriseMetrics:
        """
        Update memory weights based on surprise signal.
        
        This is the core of test-time learning: the memory
        updates its weights during inference when it encounters
        surprising/novel information.
        
        Args:
            hidden_state: Current hidden state to learn from
            force_update: Force update even if below threshold
            
        Returns:
            SurpriseMetrics with update details
        """
        # Ensure proper shape
        if hidden_state.ndim == 1:
            hidden_state = hidden_state.reshape(1, -1)
        
        # Compute surprise
        surprise, memory_output, cache = self.compute_surprise(hidden_state)
        
        # Compute forget gate
        forget_prob = self.compute_forget_gate(hidden_state)
        
        # Update momentum buffer
        self.momentum_buffer = (
            self.momentum_decay * self.momentum_buffer +
            (1 - self.momentum_decay) * cache["a1"].squeeze()
        )
        
        metrics = SurpriseMetrics(
            surprise_value=surprise,
            reconstruction_error=float(np.mean((hidden_state - memory_output.reshape(1, -1)) ** 2)),
            momentum_value=float(np.mean(self.momentum_buffer)),
            forget_gate_value=forget_prob,
            memory_updated=False,
        )
        
        # Only update if surprise exceeds threshold or forced
        if surprise < self.surprise_threshold and not force_update:
            return metrics
        
        # Gradient-based update (simplified backprop)
        # Compute gradients of reconstruction loss
        batch_size = hidden_state.shape[0]
        
        # Output gradient
        d_output = 2.0 * (memory_output - hidden_state) / batch_size
        
        # Layer 2 gradients
        d_W2 = cache["a1"].T @ d_output
        d_b2 = np.sum(d_output, axis=0)
        
        # Backprop through layer 2
        d_a1 = d_output @ self.W2.T
        
        # GELU gradient
        d_z1 = d_a1 * self._gelu_derivative(cache["z1"])
        
        # Layer 1 gradients
        d_W1 = cache["x"].T @ d_z1
        d_b1 = np.sum(d_z1, axis=0)
        
        # Apply updates (scaled by surprise - larger surprise = larger update)
        update_scale = self.learning_rate * min(surprise, 1.0)
        
        self.W1 -= update_scale * d_W1
        self.b1 -= update_scale * d_b1
        self.W2 -= update_scale * d_W2
        self.b2 -= update_scale * d_b2
        
        # Apply forgetting (decay weights slightly based on forget gate)
        decay_factor = 1.0 - (0.01 * forget_prob)
        self.W1 *= decay_factor
        self.W2 *= decay_factor
        
        self.total_updates += 1
        metrics.memory_updated = True
        
        return metrics
    
    def retrieve(self, query: np.ndarray) -> np.ndarray:
        """
        Retrieve from memory given a query embedding.
        
        Args:
            query: Query embedding
            
        Returns:
            Memory-augmented output
        """
        output, _ = self.forward(query)
        return output.squeeze()
    
    def get_state(self) -> Dict[str, Any]:
        """Get current memory state for persistence."""
        return {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "W_forget": self.W_forget.tolist(),
            "b_forget": self.b_forget.tolist(),
            "momentum_buffer": self.momentum_buffer.tolist(),
            "surprise_scale": self.surprise_scale,
            "total_updates": self.total_updates,
            "config": {
                "input_dim": self.input_dim,
                "memory_dim": self.memory_dim,
                "learning_rate": self.learning_rate,
                "momentum_decay": self.momentum_decay,
                "surprise_threshold": self.surprise_threshold,
            }
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load memory state from persistence."""
        self.W1 = np.array(state["W1"], dtype=np.float32)
        self.b1 = np.array(state["b1"], dtype=np.float32)
        self.W2 = np.array(state["W2"], dtype=np.float32)
        self.b2 = np.array(state["b2"], dtype=np.float32)
        self.W_forget = np.array(state["W_forget"], dtype=np.float32)
        self.b_forget = np.array(state["b_forget"], dtype=np.float32)
        self.momentum_buffer = np.array(state["momentum_buffer"], dtype=np.float32)
        self.surprise_scale = state.get("surprise_scale", 0.1)
        self.total_updates = state.get("total_updates", 0)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get memory statistics."""
        return {
            "total_updates": self.total_updates,
            "avg_recent_surprise": float(np.mean(self.recent_surprises)) if self.recent_surprises else 0.0,
            "max_recent_surprise": float(np.max(self.recent_surprises)) if self.recent_surprises else 0.0,
            "momentum_magnitude": float(np.linalg.norm(self.momentum_buffer)),
            "weight_magnitude_W1": float(np.linalg.norm(self.W1)),
            "weight_magnitude_W2": float(np.linalg.norm(self.W2)),
        }


if TORCH_AVAILABLE:
    class TitansMemoryTorch(nn.Module):
        """
        PyTorch-based Titans Memory for test-time learning.
        
        Implements the neural memory module from Titans (2025):
        - Memory MLP that stores information in weights
        - Surprise-driven test-time gradient updates
        - Momentum buffer for temporal aggregation
        - Learnable forget gate for adaptive decay
        """
        
        def __init__(
            self,
            input_dim: int = 768,
            memory_dim: int = 2048,
            learning_rate: float = 0.01,
            momentum_decay: float = 0.99,
            surprise_threshold: float = 0.5,
            device: str = "cpu",
        ):
            """
            Initialize PyTorch Titans memory.
            
            Args:
                input_dim: Dimension of input embeddings
                memory_dim: Internal memory dimension
                learning_rate: Learning rate for test-time updates
                momentum_decay: Decay for momentum buffer
                surprise_threshold: Threshold to trigger updates
                device: Device to use (cpu/cuda)
            """
            super().__init__()
            
            self.input_dim = input_dim
            self.memory_dim = memory_dim
            self.learning_rate = learning_rate
            self.momentum_decay = momentum_decay
            self.surprise_threshold = surprise_threshold
            self.device = device
            
            # Memory MLP
            self.memory_mlp = nn.Sequential(
                nn.Linear(input_dim, memory_dim),
                nn.GELU(),
                nn.Linear(memory_dim, input_dim),
            ).to(device)
            
            # Learnable surprise scale
            self.surprise_scale = nn.Parameter(torch.tensor(0.1, device=device))
            
            # Momentum buffer (not a parameter, just state)
            self.register_buffer(
                "momentum_buffer",
                torch.zeros(memory_dim, device=device)
            )
            
            # Forget gate
            self.forget_gate = nn.Sequential(
                nn.Linear(input_dim, 1),
                nn.Sigmoid(),
            ).to(device)
            
            # Statistics
            self.total_updates = 0
            self.recent_surprises: List[float] = []
        
        def forward(
            self,
            x: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass: retrieve from memory.
            
            Args:
                x: Input tensor (batch, input_dim)
                
            Returns:
                Tuple of (memory_output, hidden_state_after_first_layer)
            """
            # Get intermediate for momentum
            hidden = self.memory_mlp[0](x)  # First linear
            hidden_activated = self.memory_mlp[1](hidden)  # GELU
            output = self.memory_mlp[2](hidden_activated)  # Second linear
            
            return output, hidden_activated
        
        def compute_surprise(
            self,
            hidden_state: torch.Tensor,
        ) -> Tuple[float, torch.Tensor, torch.Tensor]:
            """
            Compute surprise signal.
            
            Args:
                hidden_state: Current hidden state
                
            Returns:
                Tuple of (surprise_float, memory_output, hidden_for_momentum)
            """
            memory_output, hidden_for_momentum = self.forward(hidden_state)
            
            # Reconstruction error
            reconstruction_error = F.mse_loss(memory_output, hidden_state)
            
            # Scale by learnable parameter
            surprise = (reconstruction_error * self.surprise_scale).item()
            
            # Track statistics
            self.recent_surprises.append(surprise)
            if len(self.recent_surprises) > 100:
                self.recent_surprises = self.recent_surprises[-100:]
            
            return surprise, memory_output, hidden_for_momentum
        
        @torch.enable_grad()
        def update(
            self,
            hidden_state: torch.Tensor,
            force_update: bool = False,
        ) -> SurpriseMetrics:
            """
            Update memory weights via test-time gradient descent.
            
            Args:
                hidden_state: Hidden state to learn from
                force_update: Force update even if below threshold
                
            Returns:
                SurpriseMetrics with update details
            """
            # Ensure requires grad for backprop
            hidden_state = hidden_state.detach().requires_grad_(False)
            
            # Compute surprise
            surprise, memory_output, hidden_for_momentum = self.compute_surprise(hidden_state)
            
            # Compute forget gate
            forget_prob = self.forget_gate(hidden_state).mean().item()
            
            # Update momentum buffer
            with torch.no_grad():
                self.momentum_buffer = (
                    self.momentum_decay * self.momentum_buffer +
                    (1 - self.momentum_decay) * hidden_for_momentum.mean(dim=0)
                )
            
            metrics = SurpriseMetrics(
                surprise_value=surprise,
                reconstruction_error=F.mse_loss(memory_output, hidden_state).item(),
                momentum_value=self.momentum_buffer.mean().item(),
                forget_gate_value=forget_prob,
                memory_updated=False,
            )
            
            # Only update if surprise exceeds threshold
            if surprise < self.surprise_threshold and not force_update:
                return metrics
            
            # Compute loss and gradients
            loss = F.mse_loss(memory_output, hidden_state)
            
            # Manual gradient update (test-time training)
            grads = torch.autograd.grad(
                loss,
                self.memory_mlp.parameters(),
                retain_graph=False,
                create_graph=False,
            )
            
            # Apply updates scaled by surprise
            update_scale = self.learning_rate * min(surprise, 1.0)
            
            with torch.no_grad():
                for param, grad in zip(self.memory_mlp.parameters(), grads):
                    param.data -= update_scale * grad
                    
                    # Apply forget gate decay
                    param.data *= (1.0 - 0.01 * forget_prob)
            
            self.total_updates += 1
            metrics.memory_updated = True
            
            return metrics
        
        def retrieve(self, query: torch.Tensor) -> torch.Tensor:
            """
            Retrieve from memory.
            
            Args:
                query: Query tensor
                
            Returns:
                Memory-augmented output
            """
            with torch.no_grad():
                output, _ = self.forward(query)
            return output
        
        def get_statistics(self) -> Dict[str, float]:
            """Get memory statistics."""
            return {
                "total_updates": self.total_updates,
                "avg_recent_surprise": float(np.mean(self.recent_surprises)) if self.recent_surprises else 0.0,
                "max_recent_surprise": float(np.max(self.recent_surprises)) if self.recent_surprises else 0.0,
                "momentum_magnitude": self.momentum_buffer.norm().item(),
                "surprise_scale": self.surprise_scale.item(),
            }


def create_titans_memory(
    input_dim: int = 768,
    memory_dim: int = 2048,
    learning_rate: float = 0.01,
    momentum_decay: float = 0.99,
    surprise_threshold: float = 0.5,
    use_torch: Optional[bool] = None,
    device: str = "cpu",
) -> "TitansMemoryNumpy | TitansMemoryTorch":
    """
    Factory function to create appropriate Titans memory.
    
    Creates PyTorch version if available and requested,
    otherwise falls back to NumPy implementation.
    
    Args:
        input_dim: Input embedding dimension
        memory_dim: Internal memory dimension
        learning_rate: Test-time learning rate
        momentum_decay: Momentum buffer decay
        surprise_threshold: Threshold for updates
        use_torch: Force PyTorch (True) or NumPy (False), None = auto
        device: Device for PyTorch version
        
    Returns:
        TitansMemory instance
    """
    if use_torch is None:
        use_torch = TORCH_AVAILABLE
    
    if use_torch and TORCH_AVAILABLE:
        return TitansMemoryTorch(
            input_dim=input_dim,
            memory_dim=memory_dim,
            learning_rate=learning_rate,
            momentum_decay=momentum_decay,
            surprise_threshold=surprise_threshold,
            device=device,
        )
    else:
        return TitansMemoryNumpy(
            input_dim=input_dim,
            memory_dim=memory_dim,
            learning_rate=learning_rate,
            momentum_decay=momentum_decay,
            surprise_threshold=surprise_threshold,
        )


@dataclass
class TitansMemoryConfig:
    """Configuration for Titans memory module."""
    input_dim: int = 768
    memory_dim: int = 2048
    learning_rate: float = 0.01
    momentum_decay: float = 0.99
    surprise_threshold: float = 0.5
    use_torch: bool = True
    device: str = "cpu"
    
    # Integration settings
    blend_weight: float = 0.3  # How much memory contributes to final output
    update_on_every_token: bool = False  # Update per token or per sequence
    persist_memory: bool = True  # Save memory state between sessions
    memory_file: str = "data/memory/titans_state.json"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_dim": self.input_dim,
            "memory_dim": self.memory_dim,
            "learning_rate": self.learning_rate,
            "momentum_decay": self.momentum_decay,
            "surprise_threshold": self.surprise_threshold,
            "use_torch": self.use_torch,
            "device": self.device,
            "blend_weight": self.blend_weight,
            "update_on_every_token": self.update_on_every_token,
            "persist_memory": self.persist_memory,
            "memory_file": self.memory_file,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TitansMemoryConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
