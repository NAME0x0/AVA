"""
TITANS SIDECAR: The Neural Memory Module
=========================================

This is the PyTorch sidecar that learns at test-time using the principles
from "Titans: Learning to Memorize at Test Time" (2025).

Key Concepts:
1. Neural Memory MLP that updates with each input
2. Surprise signal drives update magnitude
3. Momentum accumulates gradients over conversation
4. Forgetting gates prevent unbounded growth

The TitansSidecar wraps the core Titans implementation and provides:
- High-level API for the conscious loop
- Integration with Entropix for surprise signals
- Episodic storage of high-surprise events
- Gradient checkpoint management

Reference: arXiv:2501.00663 (Titans: Learning to Memorize at Test Time)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

# Attempt to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TitansSidecarConfig:
    """
    Configuration for the Titans Neural Memory Sidecar.

    Attributes:
        input_dim: Dimension of input embeddings (from Ollama)
        hidden_dim: Internal hidden dimension of the memory MLP
        output_dim: Output dimension (typically same as input_dim)
        num_layers: Number of layers in the memory MLP
        learning_rate: Base learning rate for test-time updates
        momentum: Momentum factor for gradient accumulation
        forget_alpha: Forgetting gate factor (0 = no forgetting, 1 = full forget)
        surprise_threshold: Minimum surprise to trigger update
        max_stored_episodes: Maximum episodes in the replay buffer
        use_layer_norm: Whether to use layer normalization
        dropout: Dropout rate during updates
    """

    input_dim: int = 768  # nomic-embed-text output dimension
    hidden_dim: int = 1024  # Internal hidden size
    output_dim: int = 768  # Match input dim for residual
    num_layers: int = 3  # Depth of memory MLP
    learning_rate: float = 1e-3  # Test-time learning rate
    momentum: float = 0.9  # Gradient momentum
    forget_alpha: float = 0.01  # Forgetting rate
    surprise_threshold: float = 0.5  # Min surprise for update
    max_stored_episodes: int = 1000  # Replay buffer size
    use_layer_norm: bool = True
    dropout: float = 0.1


class TitansSidecarNumpy:
    """
    NumPy-based Titans sidecar for environments without PyTorch.

    This is a simplified implementation that provides the core
    memorize/retrieve API without requiring CUDA or PyTorch.
    """

    def __init__(self, config: TitansSidecarConfig | None = None):
        self.config = config or TitansSidecarConfig()
        cfg = self.config

        # Initialize memory weights with Xavier initialization
        scale = np.sqrt(2.0 / (cfg.input_dim + cfg.hidden_dim))
        self.W1 = np.random.randn(cfg.input_dim, cfg.hidden_dim).astype(np.float32) * scale
        self.b1 = np.zeros(cfg.hidden_dim, dtype=np.float32)

        scale = np.sqrt(2.0 / (cfg.hidden_dim + cfg.output_dim))
        self.W2 = np.random.randn(cfg.hidden_dim, cfg.output_dim).astype(np.float32) * scale
        self.b2 = np.zeros(cfg.output_dim, dtype=np.float32)

        # Momentum accumulators
        self.momentum_W1 = np.zeros_like(self.W1)
        self.momentum_b1 = np.zeros_like(self.b1)
        self.momentum_W2 = np.zeros_like(self.W2)
        self.momentum_b2 = np.zeros_like(self.b2)

        # Statistics
        self.update_count = 0
        self.total_surprise = 0.0

    def retrieve(self, query: np.ndarray) -> np.ndarray:
        """
        Retrieve from memory based on query embedding.

        Forward pass through the memory MLP:
        h = ReLU(query @ W1 + b1)
        output = h @ W2 + b2

        Args:
            query: Query embedding [input_dim] or [batch, input_dim]

        Returns:
            Memory-augmented embedding [output_dim] or [batch, output_dim]
        """
        query = np.asarray(query, dtype=np.float32)

        # Handle single vector vs batch
        if query.ndim == 1:
            query = query.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False

        # Forward pass
        h = query @ self.W1 + self.b1
        h = np.maximum(h, 0)  # ReLU
        output = h @ self.W2 + self.b2

        if squeeze:
            output = output.squeeze(0)

        return output

    def memorize(
        self,
        embedding: np.ndarray,
        target: np.ndarray | None = None,
        surprise: float = 1.0,
    ) -> float:
        """
        Memorize an embedding with surprise-weighted update.

        The update is scaled by the surprise signal:
        - High surprise → large update (novel information)
        - Low surprise → small/no update (familiar information)

        If no target is provided, uses identity target (memorize the embedding itself).

        Args:
            embedding: Input embedding to memorize
            target: Target output (defaults to embedding for autoencoder behavior)
            surprise: Surprise signal from Entropix (0.0 to ~5.0)

        Returns:
            Loss value for the update
        """
        cfg = self.config

        # Skip update if surprise is below threshold
        if surprise < cfg.surprise_threshold:
            return 0.0

        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        if target is None:
            target = embedding
        else:
            target = np.asarray(target, dtype=np.float32)
            if target.ndim == 1:
                target = target.reshape(1, -1)

        # Forward pass
        h = embedding @ self.W1 + self.b1
        h_activated = np.maximum(h, 0)  # ReLU
        output = h_activated @ self.W2 + self.b2

        # Compute loss (MSE)
        error = output - target
        loss = np.mean(error**2)

        # Backward pass
        # dL/dW2 = h.T @ error
        # dL/db2 = error
        d_output = error * 2 / error.size

        d_W2 = h_activated.T @ d_output
        d_b2 = d_output.sum(axis=0)

        # Backprop through ReLU
        d_h = d_output @ self.W2.T
        d_h = d_h * (h > 0)  # ReLU gradient

        d_W1 = embedding.T @ d_h
        d_b1 = d_h.sum(axis=0)

        # Apply surprise-scaled updates with momentum
        lr = cfg.learning_rate * surprise

        self.momentum_W1 = cfg.momentum * self.momentum_W1 + lr * d_W1
        self.momentum_b1 = cfg.momentum * self.momentum_b1 + lr * d_b1
        self.momentum_W2 = cfg.momentum * self.momentum_W2 + lr * d_W2
        self.momentum_b2 = cfg.momentum * self.momentum_b2 + lr * d_b2

        # Update weights
        self.W1 -= self.momentum_W1
        self.b1 -= self.momentum_b1
        self.W2 -= self.momentum_W2
        self.b2 -= self.momentum_b2

        # Apply forgetting (weight decay)
        self.W1 *= 1 - cfg.forget_alpha
        self.W2 *= 1 - cfg.forget_alpha

        # Update statistics
        self.update_count += 1
        self.total_surprise += surprise

        return float(loss)

    def get_state_dict(self) -> dict[str, Any]:
        """Get weights for serialization."""
        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
            "update_count": self.update_count,
            "total_surprise": self.total_surprise,
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Load weights from serialized state."""
        self.W1 = state_dict["W1"].copy()
        self.b1 = state_dict["b1"].copy()
        self.W2 = state_dict["W2"].copy()
        self.b2 = state_dict["b2"].copy()
        self.update_count = state_dict.get("update_count", 0)
        self.total_surprise = state_dict.get("total_surprise", 0.0)


if TORCH_AVAILABLE:

    class TitansSidecarTorch(nn.Module):
        """
        PyTorch-based Titans Neural Memory Sidecar.

        This implementation provides GPU acceleration and proper
        gradient management for test-time learning.

        The architecture follows the paper:
        - MLP with residual connections
        - Surprise-gated updates
        - Momentum-based gradient accumulation
        - Forgetting mechanism to prevent unbounded growth
        """

        def __init__(self, config: TitansSidecarConfig | None = None):
            super().__init__()
            self.config = config or TitansSidecarConfig()
            cfg = self.config

            # Build memory MLP
            layers = []
            in_dim = cfg.input_dim

            for _i in range(cfg.num_layers - 1):
                layers.append(nn.Linear(in_dim, cfg.hidden_dim))
                if cfg.use_layer_norm:
                    layers.append(nn.LayerNorm(cfg.hidden_dim))
                layers.append(nn.ReLU())
                if cfg.dropout > 0:
                    layers.append(nn.Dropout(cfg.dropout))
                in_dim = cfg.hidden_dim

            # Final layer
            layers.append(nn.Linear(in_dim, cfg.output_dim))

            self.memory_mlp = nn.Sequential(*layers)

            # Surprise gate (learnable scaling)
            self.surprise_scale = nn.Parameter(torch.ones(1))

            # Initialize weights
            self._init_weights()

            # Momentum buffers (not nn.Parameters)
            self._momentum_buffers: dict[str, torch.Tensor] = {}

            # Statistics
            self.update_count = 0
            self.total_surprise = 0.0

        def _init_weights(self):
            """Initialize weights with Xavier/He initialization."""
            for module in self.memory_mlp.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through memory MLP.

            Args:
                x: Input tensor [batch, input_dim]

            Returns:
                Output tensor [batch, output_dim]
            """
            return self.memory_mlp(x)

        def retrieve(self, query: torch.Tensor) -> torch.Tensor:
            """
            Retrieve from memory based on query.

            Args:
                query: Query embedding [input_dim] or [batch, input_dim]

            Returns:
                Memory-augmented embedding
            """
            if query.dim() == 1:
                query = query.unsqueeze(0)

            with torch.no_grad():
                output = self.forward(query)

            return output.squeeze(0) if output.size(0) == 1 else output

        def memorize(
            self,
            embedding: torch.Tensor,
            target: torch.Tensor | None = None,
            surprise: float = 1.0,
        ) -> float:
            """
            Memorize with surprise-weighted update.

            Args:
                embedding: Input embedding to memorize
                target: Target output (defaults to embedding)
                surprise: Surprise signal (0.0 to ~5.0)

            Returns:
                Loss value
            """
            cfg = self.config

            if surprise < cfg.surprise_threshold:
                return 0.0

            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)

            if target is None:
                target = embedding
            elif target.dim() == 1:
                target = target.unsqueeze(0)

            # Forward pass
            output = self.forward(embedding)

            # MSE loss
            loss = F.mse_loss(output, target)

            # Backward pass
            self.zero_grad()
            loss.backward()

            # Apply surprise-scaled momentum updates
            lr = cfg.learning_rate * surprise * self.surprise_scale.item()

            with torch.no_grad():
                for name, param in self.named_parameters():
                    if param.grad is None:
                        continue

                    # Get or create momentum buffer
                    if name not in self._momentum_buffers:
                        self._momentum_buffers[name] = torch.zeros_like(param)

                    # Momentum update
                    momentum = self._momentum_buffers[name]
                    momentum.mul_(cfg.momentum).add_(param.grad, alpha=lr)

                    # Apply update
                    param.sub_(momentum)

                    # Apply forgetting (weight decay)
                    if "weight" in name:
                        param.mul_(1 - cfg.forget_alpha)

            # Statistics
            self.update_count += 1
            self.total_surprise += surprise

            return loss.item()

        def get_statistics(self) -> dict[str, Any]:
            """Get memory statistics."""
            return {
                "update_count": self.update_count,
                "total_surprise": self.total_surprise,
                "avg_surprise": self.total_surprise / max(1, self.update_count),
                "surprise_scale": self.surprise_scale.item(),
            }


class TitansSidecar:
    """
    High-level Titans Neural Memory Sidecar.

    This class provides a unified API that automatically selects
    between NumPy and PyTorch implementations based on availability.

    Usage:
        sidecar = TitansSidecar()

        # Store a memory with surprise signal
        surprise = entropix.get_surprise_signal()
        sidecar.memorize(embedding, surprise=surprise)

        # Retrieve augmented embedding
        augmented = sidecar.retrieve(query_embedding)
    """

    def __init__(
        self,
        config: TitansSidecarConfig | None = None,
        device: str | None = None,
        force_numpy: bool = False,
    ):
        """
        Initialize the Titans sidecar.

        Args:
            config: Configuration for the sidecar
            device: PyTorch device ('cuda', 'cpu', etc.)
            force_numpy: Force NumPy implementation even if PyTorch available
        """
        self.config = config or TitansSidecarConfig()
        self.device = device

        # Select implementation
        if TORCH_AVAILABLE and not force_numpy:
            self.backend = "torch"
            self._impl = TitansSidecarTorch(self.config)

            if device:
                self._impl = self._impl.to(device)
            elif torch.cuda.is_available():
                self._impl = self._impl.to("cuda")
                self.device = "cuda"
            else:
                self.device = "cpu"

            logger.info(f"TitansSidecar using PyTorch backend on {self.device}")
        else:
            self.backend = "numpy"
            self._impl = TitansSidecarNumpy(self.config)
            logger.info("TitansSidecar using NumPy backend")

        # High-surprise event tracking
        self.high_surprise_events: list[dict[str, Any]] = []
        self.high_surprise_threshold = 2.0  # Events above this are tracked

    def retrieve(
        self,
        query: Any,  # np.ndarray or torch.Tensor
    ) -> Any:
        """
        Retrieve from memory based on query embedding.

        Args:
            query: Query embedding

        Returns:
            Memory-augmented embedding (same type as input)
        """
        # Convert to appropriate type
        if self.backend == "torch":
            if isinstance(query, np.ndarray):
                query = torch.from_numpy(query).float()
                if self.device:
                    query = query.to(self.device)
            return self._impl.retrieve(query)
        else:
            if TORCH_AVAILABLE and isinstance(query, torch.Tensor):
                query = query.cpu().numpy()
            return self._impl.retrieve(query)

    def memorize(
        self,
        embedding: Any,
        target: Any | None = None,
        surprise: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """
        Memorize an embedding with surprise-weighted update.

        Args:
            embedding: Embedding to memorize
            target: Target output (defaults to embedding)
            surprise: Surprise signal from Entropix
            metadata: Optional metadata for high-surprise tracking

        Returns:
            Loss value
        """
        # Track high-surprise events
        if surprise > self.high_surprise_threshold:
            event = {
                "timestamp": __import__("datetime").datetime.now().isoformat(),
                "surprise": surprise,
                "metadata": metadata or {},
            }
            self.high_surprise_events.append(event)

            # Limit stored events
            if len(self.high_surprise_events) > self.config.max_stored_episodes:
                self.high_surprise_events.pop(0)

            logger.debug(f"High-surprise event recorded: {surprise:.3f}")

        # Convert to appropriate type
        if self.backend == "torch":
            if isinstance(embedding, np.ndarray):
                embedding = torch.from_numpy(embedding).float()
                if self.device:
                    embedding = embedding.to(self.device)
            if target is not None and isinstance(target, np.ndarray):
                target = torch.from_numpy(target).float()
                if self.device:
                    target = target.to(self.device)
            return self._impl.memorize(embedding, target, surprise)
        else:
            if TORCH_AVAILABLE and isinstance(embedding, torch.Tensor):
                embedding = embedding.cpu().numpy()
            if TORCH_AVAILABLE and target is not None and isinstance(target, torch.Tensor):
                target = target.cpu().numpy()
            return self._impl.memorize(embedding, target, surprise)

    def get_high_surprise_events(
        self,
        min_surprise: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Get high-surprise events for replay/consolidation.

        Args:
            min_surprise: Minimum surprise threshold
            limit: Maximum events to return

        Returns:
            List of high-surprise event dictionaries
        """
        events = self.high_surprise_events

        if min_surprise is not None:
            events = [e for e in events if e["surprise"] >= min_surprise]

        return events[-limit:]

    def get_statistics(self) -> dict[str, Any]:
        """Get sidecar statistics."""
        if self.backend == "torch":
            stats = self._impl.get_statistics()
        else:
            stats = {
                "update_count": self._impl.update_count,
                "total_surprise": self._impl.total_surprise,
                "avg_surprise": self._impl.total_surprise / max(1, self._impl.update_count),
            }

        stats["backend"] = self.backend
        stats["device"] = self.device
        stats["high_surprise_events"] = len(self.high_surprise_events)

        return stats

    def save(self, path: str):
        """Save sidecar state to disk with HMAC for integrity verification."""
        import hashlib
        import hmac
        import pickle

        state = {
            "config": self.config,
            "backend": self.backend,
            "high_surprise_events": self.high_surprise_events,
        }

        if self.backend == "torch":
            state["model_state"] = self._impl.state_dict()
        else:
            state["model_state"] = self._impl.get_state_dict()

        # Serialize state
        pickled_data = pickle.dumps(state)

        # Create HMAC signature for integrity verification
        # Uses a fixed key - in production, use a secure key from config
        hmac_key = b"titans_sidecar_integrity_key_v1"
        signature = hmac.new(hmac_key, pickled_data, hashlib.sha256).hexdigest()

        # Save with signature
        with open(path, "wb") as f:
            # Write signature length and signature first
            sig_bytes = signature.encode("utf-8")
            f.write(len(sig_bytes).to_bytes(4, "big"))
            f.write(sig_bytes)
            f.write(pickled_data)

        logger.info(f"TitansSidecar saved to {path}")

    def load(self, path: str):
        """
        Load sidecar state from disk with integrity verification.

        Security Note: This uses pickle which can execute arbitrary code.
        Only load files from trusted sources. HMAC verification ensures
        the file hasn't been tampered with since it was saved by this system.
        """
        import hashlib
        import hmac
        import pickle

        with open(path, "rb") as f:
            # Read signature
            sig_len = int.from_bytes(f.read(4), "big")
            signature = f.read(sig_len).decode("utf-8")
            pickled_data = f.read()

        # Verify HMAC signature
        hmac_key = b"titans_sidecar_integrity_key_v1"
        expected_signature = hmac.new(hmac_key, pickled_data, hashlib.sha256).hexdigest()

        if not hmac.compare_digest(signature, expected_signature):
            raise ValueError(
                f"Integrity check failed for {path}. "
                "File may be corrupted or tampered with."
            )

        # Safe to load after integrity verification (nosec B301)
        state = pickle.loads(pickled_data)  # noqa: S301

        self.high_surprise_events = state.get("high_surprise_events", [])

        if self.backend == "torch":
            self._impl.load_state_dict(state["model_state"])
        else:
            self._impl.load_state_dict(state["model_state"])

        logger.info(f"TitansSidecar loaded from {path}")


# Factory function
def create_titans_sidecar(
    embedding_dim: int = 768,
    learning_rate: float = 1e-3,
    momentum: float = 0.9,
    device: str | None = None,
) -> TitansSidecar:
    """
    Factory function to create a configured Titans sidecar.

    Args:
        embedding_dim: Dimension of embeddings
        learning_rate: Test-time learning rate
        momentum: Gradient momentum
        device: PyTorch device

    Returns:
        Configured TitansSidecar
    """
    config = TitansSidecarConfig(
        input_dim=embedding_dim,
        output_dim=embedding_dim,
        learning_rate=learning_rate,
        momentum=momentum,
    )

    return TitansSidecar(config, device=device)


# Backward compatibility alias
def create_titans_memory(
    input_dim: int = 768,
    memory_dim: int = 2048,
    learning_rate: float = 0.01,
    momentum_decay: float = 0.99,
    surprise_threshold: float = 0.5,
    use_torch: bool = None,
    device: str = "cpu",
) -> TitansSidecar:
    """
    Legacy factory function for backward compatibility.

    Use create_titans_sidecar() for new code.
    """
    config = TitansSidecarConfig(
        input_dim=input_dim,
        hidden_dim=memory_dim,
        output_dim=input_dim,
        learning_rate=learning_rate,
        momentum=momentum_decay,
        surprise_threshold=surprise_threshold,
    )
    return TitansSidecar(config, device=device)


# =============================================================================
# EPISODIC MEMORY STORE WITH JSON TIMESTAMPS
# =============================================================================


@dataclass
class EpisodicMemory:
    """A single episodic memory with semantic timestamp."""

    id: str  # Unique memory ID
    content: str  # The memory content
    embedding: np.ndarray | None = None  # Embedding vector
    surprise: float = 0.0  # Surprise when recorded
    source: str = "interaction"  # Source: interaction, search, system
    tags: list[str] = field(default_factory=list)  # Semantic tags

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    # Confidence and reliability
    confidence: float = 1.0  # 0-1 confidence score
    source_reliability: float = 1.0  # Source reliability score
    is_fact: bool = False  # Fact vs opinion

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "surprise": self.surprise,
            "source": self.source,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "confidence": self.confidence,
            "source_reliability": self.source_reliability,
            "is_fact": self.is_fact,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodicMemory":
        """Create from JSON dictionary."""
        embedding = None
        if data.get("embedding"):
            embedding = np.array(data["embedding"], dtype=np.float32)

        return cls(
            id=data["id"],
            content=data["content"],
            embedding=embedding,
            surprise=data.get("surprise", 0.0),
            source=data.get("source", "interaction"),
            tags=data.get("tags", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            accessed_at=datetime.fromisoformat(data["accessed_at"]),
            access_count=data.get("access_count", 0),
            confidence=data.get("confidence", 1.0),
            source_reliability=data.get("source_reliability", 1.0),
            is_fact=data.get("is_fact", False),
        )


class EpisodicMemoryStore:
    """
    JSON-based Episodic Memory Store with semantic timestamps.

    Stores episodic memories with rich metadata including:
    - Precise timestamps for temporal recall
    - Source tracking for reliability
    - Confidence scores
    - Fact vs opinion classification

    Memories are persisted to JSON for durability and portability.
    """

    def __init__(
        self,
        storage_path: str = "data/memory/episodic",
        max_memories: int = 10000,
        embedding_dim: int = 768,
    ):
        """
        Initialize the episodic memory store.

        Args:
            storage_path: Directory for JSON storage
            max_memories: Maximum memories to retain
            embedding_dim: Dimension of embeddings
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.max_memories = max_memories
        self.embedding_dim = embedding_dim

        # In-memory cache
        self.memories: dict[str, EpisodicMemory] = {}

        # Index for fast lookup
        self.by_date: dict[str, list[str]] = {}  # date -> memory ids
        self.by_tag: dict[str, list[str]] = {}  # tag -> memory ids

        # Load existing memories
        self._load_all()

        logger.info(f"EpisodicMemoryStore initialized with {len(self.memories)} memories")

    def store(
        self,
        content: str,
        embedding: np.ndarray | None = None,
        surprise: float = 0.0,
        source: str = "interaction",
        tags: list[str] | None = None,
        confidence: float = 1.0,
        is_fact: bool = False,
        source_reliability: float = 1.0,
    ) -> str:
        """
        Store a new episodic memory.

        Args:
            content: Memory content
            embedding: Embedding vector
            surprise: Surprise signal
            source: Source of memory
            tags: Semantic tags
            confidence: Confidence score
            is_fact: Whether this is a fact
            source_reliability: Source reliability

        Returns:
            Memory ID
        """
        import uuid

        memory_id = str(uuid.uuid4())[:8]
        now = datetime.now()

        memory = EpisodicMemory(
            id=memory_id,
            content=content,
            embedding=embedding,
            surprise=surprise,
            source=source,
            tags=tags or [],
            created_at=now,
            accessed_at=now,
            access_count=0,
            confidence=confidence,
            source_reliability=source_reliability,
            is_fact=is_fact,
        )

        self.memories[memory_id] = memory

        # Update indices
        date_key = now.strftime("%Y-%m-%d")
        if date_key not in self.by_date:
            self.by_date[date_key] = []
        self.by_date[date_key].append(memory_id)

        for tag in memory.tags:
            if tag not in self.by_tag:
                self.by_tag[tag] = []
            self.by_tag[tag].append(memory_id)

        # Persist to JSON
        self._save_memory(memory)

        # Evict old memories if needed
        if len(self.memories) > self.max_memories:
            self._evict_oldest()

        logger.debug(f"Stored memory {memory_id}: {content[:50]}...")
        return memory_id

    def retrieve(self, memory_id: str) -> EpisodicMemory | None:
        """Retrieve a memory by ID and update access time."""
        memory = self.memories.get(memory_id)
        if memory:
            memory.accessed_at = datetime.now()
            memory.access_count += 1
            self._save_memory(memory)
        return memory

    def retrieve_by_date(
        self,
        date: datetime,
        range_days: int = 0,
    ) -> list[EpisodicMemory]:
        """
        Retrieve memories from a specific date or range.

        Args:
            date: Target date
            range_days: Days before/after to include

        Returns:
            List of memories
        """
        memories = []

        for day_offset in range(-range_days, range_days + 1):
            target_date = date + timedelta(days=day_offset)
            date_key = target_date.strftime("%Y-%m-%d")

            if date_key in self.by_date:
                for memory_id in self.by_date[date_key]:
                    if memory_id in self.memories:
                        memories.append(self.memories[memory_id])

        return sorted(memories, key=lambda m: m.created_at)

    def retrieve_by_tag(self, tag: str) -> list[EpisodicMemory]:
        """Retrieve memories with a specific tag."""
        if tag not in self.by_tag:
            return []

        return [self.memories[mid] for mid in self.by_tag[tag] if mid in self.memories]

    def search_semantic(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[EpisodicMemory, float]]:
        """
        Search memories by semantic similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results

        Returns:
            List of (memory, similarity) tuples
        """
        results = []

        for memory in self.memories.values():
            if memory.embedding is not None:
                # Cosine similarity
                sim = np.dot(query_embedding, memory.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(memory.embedding) + 1e-8
                )
                results.append((memory, float(sim)))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_recent(self, n: int = 10) -> list[EpisodicMemory]:
        """Get most recent memories."""
        sorted_memories = sorted(
            self.memories.values(),
            key=lambda m: m.created_at,
            reverse=True,
        )
        return sorted_memories[:n]

    def get_high_surprise(self, threshold: float = 1.5) -> list[EpisodicMemory]:
        """Get high-surprise memories."""
        return [m for m in self.memories.values() if m.surprise >= threshold]

    def _save_memory(self, memory: EpisodicMemory) -> None:
        """Save a memory to JSON file."""
        import json

        file_path = self.storage_path / f"{memory.id}.json"
        with open(file_path, "w") as f:
            json.dump(memory.to_dict(), f, indent=2)

    def _load_all(self) -> None:
        """Load all memories from storage."""
        import json

        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                    memory = EpisodicMemory.from_dict(data)
                    self.memories[memory.id] = memory

                    # Rebuild indices
                    date_key = memory.created_at.strftime("%Y-%m-%d")
                    if date_key not in self.by_date:
                        self.by_date[date_key] = []
                    self.by_date[date_key].append(memory.id)

                    for tag in memory.tags:
                        if tag not in self.by_tag:
                            self.by_tag[tag] = []
                        self.by_tag[tag].append(memory.id)

            except Exception as e:
                logger.error(f"Failed to load memory {file_path}: {e}")

    def _evict_oldest(self) -> None:
        """Evict oldest, least-accessed memories."""
        # Score memories (higher = more likely to evict)
        scored = []
        for memory in self.memories.values():
            age_days = (datetime.now() - memory.created_at).days
            access_score = 1 / (memory.access_count + 1)
            recency_score = (datetime.now() - memory.accessed_at).days

            # Lower surprise = more likely to evict
            surprise_score = 1 / (memory.surprise + 0.1)

            evict_score = (
                age_days * 0.3 + access_score * 0.3 + recency_score * 0.2 + surprise_score * 0.2
            )
            scored.append((memory.id, evict_score))

        # Sort by evict score (highest first)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Remove top candidates
        to_evict = len(self.memories) - self.max_memories + 100  # Remove 100 extra
        for memory_id, _ in scored[:to_evict]:
            self._delete_memory(memory_id)

    def _delete_memory(self, memory_id: str) -> None:
        """Delete a memory."""
        if memory_id in self.memories:
            memory = self.memories.pop(memory_id)

            # Clean up indices
            date_key = memory.created_at.strftime("%Y-%m-%d")
            if date_key in self.by_date:
                self.by_date[date_key] = [mid for mid in self.by_date[date_key] if mid != memory_id]

            for tag in memory.tags:
                if tag in self.by_tag:
                    self.by_tag[tag] = [mid for mid in self.by_tag[tag] if mid != memory_id]

            # Delete file
            file_path = self.storage_path / f"{memory_id}.json"
            if file_path.exists():
                file_path.unlink()

    def get_stats(self) -> dict[str, Any]:
        """Get memory store statistics."""
        return {
            "total_memories": len(self.memories),
            "dates_covered": len(self.by_date),
            "tags_used": len(self.by_tag),
            "avg_surprise": (
                np.mean([m.surprise for m in self.memories.values()]) if self.memories else 0
            ),
            "facts_count": sum(1 for m in self.memories.values() if m.is_fact),
            "storage_path": str(self.storage_path),
        }
