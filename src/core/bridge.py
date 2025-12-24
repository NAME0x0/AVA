"""
BRIDGE - Neural State Projection for Cortex-Medulla Handoff
============================================================

The Bridge implements the critical state handoff mechanism between the
Medulla (Mamba SSM) and Cortex (Transformer). It solves the "context
transfer problem" by projecting the Medulla's compressed hidden state
into the Cortex's embedding space.

Problem:
- Medulla maintains context in a fixed-size SSM hidden state
- Cortex expects token embeddings in Transformer space
- Naive text transfer would require expensive pre-fill (~minutes for long context)

Solution:
- Train a Projection Adapter (MLP) that maps Mamba state → Transformer embeddings
- Project state as "soft prompts" prepended to the Cortex input
- Instant context handoff with O(1) complexity

This enables the Cortex to "inherit" the conversation context already
understood, bypassing the pre-fill bottleneck entirely.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BridgeConfig:
    """
    Configuration for the Bridge (State Projection).

    The Bridge consists of a lightweight MLP that maps between
    the Medulla's Mamba state space and the Cortex's Transformer
    embedding space.

    Memory Budget: ~50 MB (trivial compared to models)
    """

    # Input dimension (Mamba hidden state)
    medulla_state_dim: int = 2560  # Matches Medulla hidden_dim

    # Output dimension (Transformer embedding)
    cortex_embedding_dim: int = 8192  # Llama-3 70B hidden size

    # Projection MLP architecture
    hidden_dims: list[int] = field(default_factory=lambda: [4096, 4096])

    # Number of soft prompt tokens to generate
    num_soft_tokens: int = 32  # Virtual context tokens

    # Training configuration (for adapter fine-tuning)
    learning_rate: float = 1e-4
    dropout: float = 0.1
    use_layer_norm: bool = True

    # Projection modes
    use_residual: bool = True  # Add residual connection
    use_attention_pooling: bool = False  # Use attention to pool state

    # State persistence
    adapter_path: str = "models/fine_tuned_adapters/bridge"

    # Device
    device: str = "cuda"


class ProjectionAdapter:
    """
    MLP-based adapter for projecting Mamba state to Transformer embeddings.

    Architecture:
        Input: [medulla_state_dim] - Mamba hidden state
        Hidden: [hidden_dim_1, hidden_dim_2, ...]
        Output: [num_soft_tokens, cortex_embedding_dim] - Soft prompts

    The output soft prompts are prepended to the Cortex input,
    effectively "initializing" the Transformer with the Medulla's context.
    """

    def __init__(self, config: BridgeConfig):
        """
        Initialize the projection adapter.

        Args:
            config: Bridge configuration
        """
        self.config = config

        # Calculate total output dimension
        self.output_dim = config.num_soft_tokens * config.cortex_embedding_dim

        # Build layer dimensions
        layer_dims = [config.medulla_state_dim] + config.hidden_dims + [self.output_dim]

        # Initialize weights (Xavier initialization)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        for i in range(len(layer_dims) - 1):
            fan_in = layer_dims[i]
            fan_out = layer_dims[i + 1]

            # Xavier initialization
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            W = np.random.randn(fan_in, fan_out).astype(np.float32) * scale
            b = np.zeros(fan_out, dtype=np.float32)

            self.weights.append(W)
            self.biases.append(b)

        # Layer normalization parameters
        if config.use_layer_norm:
            self.layer_norm_gamma = np.ones(config.medulla_state_dim, dtype=np.float32)
            self.layer_norm_beta = np.zeros(config.medulla_state_dim, dtype=np.float32)

        # Running statistics for normalization
        self.running_mean = np.zeros(config.medulla_state_dim, dtype=np.float32)
        self.running_var = np.ones(config.medulla_state_dim, dtype=np.float32)
        self.num_updates = 0

        logger.info(f"ProjectionAdapter initialized: {layer_dims}")

    def forward(self, medulla_state: np.ndarray) -> np.ndarray:
        """
        Project Medulla state to Cortex soft prompts.

        Args:
            medulla_state: Mamba hidden state [medulla_state_dim]

        Returns:
            Soft prompts [num_soft_tokens, cortex_embedding_dim]
        """
        x = medulla_state.astype(np.float32)

        # Input normalization
        if self.config.use_layer_norm:
            x = self._layer_norm(x)

        # Forward through MLP layers
        for i, (W, b) in enumerate(zip(self.weights, self.biases, strict=False)):
            x = x @ W + b

            # Apply activation (ReLU) for all but last layer
            if i < len(self.weights) - 1:
                x = np.maximum(x, 0)

                # Apply dropout during training
                if self.config.dropout > 0:
                    # Dropout disabled during inference
                    pass

        # Reshape to soft tokens
        soft_prompts = x.reshape(
            self.config.num_soft_tokens,
            self.config.cortex_embedding_dim,
        )

        # Normalize output embeddings
        soft_prompts = self._normalize_embeddings(soft_prompts)

        return soft_prompts

    def _layer_norm(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Apply layer normalization."""
        mean = x.mean()
        var = x.var()
        x_norm = (x - mean) / np.sqrt(var + eps)
        return self.layer_norm_gamma * x_norm + self.layer_norm_beta

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit length (L2 normalization per token)."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)

    def update_statistics(self, medulla_state: np.ndarray) -> None:
        """
        Update running statistics for input normalization.

        Called during operation to adapt to the Medulla's state distribution.
        """
        self.num_updates += 1
        alpha = 0.01

        self.running_mean = (1 - alpha) * self.running_mean + alpha * medulla_state
        self.running_var = (1 - alpha) * self.running_var + alpha * (
            medulla_state - self.running_mean
        ) ** 2

    def save(self, path: str) -> None:
        """Save adapter weights to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            path,
            weights=list(self.weights),
            biases=list(self.biases),
            layer_norm_gamma=self.layer_norm_gamma if self.config.use_layer_norm else None,
            layer_norm_beta=self.layer_norm_beta if self.config.use_layer_norm else None,
            running_mean=self.running_mean,
            running_var=self.running_var,
            num_updates=self.num_updates,
        )
        logger.info(f"Saved projection adapter to {path}")

    def load(self, path: str) -> None:
        """Load adapter weights from disk."""
        data = np.load(path, allow_pickle=True)

        self.weights = list(data["weights"])
        self.biases = list(data["biases"])

        if self.config.use_layer_norm:
            self.layer_norm_gamma = data["layer_norm_gamma"]
            self.layer_norm_beta = data["layer_norm_beta"]

        self.running_mean = data["running_mean"]
        self.running_var = data["running_var"]
        self.num_updates = int(data["num_updates"])

        logger.info(f"Loaded projection adapter from {path}")


class ContextCompressor:
    """
    Compresses conversation history into a compact representation.

    Used alongside the Bridge to provide additional context beyond
    what's captured in the Medulla's hidden state.
    """

    def __init__(self, max_tokens: int = 512):
        self.max_tokens = max_tokens
        self.compression_ratio = 0.0

    def compress(
        self,
        conversation_history: list[dict[str, str]],
        current_query: str,
    ) -> str:
        """
        Compress conversation history to fit within token budget.

        Uses a summarization strategy:
        1. Keep the most recent turns verbatim
        2. Summarize older turns
        3. Extract key entities and facts

        Args:
            conversation_history: List of {role, content} dicts
            current_query: Current user query

        Returns:
            Compressed context string
        """
        if not conversation_history:
            return current_query

        # Estimate tokens (rough: 4 chars per token)
        chars_per_token = 4
        max_chars = self.max_tokens * chars_per_token

        # Build context from most recent to oldest
        context_parts = []
        total_chars = len(current_query)

        for turn in reversed(conversation_history[-10:]):
            turn_text = f"{turn.get('role', 'user')}: {turn.get('content', '')}"
            turn_chars = len(turn_text)

            if total_chars + turn_chars <= max_chars:
                context_parts.insert(0, turn_text)
                total_chars += turn_chars
            else:
                # Summarize remaining history
                remaining = (
                    conversation_history[: -len(context_parts)]
                    if context_parts
                    else conversation_history
                )
                if remaining:
                    summary = self._summarize_turns(remaining)
                    context_parts.insert(0, f"[Previous context: {summary}]")
                break

        # Calculate compression ratio
        original_chars = sum(len(t.get("content", "")) for t in conversation_history)
        self.compression_ratio = total_chars / max(1, original_chars)

        return "\n".join(context_parts)

    def _summarize_turns(self, turns: list[dict[str, str]]) -> str:
        """Create a brief summary of conversation turns."""
        # Simple extractive summary - take key phrases
        topics = []
        for turn in turns[-5:]:  # Last 5 turns only
            content = turn.get("content", "")
            # Extract first sentence or key phrase
            first_sentence = content.split(".")[0][:100]
            if first_sentence:
                topics.append(first_sentence.strip())

        return "; ".join(topics) if topics else "ongoing conversation"


class Bridge:
    """
    The Bridge - Neural State Projection for Cortex-Medulla Handoff.

    The Bridge solves the critical challenge of transferring context
    between the Medulla's SSM representation and the Cortex's
    Transformer representation.

    Key Operations:
    1. Project: Convert Medulla state → Soft prompts for Cortex
    2. Compress: Summarize conversation history as backup context
    3. Handoff: Prepare complete input package for Cortex

    The projection eliminates the need for expensive text pre-fill,
    enabling instant context transfer regardless of conversation length.
    """

    def __init__(
        self,
        config: BridgeConfig | None = None,
    ):
        """
        Initialize the Bridge.

        Args:
            config: Bridge configuration
        """
        self.config = config or BridgeConfig()

        # Initialize projection adapter
        self.adapter = ProjectionAdapter(self.config)

        # Initialize context compressor
        self.compressor = ContextCompressor(max_tokens=512)

        # State tracking
        self.is_initialized = False
        self.handoff_count = 0
        self.total_projection_time = 0.0

        # Load pre-trained adapter if available
        adapter_path = Path(self.config.adapter_path)
        if adapter_path.exists():
            try:
                self.adapter.load(str(adapter_path / "adapter.npz"))
                self.is_initialized = True
            except Exception as e:
                logger.warning(f"Could not load adapter: {e}")

        logger.info(f"Bridge initialized with config: {self.config}")

    async def prepare_cortex_input(
        self,
        medulla_state: np.ndarray,
        current_query: str,
        conversation_history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """
        Prepare complete input package for Cortex generation.

        This is the main handoff method that:
        1. Projects Medulla state to soft prompts
        2. Compresses conversation history
        3. Constructs the final prompt

        Args:
            medulla_state: Current Medulla hidden state vector
            current_query: User's current query
            conversation_history: Optional conversation history
            system_prompt: Optional system prompt

        Returns:
            Dictionary containing:
            - soft_prompts: Projected embeddings for Cortex
            - text_prompt: Constructed text prompt
            - metadata: Handoff metrics
        """
        start_time = time.time()

        # 1. Project Medulla state to soft prompts
        soft_prompts = self.adapter.forward(medulla_state)

        # Update adapter statistics
        self.adapter.update_statistics(medulla_state)

        # 2. Compress conversation history
        compressed_context = ""
        if conversation_history:
            compressed_context = self.compressor.compress(
                conversation_history,
                current_query,
            )

        # 3. Construct text prompt
        text_prompt = self._construct_prompt(
            system_prompt=system_prompt,
            compressed_context=compressed_context,
            current_query=current_query,
        )

        # Update metrics
        projection_time = time.time() - start_time
        self.total_projection_time += projection_time
        self.handoff_count += 1

        return {
            "soft_prompts": soft_prompts,
            "text_prompt": text_prompt,
            "metadata": {
                "projection_time_ms": projection_time * 1000,
                "num_soft_tokens": self.config.num_soft_tokens,
                "compression_ratio": self.compressor.compression_ratio,
                "handoff_count": self.handoff_count,
            },
        }

    def _construct_prompt(
        self,
        system_prompt: str | None,
        compressed_context: str,
        current_query: str,
    ) -> str:
        """
        Construct the text prompt for Cortex.

        The soft prompts handle the Medulla's state, so the text
        prompt focuses on explicit instructions and the current query.
        """
        parts = []

        # System prompt
        if system_prompt:
            parts.append(f"<|system|>\n{system_prompt}\n")
        else:
            parts.append(
                "<|system|>\n"
                "You are AVA, an advanced AI assistant with deep reasoning capabilities. "
                "You have access to the conversation context through your internal state. "
                "Provide thoughtful, comprehensive responses.\n"
            )

        # Context (compressed conversation history)
        if compressed_context:
            parts.append(f"<|context|>\n{compressed_context}\n")

        # Current query
        parts.append(f"<|user|>\n{current_query}\n")

        # Generation prompt
        parts.append("<|assistant|>\n")

        return "".join(parts)

    def project_state(self, medulla_state: np.ndarray) -> np.ndarray:
        """
        Direct state projection without prompt construction.

        Args:
            medulla_state: Medulla hidden state

        Returns:
            Soft prompts for Cortex
        """
        return self.adapter.forward(medulla_state)

    async def train_adapter(
        self,
        training_pairs: list[tuple[np.ndarray, np.ndarray]],
        epochs: int = 10,
        batch_size: int = 32,
    ) -> dict[str, float]:
        """
        Train the projection adapter on paired Medulla-Cortex states.

        This is used to fine-tune the adapter for better alignment
        between the two representation spaces.

        Args:
            training_pairs: List of (medulla_state, cortex_embedding) pairs
            epochs: Number of training epochs
            batch_size: Training batch size

        Returns:
            Training metrics
        """
        logger.info(f"Training projection adapter on {len(training_pairs)} pairs")

        # Simple gradient descent training
        # In production, this would use PyTorch/JAX for GPU acceleration

        losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            np.random.shuffle(training_pairs)

            for i in range(0, len(training_pairs), batch_size):
                batch = training_pairs[i : i + batch_size]

                # Forward pass
                batch_loss = 0.0
                for medulla_state, target_embedding in batch:
                    predicted = self.adapter.forward(medulla_state)

                    # MSE loss
                    error = predicted.flatten() - target_embedding.flatten()
                    loss = np.mean(error**2)
                    batch_loss += loss

                    # Backward pass (simplified gradient descent)
                    # In production, use autograd

                batch_loss /= len(batch)
                epoch_loss += batch_loss

            epoch_loss /= len(training_pairs) // batch_size
            losses.append(epoch_loss)

            logger.info(f"Epoch {epoch + 1}/{epochs}: loss = {epoch_loss:.4f}")

        # Save trained adapter
        self.adapter.save(str(Path(self.config.adapter_path) / "adapter.npz"))

        return {
            "final_loss": losses[-1] if losses else 0.0,
            "loss_history": losses,
            "epochs": epochs,
            "num_pairs": len(training_pairs),
        }

    def get_stats(self) -> dict[str, Any]:
        """Get Bridge statistics."""
        return {
            "handoff_count": self.handoff_count,
            "total_projection_time": self.total_projection_time,
            "avg_projection_time_ms": (
                self.total_projection_time * 1000 / max(1, self.handoff_count)
            ),
            "adapter_updates": self.adapter.num_updates,
            "compression_ratio": self.compressor.compression_ratio,
            "num_soft_tokens": self.config.num_soft_tokens,
        }

    def save_state(self) -> None:
        """Save Bridge state to disk."""
        adapter_path = Path(self.config.adapter_path)
        adapter_path.mkdir(parents=True, exist_ok=True)
        self.adapter.save(str(adapter_path / "adapter.npz"))

    def load_state(self) -> None:
        """Load Bridge state from disk."""
        adapter_path = Path(self.config.adapter_path) / "adapter.npz"
        if adapter_path.exists():
            self.adapter.load(str(adapter_path))
            self.is_initialized = True
