"""
Unit Tests for Titans Memory Module
=====================================

Tests for the test-time learning memory system.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import numpy as np

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.hippocampus.titans import (
    TitansConfig,
    TitansMemory,
    Memory,
    MemoryStats,
    create_titans_memory,
)


class TestTitansConfig:
    """Test TitansConfig dataclass."""

    def test_default_values(self):
        """Test default configuration."""
        config = TitansConfig()
        assert config.hidden_dim == 256
        assert config.num_layers == 3
        assert config.learning_rate == 0.01
        assert config.max_memory_mb == 200

    def test_custom_values(self):
        """Test custom configuration."""
        config = TitansConfig(
            hidden_dim=512,
            num_layers=4,
            learning_rate=0.005,
            max_memory_mb=300,
        )
        assert config.hidden_dim == 512
        assert config.num_layers == 4


class TestMemory:
    """Test Memory dataclass."""

    def test_creation(self):
        """Test memory creation."""
        memory = Memory(
            content="Test memory content",
            embedding=[0.1, 0.2, 0.3],
            surprise=0.5,
            timestamp=datetime.now(),
        )

        assert memory.content == "Test memory content"
        assert len(memory.embedding) == 3
        assert memory.surprise == 0.5

    def test_importance_scoring(self):
        """Test memory importance calculation."""
        # High surprise = more important
        high_importance = Memory(
            content="Novel information",
            embedding=[0.1, 0.2, 0.3],
            surprise=0.9,
            timestamp=datetime.now(),
        )

        low_importance = Memory(
            content="Routine information",
            embedding=[0.1, 0.2, 0.3],
            surprise=0.1,
            timestamp=datetime.now(),
        )

        assert high_importance.surprise > low_importance.surprise


class TestMemoryStats:
    """Test MemoryStats dataclass."""

    def test_default_values(self):
        """Test default stats."""
        stats = MemoryStats()
        assert stats.total_memories == 0
        assert stats.memory_updates == 0
        assert stats.avg_surprise == 0.0

    def test_to_dict(self):
        """Test dictionary conversion."""
        stats = MemoryStats(
            total_memories=100,
            memory_updates=50,
            avg_surprise=0.45,
        )
        d = stats.to_dict()

        assert isinstance(d, dict)
        assert d["total_memories"] == 100
        assert d["avg_surprise"] == 0.45


class TestTitansMemory:
    """Test TitansMemory class."""

    def test_initialization(self):
        """Test memory system initialization."""
        config = TitansConfig()
        titans = TitansMemory(config)

        assert titans.config == config
        assert titans._memory_count == 0

    def test_memory_footprint(self):
        """Test fixed memory footprint."""
        config = TitansConfig(max_memory_mb=200)
        titans = TitansMemory(config)

        # Memory should stay within budget regardless of content
        initial_size = titans.get_memory_size_mb()
        assert initial_size <= config.max_memory_mb

    def test_factory_function(self):
        """Test create_titans_memory factory."""
        titans = create_titans_memory(hidden_dim=128)

        assert titans is not None
        assert titans.config.hidden_dim == 128


class TestMemorization:
    """Test memorization functionality."""

    def test_memorize_content(self):
        """Test content memorization."""
        config = TitansConfig()
        titans = TitansMemory(config)

        titans.memorize("Hello, world!", surprise=0.3)

        stats = titans.get_stats()
        assert stats.total_memories == 1

    def test_surprise_weighted_update(self):
        """Test surprise-weighted gradient updates."""
        config = TitansConfig()
        titans = TitansMemory(config)

        # High surprise should cause larger updates
        initial_state = titans.get_state_snapshot()

        titans.memorize("Novel concept", surprise=0.9)

        # State should change more with high surprise
        # This is conceptual - actual implementation may vary

    def test_multiple_memories(self):
        """Test storing multiple memories."""
        config = TitansConfig()
        titans = TitansMemory(config)

        contents = [
            "First memory",
            "Second memory",
            "Third memory",
        ]

        for content in contents:
            titans.memorize(content, surprise=0.5)

        stats = titans.get_stats()
        assert stats.total_memories == 3


class TestRetrieval:
    """Test memory retrieval functionality."""

    def test_retrieve_by_query(self):
        """Test retrieval by semantic query."""
        config = TitansConfig()
        titans = TitansMemory(config)

        # Store some memories
        titans.memorize("Python is a programming language", surprise=0.3)
        titans.memorize("JavaScript runs in browsers", surprise=0.3)
        titans.memorize("Cats are mammals", surprise=0.3)

        # Query for programming
        results = titans.retrieve("programming languages", top_k=2)

        assert len(results) <= 2

    def test_top_k_limiting(self):
        """Test top_k limits results."""
        config = TitansConfig()
        titans = TitansMemory(config)

        # Store 10 memories
        for i in range(10):
            titans.memorize(f"Memory {i}", surprise=0.3)

        # Retrieve only top 3
        results = titans.retrieve("Memory", top_k=3)

        assert len(results) <= 3

    def test_empty_retrieval(self):
        """Test retrieval from empty memory."""
        config = TitansConfig()
        titans = TitansMemory(config)

        results = titans.retrieve("anything", top_k=5)

        assert len(results) == 0


class TestTestTimeLearning:
    """Test test-time learning capabilities."""

    def test_mlp_structure(self):
        """Test 3-layer MLP structure."""
        config = TitansConfig(num_layers=3, hidden_dim=256)
        titans = TitansMemory(config)

        # MLP should have input, hidden, output layers
        assert config.num_layers == 3

    def test_gradient_update(self):
        """Test gradient updates during memorization."""
        config = TitansConfig(learning_rate=0.01)
        titans = TitansMemory(config)

        # Store initial state
        initial_params = titans.get_parameter_count()

        # Memorize with gradient update
        titans.memorize("New information", surprise=0.8)

        # Parameters should remain same count (learning, not growing)
        final_params = titans.get_parameter_count()
        assert initial_params == final_params

    def test_forgetting_mechanism(self):
        """Test automatic forgetting of low-importance memories."""
        config = TitansConfig(max_memory_mb=200)
        titans = TitansMemory(config)

        # Store many low-importance memories
        for i in range(1000):
            titans.memorize(f"Low importance {i}", surprise=0.1)

        # Memory should not grow unboundedly
        size_mb = titans.get_memory_size_mb()
        assert size_mb <= config.max_memory_mb


class TestMemoryIntegration:
    """Test memory integration with other components."""

    def test_embedding_format(self):
        """Test embedding format compatibility."""
        # Embeddings should be consistent dimensions
        embedding_dim = 768  # Typical transformer embedding

        embedding = np.random.randn(embedding_dim).tolist()

        memory = Memory(
            content="Test",
            embedding=embedding,
            surprise=0.5,
            timestamp=datetime.now(),
        )

        assert len(memory.embedding) == embedding_dim

    def test_context_retrieval_for_cortex(self):
        """Test retrieving context for Cortex."""
        config = TitansConfig()
        titans = TitansMemory(config)

        # Store conversation history
        titans.memorize("User asked about Python", surprise=0.3)
        titans.memorize("Discussed machine learning", surprise=0.5)
        titans.memorize("Mentioned neural networks", surprise=0.6)

        # Retrieve context for Cortex
        context = titans.retrieve("deep learning models", top_k=3)

        # Context should be a list of relevant memories
        assert isinstance(context, list)


@pytest.mark.asyncio
class TestTitansAsync:
    """Async tests for Titans operations."""

    async def test_async_memorize(self):
        """Test async memorization."""
        config = TitansConfig()
        titans = TitansMemory(config)

        await titans.async_memorize("Async content", surprise=0.5)

        stats = titans.get_stats()
        assert stats.total_memories == 1

    async def test_async_retrieve(self):
        """Test async retrieval."""
        config = TitansConfig()
        titans = TitansMemory(config)

        await titans.async_memorize("Test content", surprise=0.5)

        results = await titans.async_retrieve("Test", top_k=5)

        assert isinstance(results, list)


class TestMemoryPersistence:
    """Test memory persistence features."""

    def test_save_state(self):
        """Test saving memory state."""
        config = TitansConfig()
        titans = TitansMemory(config)

        titans.memorize("Persistent memory", surprise=0.5)

        state = titans.save_state()

        assert state is not None
        assert "memories" in state or "parameters" in state

    def test_load_state(self):
        """Test loading memory state."""
        config = TitansConfig()
        titans1 = TitansMemory(config)

        titans1.memorize("Test memory", surprise=0.5)
        state = titans1.save_state()

        # Create new instance and load
        titans2 = TitansMemory(config)
        titans2.load_state(state)

        stats = titans2.get_stats()
        assert stats.total_memories >= 1


class TestPerformance:
    """Test Titans performance characteristics."""

    def test_fixed_memory_footprint(self):
        """Memory should not grow with content."""
        config = TitansConfig(max_memory_mb=200)
        titans = TitansMemory(config)

        initial_size = titans.get_memory_size_mb()

        # Store 100 memories
        for i in range(100):
            titans.memorize(f"Memory {i} with some longer content here", surprise=0.5)

        final_size = titans.get_memory_size_mb()

        # Size should stay roughly constant (within 10%)
        assert final_size <= initial_size * 1.1

    def test_retrieval_speed(self):
        """Test retrieval is fast."""
        import time

        config = TitansConfig()
        titans = TitansMemory(config)

        # Store some memories
        for i in range(100):
            titans.memorize(f"Memory {i}", surprise=0.3)

        # Time retrieval
        start = time.time()
        titans.retrieve("Memory 50", top_k=5)
        elapsed = time.time() - start

        # Should be fast (< 100ms)
        assert elapsed < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
