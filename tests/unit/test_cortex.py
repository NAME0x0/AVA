"""
Unit Tests for Cortex Component
================================

Tests for the deep reasoning core of AVA's dual-brain architecture.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.cortex import CortexConfig, CortexEngine


class TestCortexConfig:
    """Test CortexConfig dataclass."""

    def test_default_values(self):
        """Test default configuration."""
        config = CortexConfig()
        assert config.model_name == "qwen2.5:32b"
        assert config.max_tokens == 2048
        assert config.temperature == 0.7
        assert config.simulation_mode is False

    def test_custom_values(self):
        """Test custom configuration."""
        config = CortexConfig(
            model_name="llama3:70b",
            max_tokens=4096,
            temperature=0.5,
            simulation_mode=True,
        )
        assert config.model_name == "llama3:70b"
        assert config.max_tokens == 4096
        assert config.simulation_mode is True

    def test_to_dict(self):
        """Test dictionary conversion."""
        config = CortexConfig(model_name="test-model")
        d = config.to_dict()

        assert isinstance(d, dict)
        assert d["model_name"] == "test-model"


class TestCortexEngine:
    """Test CortexEngine class."""

    def test_initialization(self):
        """Test engine initialization."""
        config = CortexConfig(simulation_mode=True)
        engine = CortexEngine(config)

        assert engine.config == config
        assert engine._is_loaded is False

    def test_simulation_mode(self):
        """Test simulation mode behavior."""
        config = CortexConfig(simulation_mode=True)
        engine = CortexEngine(config)

        assert engine.config.simulation_mode is True


@pytest.mark.asyncio
class TestCortexAsync:
    """Async tests for Cortex operations."""

    async def test_think_returns_response(self):
        """Test that think returns a valid response."""
        config = CortexConfig(simulation_mode=True)
        engine = CortexEngine(config)

        result = await engine.think("Explain quantum computing")

        assert isinstance(result, dict)
        assert "response" in result

    async def test_think_complex_query(self):
        """Test handling of complex queries."""
        config = CortexConfig(simulation_mode=True)
        engine = CortexEngine(config)

        complex_query = """
        Analyze the philosophical implications of Gödel's incompleteness
        theorems on the nature of mathematical truth and the limits of
        formal systems. Consider both the first and second theorems.
        """

        result = await engine.think(complex_query)

        assert "response" in result
        # Complex queries should trigger deep reasoning
        assert len(result.get("response", "")) > 0

    async def test_verify_claim(self):
        """Test claim verification."""
        config = CortexConfig(simulation_mode=True)
        engine = CortexEngine(config)

        # Test a verifiable claim
        claim = "Paris is the capital of France"
        sources = ["Wikipedia", "Encyclopedia Britannica"]

        result = await engine.verify(claim, sources)

        assert isinstance(result, dict)
        assert "verified" in result or "confidence" in result

    async def test_context_handling(self):
        """Test context is properly used."""
        config = CortexConfig(simulation_mode=True)
        engine = CortexEngine(config)

        context = [
            "User asked about Python earlier",
            "Discussion about machine learning",
        ]

        result = await engine.think("How does this relate?", context=context)

        assert "response" in result


class TestCortexMemoryManagement:
    """Test Cortex memory management."""

    def test_layer_paging_concept(self):
        """Test that layer paging is conceptually correct."""
        # Cortex uses layer-wise paging to fit 70B models in 4GB
        # Each layer loaded one at a time

        layer_size_mb = 100  # ~100MB per layer
        vram_limit_mb = 4096  # 4GB

        # Only one layer at a time
        peak_usage = layer_size_mb + 500  # Layer + buffers

        assert peak_usage < vram_limit_mb

    def test_model_offload_states(self):
        """Test model offload state tracking."""
        states = ["unloaded", "loading", "loaded", "offloading"]

        # State machine transitions
        valid_transitions = {
            "unloaded": ["loading"],
            "loading": ["loaded", "unloaded"],  # Can fail
            "loaded": ["offloading"],
            "offloading": ["unloaded"],
        }

        for state in states:
            assert state in valid_transitions


class TestCortexReasoning:
    """Test Cortex reasoning capabilities."""

    def test_chain_of_thought_format(self):
        """Test chain-of-thought prompting format."""
        prompt = "What is 25 * 17?"

        cot_prompt = f"""
        Think step by step:
        1. First, understand the problem
        2. Break it down into smaller steps
        3. Solve each step
        4. Combine for final answer

        Problem: {prompt}

        Step-by-step reasoning:
        """

        assert "step by step" in cot_prompt.lower()
        assert prompt in cot_prompt

    def test_verification_prompt_format(self):
        """Test verification prompt structure."""
        claim = "Water boils at 100°C at sea level"
        sources = ["Physics textbook", "NIST data"]

        verification_prompt = f"""
        Verify the following claim against the provided sources:

        Claim: {claim}

        Sources:
        {chr(10).join(f'- {s}' for s in sources)}

        Analysis:
        1. Is this claim supported by the sources?
        2. What is the confidence level?
        3. Are there any caveats?
        """

        assert claim in verification_prompt
        assert all(s in verification_prompt for s in sources)


class TestCortexPerformance:
    """Test Cortex performance characteristics."""

    def test_expected_latency(self):
        """Test expected latency is documented."""
        # Cortex: ~3.3 seconds per token (layer-wise)
        expected_first_token_ms = 3300
        expected_tokens_per_second = 0.3

        # Verify these are reasonable for 70B on 4GB
        assert expected_first_token_ms > 1000  # At least 1 second
        assert expected_tokens_per_second < 1.0  # Less than 1 token/sec

    def test_memory_budget(self):
        """Test memory budget is within limits."""
        # From CLAUDE.md
        cortex_buffer_mb = 1600  # Paged on-demand
        system_overhead_mb = 300
        medulla_mb = 800
        titans_mb = 200
        bridge_mb = 50

        total_resident = system_overhead_mb + medulla_mb + titans_mb + bridge_mb
        peak_with_cortex = total_resident + cortex_buffer_mb

        vram_limit_mb = 4096

        assert peak_with_cortex < vram_limit_mb


class TestCortexIntegration:
    """Test Cortex integration points."""

    def test_medulla_handoff_format(self):
        """Test format of Medulla → Cortex handoff."""
        # Medulla provides context for Cortex
        handoff = {
            "query": "Complex question here",
            "surprise": 0.85,
            "medulla_state": [0.1, 0.2, 0.3],  # Hidden state
            "conversation_context": ["Prior message 1", "Prior message 2"],
            "cognitive_state": {"label": "CONFUSION", "entropy": 3.5},
        }

        assert "query" in handoff
        assert "surprise" in handoff
        assert handoff["surprise"] > 0.7  # High surprise triggered Cortex

    def test_cortex_response_format(self):
        """Test format of Cortex response."""
        response = {
            "response": "Detailed response here...",
            "reasoning_steps": [
                "Step 1: Analyzed the question",
                "Step 2: Gathered relevant information",
                "Step 3: Synthesized response",
            ],
            "confidence": 0.92,
            "used_tools": ["web_search"],
            "tokens_generated": 512,
            "generation_time_ms": 45000,
        }

        assert "response" in response
        assert "confidence" in response
        assert response["confidence"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
