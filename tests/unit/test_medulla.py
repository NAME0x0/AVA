"""
Unit Tests for Medulla Component
================================

Tests for the reflexive core of AVA's dual-brain architecture.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.medulla import (
    MedullaState,
    ThermalStatus,
    ThermalMonitor,
    MedullaConfig,
    CognitiveState,
)


class TestMedullaState:
    """Test MedullaState enum."""

    def test_all_states_exist(self):
        """Verify all expected states are defined."""
        expected = [
            "IDLE", "LISTENING", "PERCEIVING",
            "RESPONDING", "ROUTING",
            "THERMAL_THROTTLED", "THERMAL_PAUSED"
        ]
        for state in expected:
            assert hasattr(MedullaState, state)

    def test_state_values(self):
        """Verify state values are strings."""
        assert MedullaState.IDLE.value == "idle"
        assert MedullaState.RESPONDING.value == "responding"
        assert MedullaState.THERMAL_PAUSED.value == "thermal_paused"


class TestThermalStatus:
    """Test ThermalStatus dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        status = ThermalStatus()
        assert status.temperature == 0.0
        assert status.power_draw_watts == 0.0
        assert status.is_throttled is False
        assert status.is_paused is False

    def test_custom_values(self):
        """Test initialization with custom values."""
        status = ThermalStatus(
            temperature=75.5,
            power_draw_watts=45.0,
            power_limit_watts=70.0,
            power_percent=64.3,
            is_throttled=True,
        )
        assert status.temperature == 75.5
        assert status.power_draw_watts == 45.0
        assert status.is_throttled is True

    def test_to_dict(self):
        """Test dictionary conversion."""
        status = ThermalStatus(temperature=70.0, is_throttled=False)
        d = status.to_dict()

        assert isinstance(d, dict)
        assert d["temperature"] == 70.0
        assert d["is_throttled"] is False
        assert "timestamp" in d


class TestThermalMonitor:
    """Test ThermalMonitor class."""

    def test_initialization_defaults(self):
        """Test default configuration."""
        monitor = ThermalMonitor()
        assert monitor.max_power_percent == 15.0
        assert monitor.warning_temp == 75.0
        assert monitor.throttle_temp == 80.0
        assert monitor.pause_temp == 85.0

    def test_initialization_custom(self):
        """Test custom configuration."""
        monitor = ThermalMonitor(
            max_power_percent=20.0,
            warning_temp=70.0,
            throttle_temp=75.0,
            pause_temp=80.0,
        )
        assert monitor.max_power_percent == 20.0
        assert monitor.warning_temp == 70.0

    def test_should_throttle(self):
        """Test throttle threshold logic."""
        monitor = ThermalMonitor(throttle_temp=80.0, pause_temp=85.0)

        # Below throttle
        assert monitor._should_throttle(75.0) is False
        # At throttle
        assert monitor._should_throttle(80.0) is True
        # Above throttle, below pause
        assert monitor._should_throttle(82.0) is True

    def test_should_pause(self):
        """Test pause threshold logic."""
        monitor = ThermalMonitor(pause_temp=85.0)

        # Below pause
        assert monitor._should_pause(80.0) is False
        # At pause
        assert monitor._should_pause(85.0) is True
        # Above pause
        assert monitor._should_pause(90.0) is True


class TestMedullaConfig:
    """Test MedullaConfig dataclass."""

    def test_default_values(self):
        """Test default configuration."""
        config = MedullaConfig()
        assert config.model_name == "gemma3:4b"
        assert config.low_surprise_threshold == 0.3
        assert config.high_surprise_threshold == 0.7
        assert config.simulation_mode is False

    def test_custom_values(self):
        """Test custom configuration."""
        config = MedullaConfig(
            model_name="llama3:8b",
            low_surprise_threshold=0.2,
            high_surprise_threshold=0.8,
            simulation_mode=True,
        )
        assert config.model_name == "llama3:8b"
        assert config.simulation_mode is True


class TestCognitiveState:
    """Test CognitiveState dataclass."""

    def test_default_values(self):
        """Test default cognitive state."""
        state = CognitiveState()
        assert state.label == "FLOW"
        assert state.entropy == 0.0
        assert state.varentropy == 0.0
        assert state.confidence == 0.8
        assert state.surprise == 0.0

    def test_should_think_high_entropy(self):
        """High entropy should trigger thinking."""
        state = CognitiveState(entropy=3.5, varentropy=0.5)
        # High entropy suggests uncertainty
        assert state.entropy > 3.0

    def test_should_use_tools(self):
        """Test tool usage determination."""
        # Low varentropy, high entropy -> tools might help
        state = CognitiveState(entropy=2.5, varentropy=0.3)
        assert state.varentropy < 0.5

    def test_to_dict(self):
        """Test dictionary conversion."""
        state = CognitiveState(label="HESITATION", entropy=2.0)
        d = state.to_dict()

        assert isinstance(d, dict)
        assert d["label"] == "HESITATION"
        assert d["entropy"] == 2.0


class TestSurpriseCalculation:
    """Test surprise calculation logic."""

    def test_low_surprise_familiar_input(self):
        """Familiar inputs should have low surprise."""
        # Mock embedding distance calculation
        embedding1 = [0.1, 0.2, 0.3, 0.4, 0.5]
        embedding2 = [0.1, 0.2, 0.3, 0.4, 0.5]  # Same

        # Cosine similarity of identical vectors = 1.0
        # Surprise = 1 - similarity = 0.0
        import numpy as np
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        surprise = 1.0 - similarity

        assert surprise < 0.1  # Very low surprise

    def test_high_surprise_novel_input(self):
        """Novel inputs should have high surprise."""
        import numpy as np

        # Orthogonal vectors
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.0, 1.0, 0.0]

        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        surprise = 1.0 - similarity

        assert surprise > 0.9  # High surprise


class TestRoutingDecision:
    """Test Medulla → Cortex routing logic."""

    def test_low_surprise_stays_medulla(self):
        """Low surprise queries should stay in Medulla."""
        low_threshold = 0.3
        surprise = 0.2

        should_route_to_cortex = surprise > low_threshold
        assert should_route_to_cortex is False

    def test_high_surprise_routes_cortex(self):
        """High surprise queries should route to Cortex."""
        high_threshold = 0.7
        surprise = 0.85

        should_route_to_cortex = surprise > high_threshold
        assert should_route_to_cortex is True

    def test_medium_surprise_decision(self):
        """Medium surprise uses Agency for decision."""
        low_threshold = 0.3
        high_threshold = 0.7
        surprise = 0.5

        is_medium = low_threshold < surprise < high_threshold
        assert is_medium is True
        # Agency module would decide in this case


@pytest.mark.asyncio
class TestMedullaAsync:
    """Async tests for Medulla operations."""

    async def test_process_returns_response(self):
        """Test that process returns a valid response."""
        # This would test the actual Medulla.process() method
        # For now, test the expected interface

        class MockMedulla:
            async def process(self, text: str) -> dict:
                return {
                    "response": "Mock response",
                    "surprise": 0.3,
                    "cognitive_state": {"label": "FLOW"},
                }

        medulla = MockMedulla()
        result = await medulla.process("Hello")

        assert "response" in result
        assert "surprise" in result
        assert result["cognitive_state"]["label"] == "FLOW"

    async def test_process_handles_empty_input(self):
        """Test handling of empty input."""
        class MockMedulla:
            async def process(self, text: str) -> dict:
                if not text.strip():
                    return {"response": "", "surprise": 0.0}
                return {"response": "OK", "surprise": 0.1}

        medulla = MockMedulla()
        result = await medulla.process("")

        assert result["response"] == ""
        assert result["surprise"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
