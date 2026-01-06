"""Tests for the Sentinel architecture Mental Sandbox module."""

import pytest

# Import the sandbox module
from src.core.sandbox import (
    AgentFeedback,
    AgentRole,
    ButlerAgent,
    ConsensusStatus,
    MentalSandbox,
    SandboxConfig,
    SimulationResult,
    ThinkerAgent,
    VerifierAgent,
)


class TestSandboxConfig:
    """Tests for SandboxConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SandboxConfig()
        assert config.max_cycles == 5
        assert config.timeout_seconds == 30.0
        assert config.require_verification is True
        assert config.require_personalization is True
        assert config.min_confidence == 0.7

    def test_custom_config(self):
        """Test custom configuration."""
        config = SandboxConfig(
            max_cycles=3,
            min_confidence=0.9,
            require_verification=False,
        )
        assert config.max_cycles == 3
        assert config.min_confidence == 0.9
        assert config.require_verification is False


class TestSimulationResult:
    """Tests for SimulationResult."""

    def test_successful_result(self):
        """Test creating a successful simulation result."""
        result = SimulationResult(
            output="Generated response",
            consensus_status=ConsensusStatus.ACHIEVED,
            cycles_used=2,
            confidence=0.95,
        )
        assert result.output == "Generated response"
        assert result.cycles_used == 2
        assert result.confidence == 0.95
        assert result.consensus_status == ConsensusStatus.ACHIEVED

    def test_max_cycles_result(self):
        """Test creating a max cycles result."""
        result = SimulationResult(
            output="Best effort response",
            consensus_status=ConsensusStatus.MAX_CYCLES,
            cycles_used=5,
            confidence=0.6,
        )
        assert result.consensus_status == ConsensusStatus.MAX_CYCLES
        assert result.cycles_used == 5


class TestAgentFeedback:
    """Tests for AgentFeedback dataclass."""

    def test_agent_feedback_creation(self):
        """Test creating agent feedback."""
        feedback = AgentFeedback(
            agent=AgentRole.THINKER,
            approved=True,
            confidence=0.85,
            reasoning="Looks good",
        )
        assert feedback.agent == AgentRole.THINKER
        assert feedback.approved is True
        assert feedback.confidence == 0.85

    def test_agent_feedback_with_suggestions(self):
        """Test creating feedback with suggestions."""
        feedback = AgentFeedback(
            agent=AgentRole.VERIFIER,
            approved=False,
            confidence=0.6,
            reasoning="Needs improvement",
            suggestions=["Add error handling", "Check edge cases"],
        )
        assert feedback.approved is False
        assert len(feedback.suggestions) == 2


class TestThinkerAgent:
    """Tests for ThinkerAgent."""

    def test_thinker_creation(self):
        """Test creating a ThinkerAgent."""
        thinker = ThinkerAgent()
        assert thinker is not None
        assert thinker.role == AgentRole.THINKER


class TestVerifierAgent:
    """Tests for VerifierAgent."""

    def test_verifier_creation(self):
        """Test creating a VerifierAgent."""
        verifier = VerifierAgent()
        assert verifier is not None
        assert verifier.role == AgentRole.VERIFIER


class TestButlerAgent:
    """Tests for ButlerAgent."""

    def test_butler_creation(self):
        """Test creating a ButlerAgent."""
        butler = ButlerAgent()
        assert butler is not None
        assert butler.role == AgentRole.BUTLER


class TestMentalSandbox:
    """Tests for MentalSandbox class."""

    def test_sandbox_creation(self):
        """Test creating a MentalSandbox."""
        sandbox = MentalSandbox()
        assert sandbox is not None
        # Agents are keyed by AgentRole enum
        assert AgentRole.THINKER in sandbox.agents
        assert AgentRole.VERIFIER in sandbox.agents
        assert AgentRole.BUTLER in sandbox.agents

    def test_sandbox_custom_config(self):
        """Test creating a MentalSandbox with custom config."""
        config = SandboxConfig(max_cycles=3)
        sandbox = MentalSandbox(config)
        assert sandbox.config.max_cycles == 3

    @pytest.mark.asyncio
    async def test_simulate_simple_query(self):
        """Test simulating a simple query."""
        sandbox = MentalSandbox()
        result = await sandbox.simulate("What is 2+2?", {})
        assert result is not None
        assert isinstance(result, SimulationResult)
        assert result.output is not None

    @pytest.mark.asyncio
    async def test_simulate_with_context(self):
        """Test simulating with user context."""
        sandbox = MentalSandbox()
        context = {"user_name": "Alice", "preferences": {"formal": True}}
        result = await sandbox.simulate("How do I sort a list?", context)
        assert result is not None
        assert result.cycles_used >= 1

    @pytest.mark.asyncio
    async def test_simulate_max_cycles(self):
        """Test that simulation respects max cycles."""
        config = SandboxConfig(max_cycles=2)
        sandbox = MentalSandbox(config)
        result = await sandbox.simulate("Complex philosophical question", {})
        assert result.cycles_used <= config.max_cycles


class TestSandboxIntegration:
    """Integration tests for Mental Sandbox."""

    @pytest.mark.asyncio
    async def test_full_simulation_cycle(self):
        """Test a full simulation cycle with all agents."""
        sandbox = MentalSandbox()
        context = {
            "user_name": "TestUser",
            "user_level": "intermediate",
            "preferences": {"concise": True},
        }
        result = await sandbox.simulate(
            "Explain the difference between a list and a tuple in Python",
            context,
        )
        assert result is not None
        assert result.output is not None
        assert result.cycles_used >= 1
        assert 0 <= result.confidence <= 1

    @pytest.mark.asyncio
    async def test_code_query_simulation(self):
        """Test simulation for a code-related query."""
        sandbox = MentalSandbox()
        result = await sandbox.simulate(
            "Write a function to calculate fibonacci numbers",
            {},
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_factual_query_simulation(self):
        """Test simulation for a factual query."""
        sandbox = MentalSandbox()
        result = await sandbox.simulate(
            "What is the capital of France?",
            {},
        )
        assert result is not None
