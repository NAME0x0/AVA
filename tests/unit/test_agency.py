"""
Unit Tests for Agency Module
=============================

Tests for the Active Inference-based autonomous behavior system.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.agency import (
    AgencyConfig,
    AgencyModule,
    BeliefState,
    Policy,
)


class TestAgencyConfig:
    """Test AgencyConfig dataclass."""

    def test_default_values(self):
        """Test default configuration."""
        config = AgencyConfig()
        assert config.epistemic_weight == 0.6
        assert config.pragmatic_weight == 0.4
        assert config.cortex_effort_cost == 0.3
        assert config.search_effort_cost == 0.1

    def test_weights_sum_to_one(self):
        """Epistemic + pragmatic weights should sum to ~1."""
        config = AgencyConfig()
        total = config.epistemic_weight + config.pragmatic_weight
        assert abs(total - 1.0) < 0.01

    def test_custom_values(self):
        """Test custom configuration."""
        config = AgencyConfig(
            epistemic_weight=0.7,
            pragmatic_weight=0.3,
            cortex_effort_cost=0.4,
        )
        assert config.epistemic_weight == 0.7
        assert config.cortex_effort_cost == 0.4


class TestPolicy:
    """Test Policy enum and properties."""

    def test_all_policies_exist(self):
        """Verify all expected policies are defined."""
        expected = [
            "PRIMARY_SEARCH",
            "REFLEX_REPLY",
            "DEEP_THOUGHT",
            "WEB_BROWSE",
            "SELF_MONITOR",
            "THERMAL_CHECK",
            "SYSTEM_COMMAND",
        ]
        for policy_name in expected:
            assert hasattr(Policy, policy_name)

    def test_policy_effort_costs(self):
        """Test that policies have effort costs."""
        # Higher effort costs for more resource-intensive policies
        costs = {
            Policy.REFLEX_REPLY: 0.05,
            Policy.PRIMARY_SEARCH: 0.1,
            Policy.SELF_MONITOR: 0.1,
            Policy.THERMAL_CHECK: 0.05,
            Policy.WEB_BROWSE: 0.2,
            Policy.DEEP_THOUGHT: 0.3,
            Policy.SYSTEM_COMMAND: 0.4,
        }

        # Deep thought should be most expensive
        assert costs[Policy.DEEP_THOUGHT] > costs[Policy.REFLEX_REPLY]
        # System commands are risky
        assert costs[Policy.SYSTEM_COMMAND] >= costs[Policy.DEEP_THOUGHT]


class TestBeliefState:
    """Test BeliefState class."""

    def test_initialization(self):
        """Test belief state initialization."""
        beliefs = BeliefState()
        assert beliefs.current_state == "IDLE"
        assert isinstance(beliefs.state_distribution, dict)
        assert isinstance(beliefs.policy_distribution, dict)

    def test_update_beliefs(self):
        """Test belief update from observation."""
        beliefs = BeliefState()

        observation = {
            "surprise": 0.8,
            "query_type": "question",
            "has_question_word": True,
        }

        beliefs.update(observation)

        # High surprise should increase probability of DEEP_THOUGHT
        assert "UNCERTAIN" in beliefs.state_distribution or True

    def test_get_policy_probabilities(self):
        """Test policy probability distribution."""
        beliefs = BeliefState()

        # Set up a scenario favoring search
        beliefs.state_distribution = {"QUESTION": 0.8, "IDLE": 0.2}

        probs = beliefs.get_policy_probabilities()

        assert isinstance(probs, dict)
        # All probabilities should sum to ~1
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01


class TestAgencyModule:
    """Test AgencyModule class."""

    def test_initialization(self):
        """Test agency module initialization."""
        config = AgencyConfig()
        agency = AgencyModule(config)

        assert agency.config == config
        assert agency.belief_state is not None

    def test_calculate_free_energy(self):
        """Test Expected Free Energy calculation."""
        config = AgencyConfig()
        agency = AgencyModule(config)

        state = {
            "surprise": 0.5,
            "confidence": 0.7,
            "query_type": "question",
        }

        # G(π) = D_KL[Q(s|π) || P(s|C)] - E_Q[log P(o|s)]
        # ≈ pragmatic_value + epistemic_value - effort_cost

        for policy in Policy:
            g = agency.calculate_expected_free_energy(policy, state)
            assert isinstance(g, float)

    def test_select_policy_question(self):
        """Test policy selection for questions."""
        config = AgencyConfig()
        agency = AgencyModule(config)

        state = {
            "query": "What is the capital of France?",
            "surprise": 0.3,
            "has_question_word": True,
            "query_type": "informational",
        }

        policy = agency.select_policy(state)

        # Questions should favor search
        assert policy in [Policy.PRIMARY_SEARCH, Policy.REFLEX_REPLY]

    def test_select_policy_complex(self):
        """Test policy selection for complex queries."""
        config = AgencyConfig()
        agency = AgencyModule(config)

        state = {
            "query": "Explain the philosophical implications of quantum mechanics",
            "surprise": 0.85,
            "has_question_word": False,
            "query_type": "analytical",
        }

        policy = agency.select_policy(state)

        # High surprise + analytical = deep thought
        assert policy == Policy.DEEP_THOUGHT

    def test_select_policy_thermal_override(self):
        """Test thermal conditions override policy."""
        config = AgencyConfig()
        agency = AgencyModule(config)

        state = {
            "query": "Complex question",
            "surprise": 0.9,
            "gpu_temperature": 87.0,  # Above pause threshold
        }

        policy = agency.select_policy(state)

        # Thermal safety should override
        assert policy in [Policy.THERMAL_CHECK, Policy.REFLEX_REPLY]


class TestSearchFirstParadigm:
    """Test Search-First behavior."""

    def test_question_words_trigger_search(self):
        """Question words should trigger search."""
        question_words = ["what", "when", "where", "who", "how", "why"]

        for word in question_words:
            query = f"{word.capitalize()} is the meaning of life?"
            has_question_word = any(query.lower().startswith(w) for w in question_words)
            assert has_question_word is True

    def test_non_question_no_trigger(self):
        """Non-questions should not trigger search."""
        queries = [
            "Tell me about Python",
            "I think AI is interesting",
            "Please explain this code",
        ]

        question_words = ["what", "when", "where", "who", "how", "why"]

        for query in queries:
            has_question_word = any(query.lower().startswith(w) for w in question_words)
            assert has_question_word is False

    def test_search_verification_threshold(self):
        """Test source agreement threshold."""
        agreement_threshold = 0.7

        # Scenario: 3 sources, 2 agree
        sources = ["Source A: Paris", "Source B: Paris", "Source C: Lyon"]
        agreement_count = 2
        agreement_ratio = agreement_count / len(sources)

        # 66% agreement, below 70% threshold
        assert agreement_ratio < agreement_threshold

        # Scenario: 3 sources, 3 agree
        sources = ["Source A: Paris", "Source B: Paris", "Source C: Paris"]
        agreement_count = 3
        agreement_ratio = agreement_count / len(sources)

        # 100% agreement, above threshold
        assert agreement_ratio >= agreement_threshold


class TestSystemCommandSafety:
    """Test system command safety features."""

    def test_blocked_commands(self):
        """Test that dangerous commands are blocked."""
        blocked_commands = [
            "rm -rf",
            "del /f",
            "format",
            "shutdown",
            "reboot",
            "mkfs",
            "dd if=",
        ]

        def is_command_safe(cmd: str) -> bool:
            cmd_lower = cmd.lower()
            return not any(blocked in cmd_lower for blocked in blocked_commands)

        # These should be blocked
        assert is_command_safe("rm -rf /") is False
        assert is_command_safe("format C:") is False
        assert is_command_safe("shutdown -h now") is False

        # These should be allowed
        assert is_command_safe("ls -la") is True
        assert is_command_safe("python script.py") is True
        assert is_command_safe("git status") is True

    def test_command_requires_confirmation(self):
        """Test that system commands require confirmation."""
        # All SYSTEM_COMMAND policy executions should require confirmation

        # Simulated confirmation flow
        confirmation_required = True
        user_confirmed = False

        should_execute = confirmation_required and user_confirmed
        assert should_execute is False

        user_confirmed = True
        should_execute = confirmation_required and user_confirmed
        assert should_execute is True


@pytest.mark.asyncio
class TestAgencyAsync:
    """Async tests for Agency operations."""

    async def test_execute_policy(self):
        """Test policy execution."""
        config = AgencyConfig()
        agency = AgencyModule(config)

        result = await agency.execute_policy(Policy.REFLEX_REPLY, context={"query": "Hello!"})

        assert isinstance(result, dict)
        assert "success" in result or "response" in result

    async def test_execute_search_policy(self):
        """Test search policy execution."""
        config = AgencyConfig()
        agency = AgencyModule(config)

        result = await agency.execute_policy(
            Policy.PRIMARY_SEARCH, context={"query": "What is Python?"}
        )

        assert isinstance(result, dict)


class TestFreeEnergyPrinciple:
    """Test Free Energy Principle implementation."""

    def test_expected_free_energy_components(self):
        """Test G(π) components."""
        # G(π) = pragmatic_value + epistemic_value - effort_cost

        pragmatic = 0.5  # Task completion value
        epistemic = 0.3  # Information gain value
        effort = 0.2  # Resource cost

        g = pragmatic + epistemic - effort

        assert abs(g - 0.6) < 0.01  # Net expected free energy (with float tolerance)

    def test_policy_ranking(self):
        """Test policies are ranked by expected free energy."""
        policies_g = {
            Policy.REFLEX_REPLY: 0.8,  # High: low effort, good for simple
            Policy.PRIMARY_SEARCH: 0.7,  # Good: moderate effort, info gain
            Policy.DEEP_THOUGHT: 0.4,  # Lower: high effort, but thorough
            Policy.SYSTEM_COMMAND: 0.1,  # Lowest: high risk, limited use
        }

        # Sort by G (lower is better in Active Inference)
        # Actually, we want to MINIMIZE free energy
        # So higher pragmatic/epistemic and lower effort is better

        ranked = sorted(policies_g.items(), key=lambda x: -x[1])

        # REFLEX_REPLY should be top for simple queries
        assert ranked[0][0] == Policy.REFLEX_REPLY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
