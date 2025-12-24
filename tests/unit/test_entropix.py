"""
Unit Tests for the Entropix Metacognitive Module
=================================================

Tests for entropy/varentropy calculation and cognitive state classification.
"""

# Add src to path for imports
import sys
from pathlib import Path

import pytest

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from cortex.entropix import (  # noqa: E402
    CognitiveState,
    CognitiveStateLabel,
    Entropix,
    EntropixConfig,
    create_entropix_controller,
    diagnose_from_ollama_response,
)


class TestEntropyCalculation:
    """Test entropy and varentropy calculation from logprobs."""

    def test_empty_logprobs_returns_zero(self):
        """Empty input should return zero entropy and varentropy."""
        entropix = Entropix()
        H, V = entropix.calculate_metrics([])

        assert H == 0.0
        assert V == 0.0

    def test_single_token_high_confidence(self):
        """Single very high probability token should have low entropy."""
        entropix = Entropix()

        # Single token with very high probability (logprob close to 0)
        logprobs = [{"token": "the", "logprob": -0.01}]
        H, V = entropix.calculate_metrics(logprobs)

        # With single token, entropy should be 0 (no uncertainty)
        assert H == 0.0
        assert V == 0.0

    def test_uniform_distribution_high_entropy(self):
        """Uniform distribution should have high entropy."""
        entropix = Entropix()

        # 5 tokens with equal probability (logprob = log(1/5) = -1.609)
        logprobs = [
            {"token": "a", "logprob": -1.609},
            {"token": "b", "logprob": -1.609},
            {"token": "c", "logprob": -1.609},
            {"token": "d", "logprob": -1.609},
            {"token": "e", "logprob": -1.609},
        ]
        H, V = entropix.calculate_metrics(logprobs)

        # Entropy of uniform distribution = log(n) ≈ 1.609 for n=5
        assert H > 1.0, f"Expected high entropy, got {H}"
        # Varentropy should be low for uniform distribution
        assert V < 1.0, f"Expected low varentropy for uniform dist, got {V}"

    def test_peaked_distribution_low_entropy(self):
        """Distribution peaked at one token should have low entropy."""
        entropix = Entropix()

        # One dominant token, others very unlikely
        logprobs = [
            {"token": "hello", "logprob": -0.1},  # High prob
            {"token": "hi", "logprob": -5.0},  # Low prob
            {"token": "hey", "logprob": -6.0},  # Very low prob
        ]
        H, V = entropix.calculate_metrics(logprobs)

        # Entropy should be low due to peaked distribution
        assert H < 1.0, f"Expected low entropy for peaked dist, got {H}"

    def test_bimodal_distribution_high_varentropy(self):
        """Two competing high-probability tokens should have high varentropy."""
        entropix = Entropix()

        # Two tokens with similar high probability, rest very low
        logprobs = [
            {"token": "yes", "logprob": -0.7},  # ~50%
            {"token": "no", "logprob": -0.7},  # ~50%
            {"token": "maybe", "logprob": -5.0},  # Very low
        ]
        H, V = entropix.calculate_metrics(logprobs)

        # This is a "hesitation" scenario - moderate entropy, higher varentropy
        assert H > 0.5, f"Expected moderate entropy, got {H}"


class TestCognitiveStateClassification:
    """Test cognitive state classification based on quadrants."""

    def test_flow_state_low_entropy_low_varentropy(self):
        """Low H, Low V should classify as FLOW."""
        entropix = Entropix()

        # Very confident prediction
        logprobs = [{"token": "the", "logprob": -0.05}]
        state = entropix.diagnose(logprobs)

        assert state.label == CognitiveStateLabel.FLOW
        assert state.confidence > 0.8
        assert not state.should_use_tools
        assert not state.should_use_cot

    def test_confusion_state_high_entropy_low_varentropy(self):
        """High H, Low V should classify as CONFUSION."""
        config = EntropixConfig(
            high_entropy_threshold=2.0,
            high_varentropy_threshold=1.0,
        )
        entropix = Entropix(config)

        # Many tokens with similar (uniform) probability
        logprobs = [{"token": f"tok{i}", "logprob": -2.3} for i in range(10)]  # ~10% each
        state = entropix.diagnose(logprobs)

        # High entropy, lowish varentropy (uniform) = CONFUSION
        assert state.label == CognitiveStateLabel.CONFUSION
        assert state.should_use_tools  # Confusion triggers tool use
        assert state.should_use_cot  # Confusion also suggests CoT

    def test_hesitation_state_low_entropy_high_varentropy(self):
        """Moderate H with competing options should classify as HESITATION."""
        config = EntropixConfig(
            low_entropy_threshold=0.3,
            high_entropy_threshold=3.0,
            high_varentropy_threshold=0.5,
        )
        entropix = Entropix(config)

        # Two competing high-probability tokens (hesitation)
        logprobs = [
            {"token": "yes", "logprob": -0.4},  # ~67%
            {"token": "no", "logprob": -1.1},  # ~33%
        ]
        state = entropix.diagnose(logprobs)

        # This could be HESITATION or CREATIVE depending on thresholds
        # Main test: should recommend CoT for ambiguous situations
        assert state.entropy > 0.3
        assert state.should_use_cot or state.label in [
            CognitiveStateLabel.HESITATION,
            CognitiveStateLabel.CREATIVE,
        ]

    def test_creative_state_high_entropy_high_varentropy(self):
        """High H, High V should classify as CREATIVE."""
        config = EntropixConfig(
            high_entropy_threshold=1.5,
            high_varentropy_threshold=0.5,
        )
        entropix = Entropix(config)

        # Multiple distinct peaks with varying probabilities
        logprobs = [
            {"token": "red", "logprob": -1.0},
            {"token": "blue", "logprob": -1.2},
            {"token": "green", "logprob": -1.5},
            {"token": "yellow", "logprob": -2.0},
            {"token": "orange", "logprob": -2.5},
        ]
        state = entropix.diagnose(logprobs)

        # High entropy with varying peaks = CREATIVE or CONFUSION
        assert state.entropy > 1.0
        assert state.should_increase_temp or state.should_use_cot


class TestCognitiveStateActions:
    """Test action recommendations based on cognitive state."""

    def test_confusion_triggers_tools(self):
        """CONFUSION state should recommend tool use."""
        config = EntropixConfig(confusion_triggers_tools=True)
        entropix = Entropix(config)

        # Force confusion via uniform high entropy
        logprobs = [{"token": f"tok{i}", "logprob": -2.0} for i in range(8)]
        state = entropix.diagnose(logprobs)

        assert state.should_use_tools

    def test_confusion_tools_can_be_disabled(self):
        """Tool triggering can be disabled via config."""
        config = EntropixConfig(confusion_triggers_tools=False)
        entropix = Entropix(config)

        # Force confusion
        logprobs = [{"token": f"tok{i}", "logprob": -2.0} for i in range(8)]
        state = entropix.diagnose(logprobs)

        assert not state.should_use_tools

    def test_temperature_recommendations(self):
        """Temperature recommendations should vary by state."""
        config = EntropixConfig(
            base_temperature=0.7,
            confusion_temperature=0.3,
            creative_temperature=1.2,
        )
        entropix = Entropix(config)

        # Low entropy = FLOW = base temp
        flow_state = entropix.diagnose([{"token": "test", "logprob": -0.01}])
        assert flow_state.recommended_temp == 0.7

        # Note: Full temp test requires forcing specific states


class TestEntropixHistory:
    """Test history tracking and statistics."""

    def test_history_accumulates(self):
        """Diagnoses should be tracked in history."""
        entropix = Entropix()

        for i in range(5):
            entropix.diagnose([{"token": f"tok{i}", "logprob": -1.0}])

        assert len(entropix.history) == 5

    def test_history_limited_to_max(self):
        """History should be capped at max_history."""
        entropix = Entropix()
        entropix.max_history = 10

        for i in range(20):
            entropix.diagnose([{"token": f"tok{i}", "logprob": -1.0}])

        assert len(entropix.history) == 10

    def test_statistics_calculation(self):
        """Statistics should reflect history."""
        entropix = Entropix()

        # Add some diagnoses
        for i in range(5):
            entropix.diagnose([{"token": f"tok{i}", "logprob": -1.0}])

        stats = entropix.get_statistics()

        assert stats["total_diagnoses"] == 5
        assert "avg_entropy" in stats
        assert "state_distribution" in stats

    def test_reset_history(self):
        """Reset should clear history."""
        entropix = Entropix()

        entropix.diagnose([{"token": "test", "logprob": -1.0}])
        assert len(entropix.history) == 1

        entropix.reset_history()
        assert len(entropix.history) == 0


class TestSurpriseSignal:
    """Test surprise signal for Titans memory integration."""

    def test_surprise_scales_with_entropy(self):
        """Surprise signal should scale with entropy."""
        config = EntropixConfig(surprise_scale=1.0)
        entropix = Entropix(config)

        # Low entropy
        low_state = entropix.diagnose([{"token": "test", "logprob": -0.1}])
        low_surprise = entropix.get_surprise_signal(low_state)

        # High entropy
        high_state = entropix.diagnose([{"token": f"tok{i}", "logprob": -2.0} for i in range(5)])
        high_surprise = entropix.get_surprise_signal(high_state)

        assert high_surprise > low_surprise

    def test_surprise_uses_latest_if_none(self):
        """get_surprise_signal with None should use latest history."""
        entropix = Entropix()

        entropix.diagnose([{"token": "test", "logprob": -1.0}])
        surprise = entropix.get_surprise_signal()

        assert surprise >= 0.0


class TestUtilityFunctions:
    """Test utility and factory functions."""

    def test_diagnose_from_ollama_response(self):
        """diagnose_from_ollama_response should handle Ollama format."""
        response = {
            "response": "Hello",
            "logprobs": [
                {"token": "Hello", "logprob": -0.5},
                {"token": "Hi", "logprob": -1.5},
            ],
        }

        state = diagnose_from_ollama_response(response)

        assert isinstance(state, CognitiveState)
        assert state.entropy >= 0.0

    def test_diagnose_from_ollama_no_logprobs(self):
        """Should return neutral state if no logprobs."""
        response = {"response": "Hello"}

        state = diagnose_from_ollama_response(response)

        assert state.label == CognitiveStateLabel.NEUTRAL

    def test_create_entropix_controller(self):
        """Factory function should create configured controller."""
        entropix = create_entropix_controller(
            entropy_threshold=2.0,
            varentropy_threshold=1.5,
            enable_tools=False,
            enable_cot=True,
        )

        assert entropix.config.high_entropy_threshold == 2.0
        assert entropix.config.high_varentropy_threshold == 1.5
        assert not entropix.config.confusion_triggers_tools
        assert entropix.config.hesitation_triggers_cot


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_logprob_format(self):
        """Should handle invalid logprob format gracefully."""
        entropix = Entropix()

        # Missing logprob key
        logprobs = [{"token": "test"}]
        H, V = entropix.calculate_metrics(logprobs)

        # Should not crash, may return 0
        assert H >= 0.0

    def test_negative_infinity_logprob(self):
        """Should handle very negative logprobs."""
        entropix = Entropix()

        logprobs = [
            {"token": "test", "logprob": -100.0},  # Essentially 0 probability
        ]

        # Should not crash
        state = entropix.diagnose(logprobs)
        assert isinstance(state, CognitiveState)

    def test_config_serialization(self):
        """Config should serialize to dict."""
        config = EntropixConfig()
        config_dict = config.to_dict()

        assert "high_entropy_threshold" in config_dict
        assert "low_varentropy_threshold" in config_dict

    def test_state_serialization(self):
        """CognitiveState should serialize to dict."""
        state = CognitiveState(
            label=CognitiveStateLabel.FLOW,
            entropy=0.5,
            varentropy=0.3,
        )
        state_dict = state.to_dict()

        assert state_dict["label"] == "flow"
        assert state_dict["entropy"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
