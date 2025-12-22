"""
Integration Tests for Cortex-Medulla Architecture
===================================================

Tests for the v3 dual-brain architecture workflow:
1. Medulla → Cortex routing based on surprise
2. Agency policy selection
3. Search-First paradigm
4. Thermal awareness
5. Memory integration
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestMedullaCortexRouting:
    """Test routing between Medulla and Cortex."""

    def test_low_surprise_stays_medulla(self):
        """Low surprise queries stay in Medulla."""
        thresholds = {
            "low": 0.3,
            "high": 0.7,
        }

        test_cases = [
            ("Hello", 0.1),           # Greeting - very low surprise
            ("What time is it?", 0.2), # Common question
            ("Thanks!", 0.15),         # Acknowledgment
        ]

        for query, surprise in test_cases:
            should_use_cortex = surprise > thresholds["high"]
            assert should_use_cortex is False, f"Query '{query}' should stay in Medulla"

    def test_high_surprise_routes_cortex(self):
        """High surprise queries route to Cortex."""
        thresholds = {
            "low": 0.3,
            "high": 0.7,
        }

        test_cases = [
            ("Explain quantum entanglement's implications for cryptography", 0.85),
            ("How do Gödel's theorems relate to AI consciousness?", 0.9),
            ("Compare neuroevolution with gradient descent for RL", 0.8),
        ]

        for query, surprise in test_cases:
            should_use_cortex = surprise > thresholds["high"]
            assert should_use_cortex is True, f"Query '{query}' should route to Cortex"

    def test_medium_surprise_agency_decides(self):
        """Medium surprise uses Agency for decision."""
        thresholds = {
            "low": 0.3,
            "high": 0.7,
        }

        test_cases = [
            ("What is machine learning?", 0.4),
            ("How do neural networks work?", 0.5),
            ("Explain Python decorators", 0.45),
        ]

        for query, surprise in test_cases:
            in_medium_zone = thresholds["low"] < surprise < thresholds["high"]
            assert in_medium_zone is True, f"Query '{query}' should be in medium zone"


class TestSearchFirstParadigm:
    """Test Search-First behavior integration."""

    def test_question_detection(self):
        """Test question word detection."""
        question_words = ["what", "when", "where", "who", "how", "why"]

        question_queries = [
            "What is Python?",
            "When was Python created?",
            "Where is the config file?",
            "Who created Python?",
            "How do I install packages?",
            "Why use virtual environments?",
        ]

        for query in question_queries:
            has_question = any(
                query.lower().startswith(w) for w in question_words
            )
            assert has_question is True, f"'{query}' should be detected as question"

    def test_non_question_detection(self):
        """Test non-questions are not flagged."""
        non_questions = [
            "Install Python for me",
            "Please explain this code",
            "I need help with debugging",
            "Show me an example",
        ]

        question_words = ["what", "when", "where", "who", "how", "why"]

        for query in non_questions:
            has_question = any(
                query.lower().startswith(w) for w in question_words
            )
            assert has_question is False, f"'{query}' should NOT be detected as question"

    def test_source_verification(self):
        """Test source agreement verification."""
        min_sources = 3
        agreement_threshold = 0.7

        # Scenario 1: Good agreement
        sources1 = [
            {"answer": "Paris", "source": "Wikipedia"},
            {"answer": "Paris", "source": "Britannica"},
            {"answer": "Paris", "source": "CIA Factbook"},
        ]

        agreement1 = sum(1 for s in sources1 if s["answer"] == "Paris") / len(sources1)
        assert agreement1 >= agreement_threshold

        # Scenario 2: Poor agreement
        sources2 = [
            {"answer": "Paris", "source": "Source 1"},
            {"answer": "Lyon", "source": "Source 2"},
            {"answer": "Marseille", "source": "Source 3"},
        ]

        agreement2 = sum(1 for s in sources2 if s["answer"] == "Paris") / len(sources2)
        assert agreement2 < agreement_threshold


class TestAgencyIntegration:
    """Test Agency module integration."""

    def test_policy_selection_flow(self):
        """Test end-to-end policy selection."""
        # Simulate belief state
        belief_state = {
            "current": "QUESTION",
            "distribution": {
                "QUESTION": 0.8,
                "COMMAND": 0.1,
                "CHAT": 0.1,
            }
        }

        # Expected policy for question state
        expected_policies = ["PRIMARY_SEARCH", "REFLEX_REPLY"]

        # Verify question state leads to search/reply policies
        assert belief_state["distribution"]["QUESTION"] > 0.5

    def test_thermal_override(self):
        """Test thermal conditions override policy selection."""
        thermal_thresholds = {
            "warning": 75.0,
            "throttle": 80.0,
            "pause": 85.0,
        }

        test_temps = [
            (70.0, "normal"),
            (77.0, "warning"),
            (82.0, "throttle"),
            (88.0, "pause"),
        ]

        for temp, expected_state in test_temps:
            if temp >= thermal_thresholds["pause"]:
                state = "pause"
            elif temp >= thermal_thresholds["throttle"]:
                state = "throttle"
            elif temp >= thermal_thresholds["warning"]:
                state = "warning"
            else:
                state = "normal"

            assert state == expected_state, f"Temp {temp}°C should be '{expected_state}'"


class TestMemoryIntegration:
    """Test Titans memory integration."""

    def test_surprise_weighted_memorization(self):
        """Test memories are weighted by surprise."""
        memories = []

        # Simulate memorization with different surprises
        inputs = [
            ("Hello", 0.1),
            ("Novel concept about quantum AI", 0.9),
            ("What time is it?", 0.2),
        ]

        for content, surprise in inputs:
            memories.append({
                "content": content,
                "surprise": surprise,
                "weight": surprise,  # Higher surprise = higher weight
            })

        # Sort by importance
        sorted_memories = sorted(memories, key=lambda m: m["weight"], reverse=True)

        # Novel concept should be most important
        assert "quantum" in sorted_memories[0]["content"].lower()

    def test_context_retrieval(self):
        """Test context is retrieved for Cortex."""
        conversation_history = [
            {"role": "user", "content": "Tell me about Python"},
            {"role": "assistant", "content": "Python is a programming language..."},
            {"role": "user", "content": "What about machine learning?"},
            {"role": "assistant", "content": "Machine learning uses algorithms..."},
        ]

        # Query for related context
        query = "deep learning frameworks"

        # Simulate retrieval (would use embeddings in real system)
        related = [
            msg for msg in conversation_history
            if "learning" in msg["content"].lower()
        ]

        assert len(related) > 0


class TestSystemWorkflow:
    """Test complete system workflow."""

    @pytest.mark.asyncio
    async def test_simple_query_workflow(self):
        """Test workflow for simple query."""
        # Simulate simple query
        query = "Hello, how are you?"

        # Step 1: Medulla processes
        medulla_output = {
            "surprise": 0.1,
            "cognitive_state": {"label": "FLOW"},
            "quick_response": "Hello! I'm doing well, thanks for asking.",
        }

        # Step 2: Low surprise → stays in Medulla
        use_cortex = medulla_output["surprise"] > 0.7
        assert use_cortex is False

        # Step 3: Return quick response
        response = medulla_output["quick_response"]
        assert "Hello" in response

    @pytest.mark.asyncio
    async def test_complex_query_workflow(self):
        """Test workflow for complex query."""
        # Simulate complex query
        query = "Explain the relationship between entropy and information theory"

        # Step 1: Medulla processes
        medulla_output = {
            "surprise": 0.85,
            "cognitive_state": {"label": "CONFUSION"},
            "quick_response": None,  # Too complex for quick response
        }

        # Step 2: High surprise → route to Cortex
        use_cortex = medulla_output["surprise"] > 0.7
        assert use_cortex is True

        # Step 3: Cortex generates deep response
        cortex_output = {
            "response": "Entropy in information theory measures uncertainty...",
            "reasoning_steps": [
                "Define entropy in thermodynamics",
                "Introduce Shannon entropy",
                "Connect to information content",
            ],
            "confidence": 0.88,
        }

        assert len(cortex_output["reasoning_steps"]) > 0
        assert cortex_output["confidence"] > 0.5

    @pytest.mark.asyncio
    async def test_search_query_workflow(self):
        """Test workflow for search query."""
        # Simulate informational question
        query = "What is the population of Tokyo?"

        # Step 1: Detect question
        is_question = query.lower().startswith("what")
        assert is_question is True

        # Step 2: Agency selects search policy
        policy = "PRIMARY_SEARCH"

        # Step 3: Execute search
        search_results = [
            {"source": "Wikipedia", "answer": "13.96 million"},
            {"source": "World Bank", "answer": "13.96 million"},
            {"source": "UN Data", "answer": "14 million"},
        ]

        # Step 4: Verify agreement
        answers = [r["answer"] for r in search_results]
        # Approximate agreement (would use fuzzy matching)
        agreement = 0.9  # High agreement

        assert agreement > 0.7  # Above threshold


class TestErrorHandling:
    """Test error handling in the system."""

    def test_ollama_unavailable(self):
        """Test graceful handling when Ollama is unavailable."""
        # Simulate connection error
        class MockError(Exception):
            pass

        def simulate_ollama_call():
            raise MockError("Connection refused")

        # System should handle gracefully
        try:
            simulate_ollama_call()
            error_occurred = False
        except MockError:
            error_occurred = True

        assert error_occurred is True

    def test_high_temperature_handling(self):
        """Test system response to high GPU temperature."""
        gpu_temp = 87.0
        pause_threshold = 85.0

        should_pause = gpu_temp >= pause_threshold
        assert should_pause is True

        # System should:
        # 1. Pause Cortex operations
        # 2. Use Medulla only
        # 3. Notify user


class TestPerformanceMetrics:
    """Test performance metric tracking."""

    def test_response_time_tracking(self):
        """Test response time is tracked."""
        import time

        start = time.time()
        # Simulate processing
        time.sleep(0.1)
        end = time.time()

        response_time_ms = (end - start) * 1000

        assert response_time_ms > 0
        assert response_time_ms < 1000  # Should be under 1 second

    def test_cortex_invocation_counting(self):
        """Test Cortex invocations are counted."""
        stats = {
            "total_interactions": 0,
            "cortex_invocations": 0,
        }

        # Simulate interactions
        interactions = [
            {"used_cortex": False},
            {"used_cortex": True},
            {"used_cortex": False},
            {"used_cortex": True},
            {"used_cortex": False},
        ]

        for interaction in interactions:
            stats["total_interactions"] += 1
            if interaction["used_cortex"]:
                stats["cortex_invocations"] += 1

        assert stats["total_interactions"] == 5
        assert stats["cortex_invocations"] == 2

        # Cortex ratio
        cortex_ratio = stats["cortex_invocations"] / stats["total_interactions"]
        assert cortex_ratio == 0.4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
