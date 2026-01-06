"""Tests for the Sentinel architecture formal verification module."""

import pytest

# Import the verification module
from src.core.verification import (
    SYMPY_AVAILABLE,
    Z3_AVAILABLE,
    FormalVerifier,
    VerificationConfig,
    VerificationResult,
    VerificationStatus,
    VerificationType,
)


class TestVerificationConfig:
    """Tests for VerificationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = VerificationConfig()
        assert config.timeout_ms == 5000
        assert config.max_depth == 10
        assert config.entropy_threshold == 0.7

    def test_custom_config(self):
        """Test custom configuration."""
        config = VerificationConfig(
            timeout_ms=10000,
            max_depth=20,
            entropy_threshold=0.5,
        )
        assert config.timeout_ms == 10000
        assert config.max_depth == 20
        assert config.entropy_threshold == 0.5


class TestVerificationResult:
    """Tests for VerificationResult."""

    def test_valid_result(self):
        """Test creating a valid result."""
        result = VerificationResult(
            status=VerificationStatus.VALID,
            is_valid=True,
            verification_type=VerificationType.LOGIC,
            explanation="All constraints satisfied",
        )
        assert result.is_valid is True
        assert result.status == VerificationStatus.VALID
        assert "satisfied" in result.explanation

    def test_invalid_result(self):
        """Test creating an invalid result."""
        result = VerificationResult(
            status=VerificationStatus.INVALID,
            is_valid=False,
            verification_type=VerificationType.CONSTRAINT,
            explanation="Contradiction found",
        )
        assert result.is_valid is False
        assert result.status == VerificationStatus.INVALID


class TestFormalVerifier:
    """Tests for FormalVerifier class."""

    def test_verifier_creation(self):
        """Test creating a FormalVerifier instance."""
        config = VerificationConfig()
        verifier = FormalVerifier(config)
        assert verifier is not None
        assert verifier.config == config

    def test_verifier_default_creation(self):
        """Test creating a FormalVerifier with defaults."""
        verifier = FormalVerifier()
        assert verifier is not None
        assert verifier.config.timeout_ms == 5000

    @pytest.mark.asyncio
    async def test_verify_simple_arithmetic(self):
        """Test verifying simple arithmetic claim."""
        verifier = FormalVerifier()
        result = await verifier.verify_logic("2 + 2 = 4", "")
        # Should return a result (valid or skipped if Z3 not available)
        assert result is not None
        assert isinstance(result, VerificationResult)

    @pytest.mark.asyncio
    async def test_verify_code_simple(self):
        """Test verifying simple code."""
        verifier = FormalVerifier()
        code = "def add(a, b): return a + b"
        constraints = ["result == a + b"]
        result = await verifier.verify_code(code, constraints)
        # Basic code should return a result
        assert result is not None
        assert isinstance(result, VerificationResult)

    def test_should_verify_code_block(self):
        """Test should_verify detects code blocks."""
        verifier = FormalVerifier()

        # Code block should trigger verification
        text = "```python\ndef foo(): pass\n```"
        assert verifier.should_verify(text, 0.5) is True

    def test_should_verify_high_entropy(self):
        """Test should_verify on high entropy."""
        verifier = FormalVerifier()

        # High entropy should trigger verification
        text = "Some uncertain text"
        assert verifier.should_verify(text, 0.9) is True  # entropy > 0.7

    def test_should_verify_logic_indicators(self):
        """Test should_verify detects logic indicators."""
        verifier = FormalVerifier()

        # Logic indicators should trigger verification
        text = "Prove that P implies Q"
        assert verifier.should_verify(text, 0.3) is True

    def test_should_not_verify_simple_text(self):
        """Test should_verify on simple text."""
        verifier = FormalVerifier()

        # Simple text with low entropy should not trigger
        text = "Hello, how are you?"
        assert verifier.should_verify(text, 0.2) is False


class TestZ3Integration:
    """Tests for Z3 integration (skipped if Z3 not available)."""

    @pytest.mark.skipif(not Z3_AVAILABLE, reason="Z3 not installed")
    def test_z3_available(self):
        """Test that Z3 is available."""
        assert Z3_AVAILABLE is True

    @pytest.mark.skipif(not Z3_AVAILABLE, reason="Z3 not installed")
    @pytest.mark.asyncio
    async def test_z3_simple_sat(self):
        """Test Z3 satisfiability check."""
        verifier = FormalVerifier()
        # Simple satisfiable constraint
        result = await verifier.verify_logic("x > 0 and x < 10", "")
        assert result.status != VerificationStatus.UNKNOWN

    @pytest.mark.skipif(not Z3_AVAILABLE, reason="Z3 not installed")
    @pytest.mark.asyncio
    async def test_z3_unsat(self):
        """Test Z3 unsatisfiability detection."""
        verifier = FormalVerifier()
        # Unsatisfiable constraint
        result = await verifier.verify_logic("x > 10 and x < 5", "")
        assert result.status == VerificationStatus.INVALID


class TestSymPyIntegration:
    """Tests for SymPy integration (skipped if SymPy not available)."""

    @pytest.mark.skipif(not SYMPY_AVAILABLE, reason="SymPy not installed")
    def test_sympy_available(self):
        """Test that SymPy is available."""
        assert SYMPY_AVAILABLE is True
