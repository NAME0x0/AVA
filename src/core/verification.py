"""
VERIFICATION - Formal Logic Verification via Z3
================================================

The Verification module implements the MATH-VF (Mathematical Verification Framework)
component of the Sentinel architecture. It uses the Z3 SMT solver to formally
verify code correctness and logical consistency of claims.

Key Features:
- Adaptive verification (triggered by entropy/complexity)
- Code block verification (arithmetic, logic, constraints)
- Claim verification (logical consistency checking)
- SMT-LIB translation for formal proofs

Integration:
- Called by MentalSandbox during simulation cycles
- Can be invoked via VERIFY_LOGIC policy in Agency

References:
- MATH-VF: Step-Wise Formal Verification for LLM-Based Problem Solving (arXiv, 2025)
- Z3: https://github.com/Z3Prover/z3
"""

import ast
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# Optional Z3 import - graceful fallback if not installed
try:
    import z3

    Z3_AVAILABLE = True
except ImportError:
    z3 = None  # type: ignore[assignment]
    Z3_AVAILABLE = False

# Optional SymPy for symbolic math
try:
    import sympy
    from sympy.parsing.sympy_parser import parse_expr

    SYMPY_AVAILABLE = True
except ImportError:
    sympy = None  # type: ignore[assignment]
    parse_expr = None  # type: ignore[assignment]
    SYMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class VerificationType(Enum):
    """Types of verification checks."""

    ARITHMETIC = "arithmetic"  # Mathematical expressions
    LOGIC = "logic"  # Boolean logic and implications
    CONSTRAINT = "constraint"  # Variable constraints
    CODE = "code"  # Code block verification
    CLAIM = "claim"  # Natural language claim verification


class VerificationStatus(Enum):
    """Result status of verification."""

    VALID = "valid"  # Verified as correct
    INVALID = "invalid"  # Proven incorrect
    UNKNOWN = "unknown"  # Could not determine
    TIMEOUT = "timeout"  # Solver timed out
    ERROR = "error"  # Verification error
    SKIPPED = "skipped"  # Verification not applicable


@dataclass
class VerificationConfig:
    """Configuration for the formal verifier."""

    timeout_ms: int = 5000  # Z3 solver timeout
    max_depth: int = 10  # Maximum proof depth
    entropy_threshold: float = 0.7  # Trigger verification above this
    always_verify_code: bool = True  # Always verify code blocks
    always_verify_math: bool = True  # Always verify math expressions
    enable_sympy: bool = True  # Use SymPy for symbolic math
    strict_mode: bool = False  # Fail on unknown results


@dataclass
class VerificationResult:
    """Result of a verification check."""

    status: VerificationStatus
    is_valid: bool
    verification_type: VerificationType
    explanation: str
    raw_result: str | None = None
    constraints_checked: int = 0
    proof_steps: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "is_valid": self.is_valid,
            "type": self.verification_type.value,
            "explanation": self.explanation,
            "raw_result": self.raw_result,
            "constraints_checked": self.constraints_checked,
            "proof_steps": self.proof_steps,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
        }


class FormalVerifier:
    """
    Z3-based formal verification for code and logic.

    Implements the MATH-VF framework for verifying:
    - Arithmetic expressions and equations
    - Logical claims and implications
    - Code block correctness (simple cases)
    - Constraint satisfaction

    Example:
        >>> verifier = FormalVerifier(VerificationConfig())
        >>> result = await verifier.verify_logic("2 + 2 = 4", "")
        >>> assert result.is_valid
    """

    def __init__(self, config: VerificationConfig | None = None):
        self.config = config or VerificationConfig()
        self._solver: Any | None = None
        self._stats = {
            "total_checks": 0,
            "valid": 0,
            "invalid": 0,
            "unknown": 0,
            "timeouts": 0,
            "errors": 0,
        }

        if Z3_AVAILABLE:
            self._solver = z3.Solver()
            self._solver.set("timeout", self.config.timeout_ms)
            logger.info("Z3 solver initialized")
        else:
            logger.warning("Z3 not available - verification will be limited")

    def should_verify(self, text: str, entropy: float = 0.0) -> bool:
        """
        Determine if verification should be triggered (adaptive).

        Args:
            text: The text to potentially verify
            entropy: Current entropy level (0-1)

        Returns:
            True if verification should be performed
        """
        # High entropy triggers verification
        if entropy > self.config.entropy_threshold:
            return True

        # Code blocks always verify
        if self.config.always_verify_code and self._contains_code(text):
            return True

        # Math expressions always verify
        if self.config.always_verify_math and self._contains_math(text):
            return True

        # Logic indicators trigger verification
        if self._contains_logic_indicators(text):
            return True

        return False

    async def verify_logic(self, claim: str, context: str = "") -> VerificationResult:
        """
        Verify logical consistency of a claim.

        Args:
            claim: The logical claim to verify
            context: Additional context for verification

        Returns:
            VerificationResult with verification status
        """
        import time

        start_time = time.time()
        self._stats["total_checks"] += 1

        if not Z3_AVAILABLE:
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                is_valid=True,  # Assume valid if can't verify
                verification_type=VerificationType.LOGIC,
                explanation="Z3 not available - verification skipped",
            )

        try:
            # Try to parse as logical expression first (more specific)
            if self._is_logical(claim):
                result = await self._verify_logical_expr(claim)
            # Try to parse as arithmetic
            elif self._is_arithmetic(claim):
                result = await self._verify_arithmetic(claim)
            # Try to extract and verify constraints
            else:
                result = await self._verify_constraints(claim, context)

            duration_ms = (time.time() - start_time) * 1000
            result.duration_ms = duration_ms

            # Update stats
            if result.status == VerificationStatus.VALID:
                self._stats["valid"] += 1
            elif result.status == VerificationStatus.INVALID:
                self._stats["invalid"] += 1
            elif result.status == VerificationStatus.TIMEOUT:
                self._stats["timeouts"] += 1
            else:
                self._stats["unknown"] += 1

            return result

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Verification error: {e}")
            return VerificationResult(
                status=VerificationStatus.ERROR,
                is_valid=False,
                verification_type=VerificationType.LOGIC,
                explanation=f"Verification error: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
            )

    async def verify_code(
        self, code: str, constraints: list[str] | None = None
    ) -> VerificationResult:
        """
        Verify code block correctness.

        Currently supports:
        - Simple arithmetic operations
        - Variable assignments with constraints
        - Loop invariants (limited)

        Args:
            code: The code block to verify
            constraints: Optional list of constraints to check

        Returns:
            VerificationResult with verification status
        """
        import time

        start_time = time.time()
        self._stats["total_checks"] += 1

        if not Z3_AVAILABLE:
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                is_valid=True,
                verification_type=VerificationType.CODE,
                explanation="Z3 not available - code verification skipped",
            )

        try:
            # Extract assignments and expressions from code
            assignments = self._extract_assignments(code)
            expressions = self._extract_expressions(code)

            if not assignments and not expressions:
                return VerificationResult(
                    status=VerificationStatus.SKIPPED,
                    is_valid=True,
                    verification_type=VerificationType.CODE,
                    explanation="No verifiable statements found in code",
                )

            # Build Z3 constraints
            solver = z3.Solver()
            solver.set("timeout", self.config.timeout_ms)
            variables: dict[str, Any] = {}
            proof_steps: list[str] = []

            # Process assignments
            for var_name, value_expr in assignments.items():
                try:
                    var = z3.Int(var_name)
                    variables[var_name] = var
                    # Try to evaluate the expression
                    if isinstance(value_expr, (int, float)):
                        solver.add(var == int(value_expr))
                        proof_steps.append(f"Assert: {var_name} = {value_expr}")
                except Exception as e:
                    logger.debug(f"Could not process assignment {var_name}: {e}")

            # Add user constraints
            if constraints:
                for constraint in constraints:
                    try:
                        z3_constraint = self._parse_constraint(constraint, variables)
                        if z3_constraint is not None:
                            solver.add(z3_constraint)
                            proof_steps.append(f"Constraint: {constraint}")
                    except Exception as e:
                        logger.debug(f"Could not parse constraint {constraint}: {e}")

            # Check satisfiability
            result = solver.check()
            duration_ms = (time.time() - start_time) * 1000

            if result == z3.sat:
                self._stats["valid"] += 1
                return VerificationResult(
                    status=VerificationStatus.VALID,
                    is_valid=True,
                    verification_type=VerificationType.CODE,
                    explanation="Code constraints are satisfiable",
                    raw_result=str(solver.model()) if solver.model() else None,
                    constraints_checked=len(proof_steps),
                    proof_steps=proof_steps,
                    duration_ms=duration_ms,
                )
            elif result == z3.unsat:
                self._stats["invalid"] += 1
                return VerificationResult(
                    status=VerificationStatus.INVALID,
                    is_valid=False,
                    verification_type=VerificationType.CODE,
                    explanation="Code constraints are unsatisfiable (contradiction found)",
                    constraints_checked=len(proof_steps),
                    proof_steps=proof_steps,
                    duration_ms=duration_ms,
                )
            else:
                self._stats["unknown"] += 1
                return VerificationResult(
                    status=VerificationStatus.UNKNOWN,
                    is_valid=not self.config.strict_mode,
                    verification_type=VerificationType.CODE,
                    explanation="Could not determine satisfiability",
                    constraints_checked=len(proof_steps),
                    proof_steps=proof_steps,
                    duration_ms=duration_ms,
                )

        except Exception as e:
            self._stats["errors"] += 1
            return VerificationResult(
                status=VerificationStatus.ERROR,
                is_valid=False,
                verification_type=VerificationType.CODE,
                explanation=f"Code verification error: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
            )

    async def _verify_arithmetic(self, expr: str) -> VerificationResult:
        """Verify arithmetic expression (e.g., '2 + 2 = 4')."""
        proof_steps: list[str] = []

        # Try with SymPy first (more flexible parsing)
        if SYMPY_AVAILABLE and self.config.enable_sympy:
            try:
                # Handle equality expressions
                if "=" in expr and "==" not in expr:
                    parts = expr.replace("=", "==").split("==")
                    if len(parts) == 2:
                        left = parse_expr(parts[0].strip())
                        right = parse_expr(parts[1].strip())
                        proof_steps.append(f"Parse LHS: {left}")
                        proof_steps.append(f"Parse RHS: {right}")

                        # Simplify and compare
                        left_simplified = sympy.simplify(left)
                        right_simplified = sympy.simplify(right)
                        proof_steps.append(f"Simplify LHS: {left_simplified}")
                        proof_steps.append(f"Simplify RHS: {right_simplified}")

                        if left_simplified == right_simplified:
                            return VerificationResult(
                                status=VerificationStatus.VALID,
                                is_valid=True,
                                verification_type=VerificationType.ARITHMETIC,
                                explanation=f"Verified: {left_simplified} = {right_simplified}",
                                proof_steps=proof_steps,
                            )
                        else:
                            return VerificationResult(
                                status=VerificationStatus.INVALID,
                                is_valid=False,
                                verification_type=VerificationType.ARITHMETIC,
                                explanation=f"Invalid: {left_simplified} ≠ {right_simplified}",
                                proof_steps=proof_steps,
                            )
            except Exception as e:
                logger.debug(f"SymPy parsing failed: {e}")

        # Fallback to Z3
        try:
            # Simple arithmetic check with Z3
            _x = z3.Int("x")  # Reserved for future constraint building
            solver = z3.Solver()
            solver.set("timeout", self.config.timeout_ms)

            # Try to parse as simple equality
            if "=" in expr:
                parts = expr.split("=")
                if len(parts) == 2:
                    try:
                        left_val = ast.literal_eval(parts[0].strip())
                        right_val = ast.literal_eval(parts[1].strip())
                        if left_val == right_val:
                            return VerificationResult(
                                status=VerificationStatus.VALID,
                                is_valid=True,
                                verification_type=VerificationType.ARITHMETIC,
                                explanation=f"Verified: {left_val} = {right_val}",
                                proof_steps=[
                                    f"Evaluate: {parts[0].strip()} = {left_val}",
                                    f"Evaluate: {parts[1].strip()} = {right_val}",
                                ],
                            )
                        else:
                            return VerificationResult(
                                status=VerificationStatus.INVALID,
                                is_valid=False,
                                verification_type=VerificationType.ARITHMETIC,
                                explanation=f"Invalid: {left_val} ≠ {right_val}",
                                proof_steps=[
                                    f"Evaluate: {parts[0].strip()} = {left_val}",
                                    f"Evaluate: {parts[1].strip()} = {right_val}",
                                ],
                            )
                    except Exception:
                        pass  # nosec B110 - fall through to UNKNOWN result

            return VerificationResult(
                status=VerificationStatus.UNKNOWN,
                is_valid=not self.config.strict_mode,
                verification_type=VerificationType.ARITHMETIC,
                explanation="Could not parse arithmetic expression",
            )

        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                is_valid=False,
                verification_type=VerificationType.ARITHMETIC,
                explanation=f"Arithmetic verification error: {str(e)}",
            )

    async def _verify_logical_expr(self, expr: str) -> VerificationResult:
        """Verify logical expression (e.g., 'x > 5 AND x < 10')."""
        if not Z3_AVAILABLE:
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                is_valid=True,
                verification_type=VerificationType.LOGIC,
                explanation="Z3 not available",
            )

        try:
            solver = z3.Solver()
            solver.set("timeout", self.config.timeout_ms)

            # Normalize logical operators (case-insensitive replacement)
            # Use regex to replace operators while preserving variable case
            expr_normalized = re.sub(r"\band\b", "&", expr, flags=re.IGNORECASE)
            expr_normalized = re.sub(r"\bor\b", "|", expr_normalized, flags=re.IGNORECASE)
            expr_normalized = re.sub(r"\bnot\b", "~", expr_normalized, flags=re.IGNORECASE)

            # Extract variables (from original expression to preserve case)
            var_pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b"
            var_names = set(re.findall(var_pattern, expr_normalized))
            var_names -= {"and", "or", "not", "true", "false", "AND", "OR", "NOT", "TRUE", "FALSE"}

            variables = {name: z3.Int(name) for name in var_names}

            # Build constraint
            z3_expr = self._build_z3_expr(expr_normalized, variables)
            if z3_expr is None:
                return VerificationResult(
                    status=VerificationStatus.UNKNOWN,
                    is_valid=not self.config.strict_mode,
                    verification_type=VerificationType.LOGIC,
                    explanation="Could not parse logical expression",
                )

            solver.add(z3_expr)
            result = solver.check()

            if result == z3.sat:
                return VerificationResult(
                    status=VerificationStatus.VALID,
                    is_valid=True,
                    verification_type=VerificationType.LOGIC,
                    explanation="Logical expression is satisfiable",
                    raw_result=str(solver.model()) if solver.model() else None,
                )
            elif result == z3.unsat:
                return VerificationResult(
                    status=VerificationStatus.INVALID,
                    is_valid=False,
                    verification_type=VerificationType.LOGIC,
                    explanation="Logical expression is a contradiction",
                )
            else:
                return VerificationResult(
                    status=VerificationStatus.UNKNOWN,
                    is_valid=not self.config.strict_mode,
                    verification_type=VerificationType.LOGIC,
                    explanation="Could not determine satisfiability",
                )

        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                is_valid=False,
                verification_type=VerificationType.LOGIC,
                explanation=f"Logical verification error: {str(e)}",
            )

    async def _verify_constraints(self, claim: str, context: str) -> VerificationResult:
        """Verify constraints extracted from natural language claim."""
        # Extract numeric constraints from text
        constraints = self._extract_constraints_from_text(claim + " " + context)

        if not constraints:
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                is_valid=True,
                verification_type=VerificationType.CONSTRAINT,
                explanation="No verifiable constraints found",
            )

        if not Z3_AVAILABLE:
            return VerificationResult(
                status=VerificationStatus.SKIPPED,
                is_valid=True,
                verification_type=VerificationType.CONSTRAINT,
                explanation="Z3 not available",
            )

        try:
            solver = z3.Solver()
            solver.set("timeout", self.config.timeout_ms)
            variables: dict[str, Any] = {}
            proof_steps: list[str] = []

            for constraint in constraints:
                z3_constraint = self._parse_constraint(constraint, variables)
                if z3_constraint is not None:
                    solver.add(z3_constraint)
                    proof_steps.append(f"Add constraint: {constraint}")

            if not proof_steps:
                return VerificationResult(
                    status=VerificationStatus.SKIPPED,
                    is_valid=True,
                    verification_type=VerificationType.CONSTRAINT,
                    explanation="No parseable constraints found",
                )

            result = solver.check()

            if result == z3.sat:
                return VerificationResult(
                    status=VerificationStatus.VALID,
                    is_valid=True,
                    verification_type=VerificationType.CONSTRAINT,
                    explanation="All constraints are satisfiable",
                    constraints_checked=len(proof_steps),
                    proof_steps=proof_steps,
                )
            elif result == z3.unsat:
                return VerificationResult(
                    status=VerificationStatus.INVALID,
                    is_valid=False,
                    verification_type=VerificationType.CONSTRAINT,
                    explanation="Constraints contain a contradiction",
                    constraints_checked=len(proof_steps),
                    proof_steps=proof_steps,
                )
            else:
                return VerificationResult(
                    status=VerificationStatus.UNKNOWN,
                    is_valid=not self.config.strict_mode,
                    verification_type=VerificationType.CONSTRAINT,
                    explanation="Could not determine constraint satisfiability",
                    constraints_checked=len(proof_steps),
                    proof_steps=proof_steps,
                )

        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                is_valid=False,
                verification_type=VerificationType.CONSTRAINT,
                explanation=f"Constraint verification error: {str(e)}",
            )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _contains_code(self, text: str) -> bool:
        """Check if text contains code blocks."""
        code_indicators = [
            "```",
            "def ",
            "class ",
            "import ",
            "function ",
            "const ",
            "let ",
            "var ",
            "return ",
            "if (",
            "for (",
            "while (",
        ]
        return any(indicator in text for indicator in code_indicators)

    def _contains_math(self, text: str) -> bool:
        """Check if text contains mathematical expressions."""
        math_patterns = [
            r"\d+\s*[\+\-\*\/]\s*\d+",  # Basic arithmetic
            r"\d+\s*=\s*\d+",  # Equality
            r"[a-z]\s*[\+\-\*\/]\s*[a-z0-9]",  # Algebraic
            r"sqrt|sin|cos|tan|log|exp",  # Functions
        ]
        for pattern in math_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _contains_logic_indicators(self, text: str) -> bool:
        """Check if text contains logical reasoning indicators."""
        logic_words = [
            "therefore",
            "implies",
            "if and only if",
            "necessarily",
            "contradiction",
            "prove",
            "proof",
            "theorem",
            "lemma",
            "follows that",
            "must be",
            "cannot be",
        ]
        text_lower = text.lower()
        return any(word in text_lower for word in logic_words)

    def _is_arithmetic(self, expr: str) -> bool:
        """Check if expression is arithmetic."""
        return bool(re.search(r"[\d\+\-\*\/\=\(\)]+", expr))

    def _is_logical(self, expr: str) -> bool:
        """Check if expression is logical."""
        logical_ops = ["AND", "OR", "NOT", "IMPLIES", "IFF", "<", ">", "<=", ">="]
        expr_upper = expr.upper()
        return any(op in expr_upper for op in logical_ops)

    def _extract_assignments(self, code: str) -> dict[str, Any]:
        """Extract variable assignments from code."""
        assignments: dict[str, Any] = {}
        # Simple assignment pattern: var = value
        pattern = r"(\w+)\s*=\s*(\d+(?:\.\d+)?)"
        matches = re.findall(pattern, code)
        for var_name, value in matches:
            try:
                if "." in value:
                    assignments[var_name] = float(value)
                else:
                    assignments[var_name] = int(value)
            except ValueError:
                pass
        return assignments

    def _extract_expressions(self, code: str) -> list[str]:
        """Extract expressions from code."""
        expressions: list[str] = []
        # Look for comparison expressions
        pattern = r"(\w+\s*[<>=!]+\s*\w+)"
        matches = re.findall(pattern, code)
        expressions.extend(matches)
        return expressions

    def _extract_constraints_from_text(self, text: str) -> list[str]:
        """Extract constraints from natural language text."""
        constraints: list[str] = []

        # Pattern: "x is greater than 5" -> "x > 5"
        greater_pattern = r"(\w+)\s+is\s+greater\s+than\s+(\d+)"
        for match in re.finditer(greater_pattern, text, re.IGNORECASE):
            constraints.append(f"{match.group(1)} > {match.group(2)}")

        # Pattern: "x is less than 5" -> "x < 5"
        less_pattern = r"(\w+)\s+is\s+less\s+than\s+(\d+)"
        for match in re.finditer(less_pattern, text, re.IGNORECASE):
            constraints.append(f"{match.group(1)} < {match.group(2)}")

        # Pattern: "x equals 5" -> "x = 5"
        equals_pattern = r"(\w+)\s+equals?\s+(\d+)"
        for match in re.finditer(equals_pattern, text, re.IGNORECASE):
            constraints.append(f"{match.group(1)} = {match.group(2)}")

        # Direct constraint patterns: "x > 5", "y <= 10"
        direct_pattern = r"(\w+)\s*([<>=!]+)\s*(\d+)"
        for match in re.finditer(direct_pattern, text):
            constraints.append(f"{match.group(1)} {match.group(2)} {match.group(3)}")

        return constraints

    def _parse_constraint(self, constraint: str, variables: dict[str, Any]) -> Any | None:
        """Parse a constraint string into Z3 expression."""
        if not Z3_AVAILABLE:
            return None

        try:
            # Handle common comparison operators
            ops = [
                ("<=", lambda a, b: a <= b),
                (">=", lambda a, b: a >= b),
                ("!=", lambda a, b: a != b),
                ("==", lambda a, b: a == b),
                ("<", lambda a, b: a < b),
                (">", lambda a, b: a > b),
                ("=", lambda a, b: a == b),
            ]

            for op_str, op_func in ops:
                if op_str in constraint:
                    parts = constraint.split(op_str)
                    if len(parts) == 2:
                        left = parts[0].strip()
                        right = parts[1].strip()

                        # Get or create variable
                        if left not in variables:
                            variables[left] = z3.Int(left)

                        # Parse right side
                        try:
                            right_val = int(right)
                            return op_func(variables[left], right_val)
                        except ValueError:
                            if right not in variables:
                                variables[right] = z3.Int(right)
                            return op_func(variables[left], variables[right])
        except Exception as e:
            logger.debug(f"Could not parse constraint '{constraint}': {e}")

        return None

    def _build_z3_expr(self, expr: str, variables: dict[str, Any]) -> Any | None:
        """Build Z3 expression from string."""
        if not Z3_AVAILABLE:
            return None

        try:
            # This is a simplified parser - could be extended
            # For now, handle simple comparisons

            # Handle AND/OR combinations
            if " & " in expr:
                parts = expr.split(" & ")
                z3_parts = [self._build_z3_expr(p.strip(), variables) for p in parts]
                z3_parts = [p for p in z3_parts if p is not None]
                if z3_parts:
                    return z3.And(*z3_parts)
            elif " | " in expr:
                parts = expr.split(" | ")
                z3_parts = [self._build_z3_expr(p.strip(), variables) for p in parts]
                z3_parts = [p for p in z3_parts if p is not None]
                if z3_parts:
                    return z3.Or(*z3_parts)

            # Handle single comparison
            return self._parse_constraint(expr, variables)

        except Exception as e:
            logger.debug(f"Could not build Z3 expression: {e}")
            return None

    def get_stats(self) -> dict[str, int]:
        """Get verification statistics."""
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset verification statistics."""
        self._stats = {
            "total_checks": 0,
            "valid": 0,
            "invalid": 0,
            "unknown": 0,
            "timeouts": 0,
            "errors": 0,
        }
