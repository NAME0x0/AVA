#!/usr/bin/env python3
"""
Enhanced Calculator Tool for AVA
Production-Ready Mathematical Computation with Advanced Features
"""

import ast
import logging
import math
import operator
import re
import time
from dataclasses import dataclass, field
from decimal import getcontext
from enum import Enum
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set high precision for decimal calculations
getcontext().prec = 50


class CalculationType(Enum):
    """Types of calculations supported."""
    BASIC_ARITHMETIC = "basic"
    ADVANCED_MATH = "advanced"
    SCIENTIFIC = "scientific"
    STATISTICAL = "statistical"
    UNIT_CONVERSION = "conversion"


class NumberFormat(Enum):
    """Number format options."""
    DECIMAL = "decimal"
    SCIENTIFIC = "scientific"
    ENGINEERING = "engineering"
    PERCENTAGE = "percentage"
    FRACTION = "fraction"


@dataclass
class CalculationResult:
    """Represents the result of a calculation."""
    success: bool
    result: float | int | complex | str = None
    error: str | None = None
    expression: str = ""
    calculation_type: CalculationType = CalculationType.BASIC_ARITHMETIC
    precision_used: int = 15
    execution_time_ms: float = 0.0
    steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class SafeExpressionParser:
    """Safe mathematical expression parser using AST."""

    SAFE_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    SAFE_FUNCTIONS = {
        # Basic math functions
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,

        # Math module functions
        'sqrt': math.sqrt,
        'pow': pow,
        'exp': math.exp,
        'log': math.log,
        'log10': math.log10,
        'log2': math.log2,

        # Trigonometric functions
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'asin': math.asin,
        'acos': math.acos,
        'atan': math.atan,
        'atan2': math.atan2,
        'sinh': math.sinh,
        'cosh': math.cosh,
        'tanh': math.tanh,
        'asinh': math.asinh,
        'acosh': math.acosh,
        'atanh': math.atanh,

        # Other math functions
        'ceil': math.ceil,
        'floor': math.floor,
        'trunc': math.trunc,
        'factorial': math.factorial,
        'gcd': math.gcd,
        'lcm': math.lcm if hasattr(math, 'lcm') else lambda a, b: abs(a * b) // math.gcd(a, b),
        'degrees': math.degrees,
        'radians': math.radians,
    }

    SAFE_CONSTANTS = {
        'pi': math.pi,
        'e': math.e,
        'tau': math.tau,
        'inf': math.inf,
        'nan': math.nan,
    }

    def __init__(self):
        """Initialize the safe expression parser."""
        self.namespace = {**self.SAFE_FUNCTIONS, **self.SAFE_CONSTANTS}

    def parse_and_evaluate(self, expression: str) -> float | int | complex:
        """
        Safely parse and evaluate a mathematical expression.

        Args:
            expression: Mathematical expression string

        Returns:
            Numerical result of the expression

        Raises:
            ValueError: If expression contains unsafe elements
            Exception: For calculation errors
        """
        try:
            # Clean and preprocess expression
            cleaned_expr = self._preprocess_expression(expression)

            # Parse expression to AST
            tree = ast.parse(cleaned_expr, mode='eval')

            # Validate AST for safety
            self._validate_ast(tree)

            # Evaluate the expression
            result = self._evaluate_ast(tree.body)

            return result

        except Exception as e:
            raise ValueError(
                f"Error evaluating expression '{expression}': {str(e)}"
            ) from e

    def _preprocess_expression(self, expression: str) -> str:
        """Preprocess expression to handle common mathematical notation."""
        # Remove whitespace
        cleaned = re.sub(r'\s+', '', expression)

        # Replace common mathematical notation
        replacements = {
            '^': '**',  # Power operator
            '÷': '/',   # Division symbol
            '×': '*',   # Multiplication symbol
            '√': 'sqrt', # Square root symbol
        }

        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)

        # Handle implicit multiplication (e.g., 2(3+4) -> 2*(3+4))
        cleaned = re.sub(r'(\d)(\()', r'\1*\2', cleaned)
        cleaned = re.sub(r'(\))(\d)', r'\1*\2', cleaned)
        cleaned = re.sub(r'(\))(\()', r'\1*\2', cleaned)

        return cleaned

    def _validate_ast(self, node: ast.AST) -> None:
        """Validate AST node for safety."""
        if isinstance(node, ast.Expression):
            self._validate_ast(node.body)
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in self.SAFE_OPERATORS:
                raise ValueError(f"Unsafe operator: {type(node.op).__name__}")
            self._validate_ast(node.left)
            self._validate_ast(node.right)
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in self.SAFE_OPERATORS:
                raise ValueError(f"Unsafe unary operator: {type(node.op).__name__}")
            self._validate_ast(node.operand)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in self.SAFE_FUNCTIONS:
                raise ValueError(f"Unsafe function call: {getattr(node.func, 'id', 'unknown')}")
            for arg in node.args:
                self._validate_ast(arg)
        elif isinstance(node, ast.Name):
            if node.id not in self.namespace:
                raise ValueError(f"Undefined variable: {node.id}")
        elif isinstance(node, (ast.Constant, ast.Num)):  # ast.Num for older Python versions
            pass  # Numbers are safe
        else:
            raise ValueError(f"Unsafe AST node type: {type(node).__name__}")

    def _evaluate_ast(self, node: ast.AST) -> float | int | complex:
        """Evaluate validated AST node."""
        if isinstance(node, ast.BinOp):
            left = self._evaluate_ast(node.left)
            right = self._evaluate_ast(node.right)
            op_func = self.SAFE_OPERATORS[type(node.op)]
            return op_func(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._evaluate_ast(node.operand)
            op_func = self.SAFE_OPERATORS[type(node.op)]
            return op_func(operand)
        elif isinstance(node, ast.Call):
            func = self.SAFE_FUNCTIONS[node.func.id]
            args = [self._evaluate_ast(arg) for arg in node.args]
            return func(*args)
        elif isinstance(node, ast.Name):
            return self.namespace[node.id]
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # For older Python versions
            return node.n
        else:
            raise ValueError(f"Cannot evaluate AST node type: {type(node).__name__}")


class EnhancedCalculator:
    """Enhanced calculator tool with advanced mathematical capabilities."""

    def __init__(self, precision: int = 15):
        """Initialize the enhanced calculator."""
        self.name = "enhanced_calculator"
        self.description = """Advanced mathematical calculator supporting:
        - Basic arithmetic (+, -, *, /, %, //, **)
        - Advanced functions (sqrt, exp, log, trigonometric)
        - Scientific notation and precision control
        - Unit conversions and statistical operations
        - Safe expression parsing with AST validation"""

        self.precision = precision
        self.parser = SafeExpressionParser()

        # Tool parameters for function calling
        self.parameters = [
            {
                "name": "expression",
                "type": "string",
                "description": "Mathematical expression to evaluate. Supports basic arithmetic, advanced math functions, and constants.",
                "required": True
            },
            {
                "name": "format",
                "type": "string",
                "description": "Output format: 'decimal', 'scientific', 'engineering', 'percentage', 'fraction'",
                "default": "decimal"
            },
            {
                "name": "precision",
                "type": "integer",
                "description": "Number of decimal places for result (1-50)",
                "default": 15
            }
        ]

        # Unit conversion factors (to base units)
        self.unit_conversions = {
            # Length (to meters)
            'mm': 0.001, 'cm': 0.01, 'm': 1.0, 'km': 1000.0,
            'in': 0.0254, 'ft': 0.3048, 'yd': 0.9144, 'mi': 1609.344,

            # Weight (to grams)
            'mg': 0.001, 'g': 1.0, 'kg': 1000.0, 'lb': 453.592, 'oz': 28.3495,

            # Temperature conversions handled separately
            # Time (to seconds)
            'ms': 0.001, 's': 1.0, 'min': 60.0, 'h': 3600.0, 'day': 86400.0,
        }

        logger.info(f"Enhanced Calculator initialized with {precision} decimal precision")

    def calculate(
        self,
        expression: str,
        output_format: str = "decimal",
        precision: int | None = None
    ) -> CalculationResult:
        """
        Perform calculation with comprehensive error handling and formatting.

        Args:
            expression: Mathematical expression to evaluate
            output_format: Format for the result output
            precision: Decimal precision for the result

        Returns:
            CalculationResult object with result and metadata
        """
        start_time = time.time()
        precision = precision or self.precision

        result = CalculationResult(
            success=False,
            expression=expression,
            precision_used=precision
        )

        try:
            # Validate input
            if not expression or not isinstance(expression, str):
                raise ValueError("Expression must be a non-empty string")

            if len(expression) > 1000:
                raise ValueError("Expression too long (max 1000 characters)")

            # Determine calculation type
            result.calculation_type = self._determine_calculation_type(expression)

            # Parse and evaluate expression
            logger.debug(f"Evaluating expression: {expression}")
            raw_result = self.parser.parse_and_evaluate(expression)

            # Handle special cases
            if math.isnan(raw_result):
                raise ValueError("Result is not a number (NaN)")
            elif math.isinf(raw_result):
                result.warnings.append("Result is infinite")

            # Format result
            formatted_result = self._format_result(
                raw_result,
                NumberFormat(output_format),
                precision
            )

            # Success
            result.success = True
            result.result = formatted_result
            result.metadata = {
                "raw_result": raw_result,
                "result_type": type(raw_result).__name__,
                "is_integer": isinstance(raw_result, int) or (isinstance(raw_result, float) and raw_result.is_integer()),
                "absolute_value": abs(raw_result) if isinstance(raw_result, (int, float)) else None
            }

            logger.info(f"Calculation successful: {expression} = {formatted_result}")

        except Exception as e:
            result.error = str(e)
            logger.error(f"Calculation failed for '{expression}': {result.error}")

        finally:
            result.execution_time_ms = (time.time() - start_time) * 1000

        return result

    def _determine_calculation_type(self, expression: str) -> CalculationType:
        """Determine the type of calculation based on the expression."""
        expression_lower = expression.lower()

        if any(func in expression_lower for func in ['sin', 'cos', 'tan', 'log', 'exp', 'sqrt']):
            return CalculationType.SCIENTIFIC
        elif any(func in expression_lower for func in ['mean', 'median', 'std', 'var']):
            return CalculationType.STATISTICAL
        elif 'convert' in expression_lower or any(unit in expression_lower for unit in self.unit_conversions):
            return CalculationType.UNIT_CONVERSION
        elif any(func in expression_lower for func in ['factorial', 'gcd', 'lcm']):
            return CalculationType.ADVANCED_MATH
        else:
            return CalculationType.BASIC_ARITHMETIC

    def _format_result(
        self,
        value: float | int | complex,
        format_type: NumberFormat,
        precision: int
    ) -> str:
        """Format the calculation result according to specified format."""
        try:
            if isinstance(value, complex):
                return f"{value.real:.{precision}f} + {value.imag:.{precision}f}i"

            if format_type == NumberFormat.DECIMAL:
                if isinstance(value, int):
                    return str(value)
                return f"{value:.{precision}f}".rstrip('0').rstrip('.')

            elif format_type == NumberFormat.SCIENTIFIC:
                return f"{value:.{precision}e}"

            elif format_type == NumberFormat.ENGINEERING:
                # Engineering notation (powers of 3)
                if value == 0:
                    return "0"
                exp = math.floor(math.log10(abs(value)))
                eng_exp = exp - (exp % 3)
                mantissa = value / (10 ** eng_exp)
                return f"{mantissa:.{precision}f}e{eng_exp:+d}"

            elif format_type == NumberFormat.PERCENTAGE:
                return f"{value * 100:.{precision}f}%"

            elif format_type == NumberFormat.FRACTION:
                return self._to_fraction(value, precision)

            else:
                return str(value)

        except Exception:
            return str(value)

    def _to_fraction(self, value: float, max_denominator: int = 1000000) -> str:
        """Convert decimal to fraction representation."""
        try:
            from fractions import Fraction
            frac = Fraction(value).limit_denominator(max_denominator)
            return str(frac)
        except ImportError:
            # Fallback if fractions module not available
            return str(value)

    def convert_units(self, value: float, from_unit: str, to_unit: str) -> CalculationResult:
        """Convert between units."""
        result = CalculationResult(success=False, calculation_type=CalculationType.UNIT_CONVERSION)

        try:
            from_unit = from_unit.lower()
            to_unit = to_unit.lower()

            # Special case for temperature
            if from_unit in ['c', 'f', 'k'] or to_unit in ['c', 'f', 'k']:
                converted = self._convert_temperature(value, from_unit, to_unit)
            else:
                # Regular unit conversion
                if from_unit not in self.unit_conversions or to_unit not in self.unit_conversions:
                    raise ValueError(f"Unsupported unit conversion: {from_unit} to {to_unit}")

                # Convert to base unit, then to target unit
                base_value = value * self.unit_conversions[from_unit]
                converted = base_value / self.unit_conversions[to_unit]

            result.success = True
            result.result = converted
            result.expression = f"{value} {from_unit} to {to_unit}"
            result.metadata = {"conversion_factor": converted / value if value != 0 else 0}

        except Exception as e:
            result.error = str(e)

        return result

    def _convert_temperature(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert temperature between Celsius, Fahrenheit, and Kelvin."""
        # Convert to Celsius first
        if from_unit == 'f':
            celsius = (value - 32) * 5/9
        elif from_unit == 'k':
            celsius = value - 273.15
        else:  # from_unit == 'c'
            celsius = value

        # Convert from Celsius to target
        if to_unit == 'f':
            return celsius * 9/5 + 32
        elif to_unit == 'k':
            return celsius + 273.15
        else:  # to_unit == 'c'
            return celsius

    def run(self, expression: str, **kwargs) -> dict[str, Any]:
        """
        Main interface for function calling compatibility.

        Args:
            expression: Mathematical expression to evaluate
            **kwargs: Additional parameters (format, precision)

        Returns:
            Dictionary with result or error information
        """
        output_format = kwargs.get('format', 'decimal')
        precision = kwargs.get('precision', self.precision)

        result = self.calculate(expression, output_format, precision)

        if result.success:
            return {
                "result": result.result,
                "expression": result.expression,
                "calculation_type": result.calculation_type.value,
                "execution_time_ms": result.execution_time_ms,
                "precision": result.precision_used,
                "warnings": result.warnings,
                "metadata": result.metadata
            }
        else:
            return {
                "error": result.error,
                "expression": result.expression,
                "execution_time_ms": result.execution_time_ms
            }


def test_calculator():
    """Test the enhanced calculator with various expressions."""
    logger.info("=== Testing Enhanced Calculator ===")

    calc = EnhancedCalculator()

    test_expressions = [
        # Basic arithmetic
        "2 + 2",
        "10 - 5.5",
        "3 * 7",
        "10 / 2",
        "2**8",  # Power
        "17 % 5",  # Modulo
        "17 // 5",  # Floor division

        # Advanced math
        "sqrt(16)",
        "sin(pi/2)",
        "log(e)",
        "factorial(5)",
        "gcd(48, 18)",

        # Complex expressions
        "(2 + 3) * 4",
        "sqrt(sin(pi/4)**2 + cos(pi/4)**2)",
        "log10(1000)",
        "degrees(pi/2)",

        # Error cases
        "10 / 0",  # Division by zero
        "sqrt(-4)",  # Domain error
        "log(-1)",  # Domain error
        "factorial(-1)",  # Domain error
    ]

    for expr in test_expressions:
        logger.info(f"\nTesting: {expr}")
        result = calc.run(expr)
        if "error" in result:
            logger.error(f"Error: {result['error']}")
        else:
            logger.info(f"Result: {result['result']}")
            logger.info(f"Type: {result['calculation_type']}")
            logger.info(f"Time: {result['execution_time_ms']:.2f}ms")

    # Test unit conversion
    logger.info("\n=== Testing Unit Conversion ===")
    conversion_result = calc.convert_units(100, "cm", "m")
    if conversion_result.success:
        logger.info(f"100 cm = {conversion_result.result} m")
    else:
        logger.error(f"Conversion error: {conversion_result.error}")

    # Test temperature conversion
    temp_result = calc.convert_units(100, "c", "f")
    if temp_result.success:
        logger.info(f"100°C = {temp_result.result}°F")
    else:
        logger.error(f"Temperature conversion error: {temp_result.error}")


def main():
    """Main function for standalone testing."""
    test_calculator()


if __name__ == "__main__":
    main()
