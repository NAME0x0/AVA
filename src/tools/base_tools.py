"""
Base Tools for AVA

Implements tools at each safety level, from baby-safe to mature.
"""

import math
import random
from datetime import datetime
from typing import Any, Dict, Optional

from .registry import ToolRegistry, ToolSafetyLevel


# =============================================================================
# LEVEL 0 - Baby Safe Tools
# =============================================================================

def echo(text: str) -> str:
    """Simply repeat back the input text."""
    return text


def current_time(format: str = "full") -> str:
    """
    Get the current time.

    Args:
        format: "full", "time", "date", or custom strftime format
    """
    now = datetime.now()

    if format == "full":
        return now.strftime("%Y-%m-%d %H:%M:%S")
    elif format == "time":
        return now.strftime("%H:%M:%S")
    elif format == "date":
        return now.strftime("%Y-%m-%d")
    else:
        try:
            return now.strftime(format)
        except Exception:
            return now.strftime("%Y-%m-%d %H:%M:%S")


def simple_math(operation: str, a: float, b: float) -> str:
    """
    Perform simple arithmetic.

    Args:
        operation: "add", "subtract", "multiply", "divide"
        a: First number
        b: Second number
    """
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else "Cannot divide by zero",
    }

    if operation not in operations:
        return f"Unknown operation: {operation}. Use: add, subtract, multiply, divide"

    result = operations[operation](a, b)
    return str(result)


def random_number(min_val: int = 1, max_val: int = 100) -> str:
    """
    Generate a random number.

    Args:
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)
    """
    return str(random.randint(min_val, max_val))


# =============================================================================
# LEVEL 1 - Toddler Tools
# =============================================================================

def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.

    Supports: +, -, *, /, **, sqrt(), abs(), sin(), cos(), tan()

    Args:
        expression: Mathematical expression to evaluate
    """
    # Safe math functions
    safe_dict = {
        "sqrt": math.sqrt,
        "abs": abs,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
    }

    try:
        # Remove any dangerous characters
        allowed_chars = set("0123456789+-*/().sqrtabsincotagloexpe ")
        expression_clean = "".join(c for c in expression if c in allowed_chars or c.isalpha())

        # Evaluate with restricted globals
        result = eval(expression_clean, {"__builtins__": {}}, safe_dict)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def word_count(text: str) -> str:
    """
    Count words, characters, and sentences in text.

    Args:
        text: Text to analyze
    """
    words = len(text.split())
    chars = len(text)
    chars_no_space = len(text.replace(" ", ""))
    sentences = text.count(".") + text.count("!") + text.count("?")

    return f"Words: {words}, Characters: {chars}, Characters (no spaces): {chars_no_space}, Sentences: {sentences}"


def text_reverse(text: str) -> str:
    """Reverse the input text."""
    return text[::-1]


def temperature_convert(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert temperature between units.

    Args:
        value: Temperature value
        from_unit: "C", "F", or "K"
        to_unit: "C", "F", or "K"
    """
    # Convert to Celsius first
    if from_unit.upper() == "F":
        celsius = (value - 32) * 5 / 9
    elif from_unit.upper() == "K":
        celsius = value - 273.15
    elif from_unit.upper() == "C":
        celsius = value
    else:
        return f"Unknown unit: {from_unit}"

    # Convert from Celsius to target
    if to_unit.upper() == "F":
        result = celsius * 9 / 5 + 32
    elif to_unit.upper() == "K":
        result = celsius + 273.15
    elif to_unit.upper() == "C":
        result = celsius
    else:
        return f"Unknown unit: {to_unit}"

    return f"{value}{from_unit} = {result:.2f}{to_unit}"


# =============================================================================
# LEVEL 2 - Child Tools (Read-only external access)
# =============================================================================

def dictionary_lookup(word: str) -> str:
    """
    Look up a word definition (simplified).

    This is a placeholder - would integrate with real dictionary API.
    """
    # Simple built-in definitions for common words
    definitions = {
        "happy": "Feeling or showing pleasure or contentment",
        "sad": "Feeling or showing sorrow; unhappy",
        "computer": "An electronic device for processing data",
        "learn": "To gain knowledge or skill through study or experience",
        "friend": "A person with whom one has a bond of mutual affection",
    }

    word_lower = word.lower()
    if word_lower in definitions:
        return f"{word}: {definitions[word_lower]}"
    return f"Definition not found for: {word}"


def day_of_week(date_str: Optional[str] = None) -> str:
    """
    Get the day of the week for a date.

    Args:
        date_str: Date in YYYY-MM-DD format (default: today)
    """
    try:
        if date_str:
            date = datetime.strptime(date_str, "%Y-%m-%d")
        else:
            date = datetime.now()

        day_name = date.strftime("%A")
        return f"{date.strftime('%Y-%m-%d')} is a {day_name}"
    except Exception as e:
        return f"Error parsing date: {e}"


# =============================================================================
# LEVEL 3 - Adolescent Tools (Write access)
# =============================================================================

def note_create(title: str, content: str) -> str:
    """
    Create a note (simulated - actual implementation would write to file).

    Args:
        title: Note title
        content: Note content
    """
    # In real implementation, would save to data/notes/
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"Note created: '{title}' at {timestamp}\nContent: {content[:100]}..."


# =============================================================================
# LEVEL 4 - Young Adult Tools (Powerful)
# =============================================================================

def code_evaluate(code: str, language: str = "python") -> str:
    """
    Evaluate simple code (very restricted).

    This is intentionally limited for safety.
    """
    if language != "python":
        return f"Only Python is supported (got: {language})"

    # Very restricted - only allow simple expressions
    try:
        # Only allow basic operations
        allowed = set("0123456789+-*/() ")
        if all(c in allowed for c in code):
            result = eval(code, {"__builtins__": {}}, {})
            return f"Result: {result}"
        return "Code contains disallowed characters"
    except Exception as e:
        return f"Error: {e}"


# =============================================================================
# Tool Registration Helpers
# =============================================================================

def register_all_tools(registry: ToolRegistry):
    """Register all base tools with the registry."""

    # Level 0 - Baby Safe
    registry.register(
        name="echo",
        function=echo,
        description="Repeat back the input text",
        safety_level=ToolSafetyLevel.LEVEL_0,
        parameters={"text": {"type": "string", "description": "Text to echo"}},
        required_params=["text"],
        example_usage="echo(text='Hello')",
        teaches_competencies=["communication"],
    )

    registry.register(
        name="current_time",
        function=current_time,
        description="Get the current date and time",
        safety_level=ToolSafetyLevel.LEVEL_0,
        parameters={"format": {"type": "string", "description": "Output format"}},
        required_params=[],
        example_usage="current_time(format='full')",
        teaches_competencies=[],
    )

    registry.register(
        name="simple_math",
        function=simple_math,
        description="Perform basic arithmetic (add, subtract, multiply, divide)",
        safety_level=ToolSafetyLevel.LEVEL_0,
        parameters={
            "operation": {"type": "string", "description": "add/subtract/multiply/divide"},
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"},
        },
        required_params=["operation", "a", "b"],
        example_usage="simple_math(operation='add', a=5, b=3)",
        teaches_competencies=["reasoning"],
    )

    registry.register(
        name="random_number",
        function=random_number,
        description="Generate a random number in a range",
        safety_level=ToolSafetyLevel.LEVEL_0,
        parameters={
            "min_val": {"type": "integer", "description": "Minimum value"},
            "max_val": {"type": "integer", "description": "Maximum value"},
        },
        required_params=[],
        example_usage="random_number(min_val=1, max_val=10)",
        teaches_competencies=[],
    )

    # Level 1 - Toddler
    registry.register(
        name="calculator",
        function=calculator,
        description="Evaluate mathematical expressions",
        safety_level=ToolSafetyLevel.LEVEL_1,
        parameters={"expression": {"type": "string", "description": "Math expression"}},
        required_params=["expression"],
        example_usage="calculator(expression='sqrt(16) + 5')",
        teaches_competencies=["reasoning", "tool_use"],
    )

    registry.register(
        name="word_count",
        function=word_count,
        description="Count words, characters, and sentences in text",
        safety_level=ToolSafetyLevel.LEVEL_1,
        parameters={"text": {"type": "string", "description": "Text to analyze"}},
        required_params=["text"],
        example_usage="word_count(text='Hello world!')",
        teaches_competencies=["communication"],
    )

    registry.register(
        name="text_reverse",
        function=text_reverse,
        description="Reverse text",
        safety_level=ToolSafetyLevel.LEVEL_1,
        parameters={"text": {"type": "string", "description": "Text to reverse"}},
        required_params=["text"],
        example_usage="text_reverse(text='hello')",
        teaches_competencies=[],
    )

    registry.register(
        name="temperature_convert",
        function=temperature_convert,
        description="Convert temperature between Celsius, Fahrenheit, and Kelvin",
        safety_level=ToolSafetyLevel.LEVEL_1,
        parameters={
            "value": {"type": "number", "description": "Temperature value"},
            "from_unit": {"type": "string", "description": "C, F, or K"},
            "to_unit": {"type": "string", "description": "C, F, or K"},
        },
        required_params=["value", "from_unit", "to_unit"],
        example_usage="temperature_convert(value=100, from_unit='C', to_unit='F')",
        teaches_competencies=["reasoning"],
    )

    # Level 2 - Child
    registry.register(
        name="dictionary",
        function=dictionary_lookup,
        description="Look up word definitions",
        safety_level=ToolSafetyLevel.LEVEL_2,
        parameters={"word": {"type": "string", "description": "Word to look up"}},
        required_params=["word"],
        example_usage="dictionary(word='happy')",
        teaches_competencies=["communication", "memory"],
    )

    registry.register(
        name="day_of_week",
        function=day_of_week,
        description="Get the day of the week for a date",
        safety_level=ToolSafetyLevel.LEVEL_2,
        parameters={"date_str": {"type": "string", "description": "Date in YYYY-MM-DD format"}},
        required_params=[],
        example_usage="day_of_week(date_str='2024-01-01')",
        teaches_competencies=["reasoning"],
    )

    # Level 3 - Adolescent
    registry.register(
        name="note_create",
        function=note_create,
        description="Create and save a note",
        safety_level=ToolSafetyLevel.LEVEL_3,
        parameters={
            "title": {"type": "string", "description": "Note title"},
            "content": {"type": "string", "description": "Note content"},
        },
        required_params=["title", "content"],
        example_usage="note_create(title='Meeting', content='Discussed project...')",
        teaches_competencies=["memory", "tool_use"],
        required_milestones=["tool_mastery_level_2"],
    )

    # Level 4 - Young Adult
    registry.register(
        name="code_evaluate",
        function=code_evaluate,
        description="Evaluate simple code expressions",
        safety_level=ToolSafetyLevel.LEVEL_4,
        parameters={
            "code": {"type": "string", "description": "Code to evaluate"},
            "language": {"type": "string", "description": "Programming language"},
        },
        required_params=["code"],
        example_usage="code_evaluate(code='2 + 2')",
        teaches_competencies=["reasoning", "tool_use"],
        required_milestones=["tool_mastery_level_3", "independent_problem_solving"],
        minimum_competencies={"tool_use": 0.7, "reasoning": 0.6},
    )


# Tool instances for direct import
echo_tool = echo
current_time_tool = current_time
simple_math_tool = simple_math
calculator_tool = calculator
word_count_tool = word_count
