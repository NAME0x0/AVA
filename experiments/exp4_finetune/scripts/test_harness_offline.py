"""Test the AVA harness components that don't require GPU.

Tests: tool execution, memory store, tool parsing, message building.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "harness"))

from engine import (
    CalculatorTool, PythonTool, MemoryStore, Message, ToolResult,
    GenerationConfig,
)


def test_calculator_tool():
    calc = CalculatorTool()
    assert calc.name == "calculator"

    # Basic arithmetic
    assert calc.execute(expression="2 + 3") == "5"
    assert calc.execute(expression="10 * 5") == "50"
    assert calc.execute(expression="100 / 4") == "25.0"
    assert calc.execute(expression="2 ** 10") == "1024"

    # Math functions
    result = calc.execute(expression="sqrt(144)")
    assert float(result) == 12.0

    result = calc.execute(expression="round(3.14159, 2)")
    assert result == "3.14"

    # Error handling
    result = calc.execute(expression="1/0")
    assert "Error" in result

    # No dangerous builtins
    result = calc.execute(expression="__import__('os').system('echo hack')")
    assert "Error" in result

    print("  [PASS] CalculatorTool")


def test_python_tool():
    py = PythonTool()
    assert py.name == "python"

    # Basic execution
    result = py.execute(code="print(2 + 3)")
    assert result.strip() == "5"

    # Multi-line
    result = py.execute(code="for i in range(3):\n    print(i)")
    assert "0" in result and "1" in result and "2" in result

    # Error handling
    result = py.execute(code="raise ValueError('test')")
    assert "Error" in result and "ValueError" in result

    # No output
    result = py.execute(code="x = 42")
    assert "no output" in result.lower()

    print("  [PASS] PythonTool")


def test_memory_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        mem = MemoryStore(tmpdir)

        # Add memories
        mem.add("We discussed Python programming", kind="conversation")
        mem.add("User prefers dark mode", kind="preference")
        mem.add("Machine learning project deadline is Friday", kind="task")

        assert len(mem.memories) == 3

        # Search
        results = mem.search("Python programming")
        assert len(results) >= 1
        assert "Python" in results[0]["content"]

        results = mem.search("machine learning project")
        assert len(results) >= 1
        assert "Machine learning" in results[0]["content"]

        # No results for irrelevant query
        results = mem.search("quantum physics entanglement")
        # Some words might partially match, so just check it doesn't crash

        # Recent memories
        recent = mem.get_recent(2)
        assert len(recent) == 2

        # Persistence — reload from disk
        mem2 = MemoryStore(tmpdir)
        assert len(mem2.memories) == 3

    print("  [PASS] MemoryStore")


def test_tool_parsing():
    """Test that tool call parsing works correctly."""
    # We can't instantiate AVAEngine without GPU, so test the regex directly
    import re
    pattern = r'<tool_call>(.*?)</tool_call>'

    # Single tool call
    text = 'Let me calculate. <tool_call>{"name": "calculator", "arguments": {"expression": "2+3"}}</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)
    assert len(matches) == 1
    parsed = json.loads(matches[0])
    assert parsed["name"] == "calculator"
    assert parsed["arguments"]["expression"] == "2+3"

    # Multiple tool calls
    text = '''First: <tool_call>{"name": "calculator", "arguments": {"expression": "10*5"}}</tool_call>
    Then: <tool_call>{"name": "python", "arguments": {"code": "print('hello')"}}</tool_call>'''
    matches = re.findall(pattern, text, re.DOTALL)
    assert len(matches) == 2

    # No tool calls
    text = "Just a regular response without any tools."
    matches = re.findall(pattern, text, re.DOTALL)
    assert len(matches) == 0

    # Malformed JSON
    text = '<tool_call>not json</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)
    assert len(matches) == 1
    try:
        json.loads(matches[0])
        assert False, "Should have raised"
    except json.JSONDecodeError:
        pass  # Expected

    print("  [PASS] Tool call parsing")


def test_message_dataclass():
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert msg.tool_calls is None
    assert msg.timestamp > 0

    msg2 = Message(
        role="assistant",
        content="Let me help",
        tool_calls=[{"name": "calculator", "arguments": {"expression": "1+1"}}],
        thinking="I should use the calculator",
    )
    assert msg2.thinking is not None
    assert len(msg2.tool_calls) == 1

    print("  [PASS] Message dataclass")


def test_generation_config():
    config = GenerationConfig()
    assert config.max_new_tokens == 512
    assert config.temperature == 0.7
    assert config.enable_thinking is True

    config2 = GenerationConfig(max_new_tokens=1024, temperature=0.0, do_sample=False)
    assert config2.max_new_tokens == 1024
    assert config2.do_sample is False

    print("  [PASS] GenerationConfig")


def test_tool_schema():
    calc = CalculatorTool()
    schema = calc.schema()
    assert schema["name"] == "calculator"
    assert "expression" in schema["parameters"]

    py = PythonTool()
    schema = py.schema()
    assert schema["name"] == "python"
    assert "code" in schema["parameters"]

    print("  [PASS] Tool schemas")


if __name__ == "__main__":
    print("AVA Harness Offline Tests")
    print("=" * 40)
    test_calculator_tool()
    test_python_tool()
    test_memory_store()
    test_tool_parsing()
    test_message_dataclass()
    test_generation_config()
    test_tool_schema()
    print(f"\n{'='*40}")
    print("All tests passed!")
