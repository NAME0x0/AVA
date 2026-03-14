from ava.tools import calculate, list_protocol_names, render_tool_trace


def test_calculate_basic_expression() -> None:
    assert calculate("17 * 29") == "493"


def test_calculate_safe_function() -> None:
    assert calculate("sqrt(81)") == "9"


def test_protocol_rendering() -> None:
    protocols = list_protocol_names()
    assert "compact_tags" in protocols
    assert "493" in render_tool_trace("compact_tags", "17*29", "493")
