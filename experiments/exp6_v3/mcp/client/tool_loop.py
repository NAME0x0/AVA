"""MCP tool-call loop for AVA v3.

Stub for P7. The production loop will live inside llama-server (which has
native MCP client support merged Mar 2026). This module provides:

1. A pure-Python reference loop for offline evaluation
2. The constrained-decoding state machine that switches between free decoding
   and JSON-schema-constrained decoding when inside <tool_call> tags
3. Glue to FastMCP stdio transport for the offline reference loop

Reference:
- aiproductivity.ai/news/llamacpp-merges-mcp-support-agentic-loop
- guidance-ai/llguidance
"""
from __future__ import annotations


class ToolLoop:
    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError("P7 task")
