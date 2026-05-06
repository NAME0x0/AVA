"""FastMCP 3.0 stdio server for AVA v3.

Exposes the tools listed in TOOLS.md. Each tool lives in tools/<name>.py and
registers via @mcp.tool() with type-hint-driven JSON schema generation.

P0 stub. Wire up actual tool registration at P7.

Run:
    python main.py            # stdio transport (default for llama-server MCP client)
    python main.py --http     # HTTP transport on :8770 (debugging)

Reference:
- gofastmcp.com/servers/tools
- modelcontextprotocol.io
"""
from __future__ import annotations


def build_server():  # type: ignore[no-untyped-def]
    raise NotImplementedError("P7 task — wire FastMCP @mcp.tool() registrations")


if __name__ == "__main__":
    raise SystemExit("AVA v3 MCP server skeleton — not implemented yet")
