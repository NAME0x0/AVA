"""
AVA Tool System
===============

Extensible tool framework with MCP (Model Context Protocol) support.

Features:
- Built-in tools (calculator, web search, file access)
- MCP server integration for external tools
- Automatic tool selection
- Safe execution with timeouts
"""

import asyncio
import json
import logging
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# =============================================================================
# Base Tool Interface
# =============================================================================


@dataclass
class ToolResult:
    """Result from tool execution."""

    success: bool
    output: Any
    error: str | None = None
    execution_time_ms: float = 0.0


@dataclass
class ToolDefinition:
    """Definition of a tool."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    required_params: list[str] = field(default_factory=list)

    def to_prompt_format(self) -> str:
        """Format for including in prompts."""
        params = ", ".join(
            f"{name}: {info.get('type', 'any')}" for name, info in self.parameters.items()
        )
        return f"{self.name}({params}) - {self.description}"


class Tool(ABC):
    """Base class for all tools."""

    name: str = "base_tool"
    description: str = "A tool"
    parameters: dict[str, Any] = {}
    required_params: list[str] = []

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            required_params=self.required_params,
        )


# =============================================================================
# Built-in Tools
# =============================================================================


class CalculatorTool(Tool):
    """Safe math calculator using AST parsing (no eval)."""

    name = "calculator"
    description = "Evaluate mathematical expressions safely"
    parameters = {"expression": {"type": "string", "description": "Math expression to evaluate"}}
    required_params = ["expression"]

    # Safe operators for arithmetic
    _SAFE_OPERATORS = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "/": lambda a, b: a / b if b != 0 else float("inf"),
        "%": lambda a, b: a % b if b != 0 else 0,
        "**": lambda a, b: a**b if b < 100 else float("inf"),  # Limit exponent
    }

    def _safe_eval(self, expression: str) -> float:
        """Safely evaluate arithmetic expression using AST."""
        import ast
        import operator

        # Supported AST node types
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        def _eval_node(node):
            if isinstance(node, ast.Constant):  # Python 3.8+
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError(f"Unsupported constant: {node.value}")
            elif isinstance(node, ast.Num):  # Python 3.7 compatibility
                return node.n
            elif isinstance(node, ast.BinOp):
                left = _eval_node(node.left)
                right = _eval_node(node.right)
                op = operators.get(type(node.op))
                if op is None:
                    raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
                # Safety limits
                if isinstance(node.op, ast.Pow) and right > 100:
                    raise ValueError("Exponent too large (max 100)")
                if isinstance(node.op, (ast.Div, ast.Mod)) and right == 0:
                    raise ValueError("Division by zero")
                return op(left, right)
            elif isinstance(node, ast.UnaryOp):
                operand = _eval_node(node.operand)
                op = operators.get(type(node.op))
                if op is None:
                    raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
                return op(operand)
            elif isinstance(node, ast.Expression):
                return _eval_node(node.body)
            else:
                raise ValueError(f"Unsupported expression type: {type(node).__name__}")

        tree = ast.parse(expression, mode="eval")
        return _eval_node(tree)

    async def execute(self, expression: str = "", **kwargs) -> ToolResult:
        import time

        start = time.time()

        try:
            # Validate input characters (extra safety layer)
            allowed = set("0123456789+-*/().% ")
            if not all(c in allowed for c in expression):
                return ToolResult(
                    success=False, output=None, error="Expression contains invalid characters"
                )

            result = self._safe_eval(expression)
            return ToolResult(
                success=True, output=result, execution_time_ms=(time.time() - start) * 1000
            )
        except (ValueError, SyntaxError) as e:
            return ToolResult(success=False, output=None, error=str(e))
        except Exception as e:
            return ToolResult(success=False, output=None, error=f"Calculation error: {e}")


class DateTimeTool(Tool):
    """Get current date and time."""

    name = "datetime"
    description = "Get current date, time, or perform date calculations"
    parameters = {
        "format": {"type": "string", "description": "Output format (optional)"},
        "timezone": {"type": "string", "description": "Timezone (optional)"},
    }

    async def execute(self, format: str = None, **kwargs) -> ToolResult:
        import time

        start = time.time()

        now = datetime.now()
        if format:
            try:
                output = now.strftime(format)
            except (ValueError, TypeError):
                # Invalid format string - fall back to ISO format
                output = now.isoformat()
        else:
            output = {
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "day": now.strftime("%A"),
                "iso": now.isoformat(),
            }

        return ToolResult(
            success=True, output=output, execution_time_ms=(time.time() - start) * 1000
        )


class FileReadTool(Tool):
    """Read files from allowed directories (path traversal protected)."""

    name = "read_file"
    description = "Read contents of a file"
    parameters = {
        "path": {"type": "string", "description": "Path to file"},
        "lines": {"type": "integer", "description": "Max lines to read (optional)"},
    }
    required_params = ["path"]

    # Allowed directories (security) - resolved to absolute paths
    allowed_dirs: list[str] = ["data", "docs", "config"]

    def _is_path_allowed(self, file_path: Path) -> bool:
        """Check if path is within allowed directories (path traversal safe)."""
        # Resolve to absolute path to prevent /../ traversal attacks
        try:
            resolved = file_path.resolve()
        except (OSError, ValueError):
            return False

        # Get current working directory
        cwd = Path.cwd()

        # Check if resolved path is under any allowed directory
        for allowed_dir in self.allowed_dirs:
            allowed_path = (cwd / allowed_dir).resolve()
            try:
                # is_relative_to is the safe way to check containment
                resolved.relative_to(allowed_path)
                return True
            except ValueError:
                # Not relative to this allowed directory
                continue

        return False

    async def execute(self, path: str = "", lines: int = None, **kwargs) -> ToolResult:
        import time

        start = time.time()

        try:
            file_path = Path(path)

            # Security check (path traversal safe)
            if not self._is_path_allowed(file_path):
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Access denied. Allowed directories: {self.allowed_dirs}",
                )

            # Resolve to prevent symlink attacks
            resolved_path = file_path.resolve()

            if not resolved_path.exists():
                return ToolResult(success=False, output=None, error="File not found")

            if not resolved_path.is_file():
                return ToolResult(success=False, output=None, error="Not a file")

            content = resolved_path.read_text()
            if lines:
                content = "\n".join(content.split("\n")[:lines])

            return ToolResult(
                success=True, output=content, execution_time_ms=(time.time() - start) * 1000
            )
        except PermissionError:
            return ToolResult(success=False, output=None, error="Permission denied")
        except UnicodeDecodeError:
            return ToolResult(success=False, output=None, error="File is not text/UTF-8")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class WebSearchTool(Tool):
    """
    Search-First Web Search Tool.

    Primary epistemic action for the Answer Machine paradigm.
    Uses multiple search providers for fact convergence.
    """

    name = "web_search"
    description = "Search the web for factual information (PRIMARY action for queries)"
    parameters = {
        "query": {"type": "string", "description": "Search query"},
        "num_results": {"type": "integer", "description": "Number of results (default 10)"},
        "provider": {
            "type": "string",
            "description": "Search provider: duckduckgo, brave (default duckduckgo)",
        },
    }
    required_params = ["query"]

    # Search providers
    PROVIDERS = {
        "duckduckgo": "https://html.duckduckgo.com/html/",
        "brave": "https://search.brave.com/search",
    }

    async def execute(
        self, query: str = "", num_results: int = 10, provider: str = "duckduckgo", **kwargs
    ) -> ToolResult:
        import time

        start = time.time()

        try:
            results = await self._search_provider(query, num_results, provider)

            if not results:
                # Fallback to alternative provider
                alt_provider = "brave" if provider == "duckduckgo" else "duckduckgo"
                results = await self._search_provider(query, num_results, alt_provider)

            return ToolResult(
                success=len(results) > 0,
                output={
                    "query": query,
                    "provider": provider,
                    "results": results,
                    "num_results": len(results),
                },
                error=None if results else "No results found",
                execution_time_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

    async def _search_provider(
        self,
        query: str,
        num_results: int,
        provider: str,
    ) -> list[dict[str, str]]:
        """Execute search on a specific provider."""
        import re

        results = []

        try:
            async with httpx.AsyncClient() as client:
                if provider == "duckduckgo":
                    response = await client.get(
                        self.PROVIDERS["duckduckgo"],
                        params={"q": query},
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                        },
                        timeout=15.0,
                    )

                    if response.status_code == 200:
                        text = response.text

                        # Extract results
                        snippets = re.findall(r'<a class="result__snippet"[^>]*>([^<]+)</a>', text)
                        titles = re.findall(
                            r'<a class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>', text
                        )

                        for _i, ((url, title), snippet) in enumerate(
                            zip(titles[:num_results], snippets[:num_results], strict=False)
                        ):
                            results.append(
                                {
                                    "title": title.strip(),
                                    "snippet": snippet.strip(),
                                    "url": url,
                                    "source": "duckduckgo",
                                }
                            )

                elif provider == "brave":
                    response = await client.get(
                        self.PROVIDERS["brave"],
                        params={"q": query, "source": "web"},
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                        },
                        timeout=15.0,
                    )

                    if response.status_code == 200:
                        text = response.text

                        # Extract from Brave's HTML
                        # Simplified extraction
                        desc_matches = re.findall(
                            r'<p class="snippet-description"[^>]*>([^<]+)</p>', text
                        )
                        title_matches = re.findall(
                            r'<a class="result-header"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>', text
                        )

                        for _i, ((url, title), desc) in enumerate(
                            zip(
                                title_matches[:num_results],
                                desc_matches[:num_results],
                                strict=False,
                            )
                        ):
                            results.append(
                                {
                                    "title": title.strip(),
                                    "snippet": desc.strip(),
                                    "url": url,
                                    "source": "brave",
                                }
                            )

        except Exception as e:
            logger.error(f"Search provider {provider} failed: {e}")

        return results


class WebBrowseTool(Tool):
    """
    Web browsing tool for detailed information retrieval.

    Fetches and extracts content from web pages for deeper
    information gathering in the Answer Machine workflow.
    """

    name = "web_browse"
    description = "Browse a web page and extract its content"
    parameters = {
        "url": {"type": "string", "description": "URL to browse"},
        "extract_type": {
            "type": "string",
            "description": "What to extract: text, links, structured (default text)",
        },
        "max_chars": {
            "type": "integer",
            "description": "Maximum characters to return (default 5000)",
        },
    }
    required_params = ["url"]

    async def execute(
        self, url: str = "", extract_type: str = "text", max_chars: int = 5000, **kwargs
    ) -> ToolResult:
        import time

        start = time.time()

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    },
                    timeout=20.0,
                    follow_redirects=True,
                )

                if response.status_code != 200:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"Failed to fetch URL: {response.status_code}",
                    )

                content = response.text
                extracted = self._extract_content(content, extract_type, max_chars)

                return ToolResult(
                    success=True,
                    output={
                        "url": url,
                        "extract_type": extract_type,
                        "content": extracted,
                        "content_length": len(extracted),
                    },
                    execution_time_ms=(time.time() - start) * 1000,
                )

        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

    def _extract_content(
        self,
        html: str,
        extract_type: str,
        max_chars: int,
    ) -> str:
        """Extract content from HTML."""
        import re

        # Remove script and style tags
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)

        if extract_type == "text":
            # Extract plain text
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text)
            text = text.strip()
            return text[:max_chars]

        elif extract_type == "links":
            # Extract links
            links = re.findall(r'href="([^"]+)"', html)
            return "\n".join(links[:100])

        elif extract_type == "structured":
            # Extract structured content (headings, paragraphs)
            result = []

            # Headings
            for h_tag in ["h1", "h2", "h3"]:
                headings = re.findall(f"<{h_tag}[^>]*>([^<]+)</{h_tag}>", html, re.IGNORECASE)
                for h in headings[:5]:
                    result.append(f"## {h.strip()}")

            # Paragraphs
            paragraphs = re.findall(r"<p[^>]*>([^<]+)</p>", html, re.IGNORECASE)
            for p in paragraphs[:20]:
                if len(p.strip()) > 50:
                    result.append(p.strip())

            return "\n\n".join(result)[:max_chars]

        return html[:max_chars]


class FactVerificationTool(Tool):
    """
    Tool for verifying facts across multiple sources.

    Implements the Audit-Verify-Research-Update workflow
    for the Answer Machine paradigm.
    """

    name = "verify_fact"
    description = "Verify a fact by checking multiple sources"
    parameters = {
        "claim": {"type": "string", "description": "The claim to verify"},
        "num_sources": {"type": "integer", "description": "Number of sources to check (default 3)"},
    }
    required_params = ["claim"]

    async def execute(self, claim: str = "", num_sources: int = 3, **kwargs) -> ToolResult:
        import time

        start = time.time()

        try:
            # Search for verification
            search_tool = WebSearchTool()
            search_result = await search_tool.execute(
                query=f"fact check {claim}",
                num_results=num_sources * 2,
            )

            if not search_result.success:
                return search_result

            # Analyze results for convergence
            results = search_result.output.get("results", [])

            verification = {
                "claim": claim,
                "sources_checked": len(results),
                "sources": results,
                "convergence": self._calculate_convergence(results, claim),
                "confidence": 0.0,
            }

            # Calculate confidence based on convergence
            verification["confidence"] = verification["convergence"]

            return ToolResult(
                success=True, output=verification, execution_time_ms=(time.time() - start) * 1000
            )

        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

    def _calculate_convergence(
        self,
        results: list[dict],
        claim: str,
    ) -> float:
        """
        Calculate fact convergence across sources.

        Higher convergence = more sources agree.
        """
        if not results:
            return 0.0

        # Simple keyword matching for convergence
        claim_words = set(claim.lower().split())

        matches = 0
        for result in results:
            snippet = result.get("snippet", "").lower()
            snippet_words = set(snippet.split())

            overlap = len(claim_words & snippet_words) / len(claim_words) if claim_words else 0
            if overlap > 0.3:
                matches += 1

        return matches / len(results) if results else 0.0


# =============================================================================
# MCP Client
# =============================================================================


@dataclass
class MCPServer:
    """Configuration for an MCP server."""

    name: str
    command: str  # Command to start the server
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    transport: str = "stdio"  # "stdio" or "http"
    url: str | None = None  # For HTTP transport


class MCPClient:
    """
    Model Context Protocol client.

    Connects to MCP servers to access external tools and data sources.
    Follows the MCP specification: https://modelcontextprotocol.io/
    """

    def __init__(self):
        self.servers: dict[str, MCPServer] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._tools: dict[str, dict[str, Any]] = {}  # server_name -> tools

    async def add_server(self, server: MCPServer) -> bool:
        """Add and connect to an MCP server."""
        try:
            if server.transport == "stdio":
                # Start the server process
                process = subprocess.Popen(
                    [server.command] + server.args,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env={**dict(subprocess.os.environ), **server.env},
                )
                self._processes[server.name] = process

                # Initialize connection
                await self._initialize_server(server.name)

            elif server.transport == "http" and server.url:
                # HTTP-based MCP server
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{server.url}/initialize", json={"protocolVersion": "0.1.0"}
                    )
                    if response.status_code != 200:
                        return False

            self.servers[server.name] = server
            logger.info(f"MCP server connected: {server.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect MCP server {server.name}: {e}")
            return False

    async def _initialize_server(self, server_name: str):
        """Initialize an MCP server and get available tools."""
        # Send initialize request via JSON-RPC
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "0.1.0",
                "clientInfo": {"name": "AVA", "version": "3.1.0"},
            },
        }

        response = await self._send_request(server_name, request)
        if response:
            # Get available tools
            tools_request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
            tools_response = await self._send_request(server_name, tools_request)
            if tools_response and "result" in tools_response:
                self._tools[server_name] = tools_response["result"].get("tools", [])

    async def _send_request(self, server_name: str, request: dict) -> dict | None:
        """Send JSON-RPC request to server."""
        if server_name not in self._processes:
            return None

        process = self._processes[server_name]
        try:
            # Write request
            request_str = json.dumps(request) + "\n"
            process.stdin.write(request_str.encode())
            process.stdin.flush()

            # Read response
            response_str = process.stdout.readline().decode()
            return json.loads(response_str)

        except Exception as e:
            logger.error(f"MCP request failed: {e}")
            return None

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> ToolResult:
        """Call a tool on an MCP server."""
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }

        response = await self._send_request(server_name, request)

        if response and "result" in response:
            return ToolResult(success=True, output=response["result"])
        elif response and "error" in response:
            return ToolResult(
                success=False, output=None, error=response["error"].get("message", "Unknown error")
            )
        else:
            return ToolResult(success=False, output=None, error="No response from server")

    def get_all_tools(self) -> list[dict[str, Any]]:
        """Get all available tools from all servers."""
        all_tools = []
        for server_name, tools in self._tools.items():
            for tool in tools:
                tool["server"] = server_name
                all_tools.append(tool)
        return all_tools

    async def shutdown(self):
        """Shutdown all MCP servers."""
        for _name, process in self._processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                process.kill()
        self._processes.clear()


# =============================================================================
# Tool Manager
# =============================================================================


class ToolManager:
    """
    Manages all tools (built-in and MCP).

    Responsibilities:
    - Register and discover tools
    - Select appropriate tools for queries
    - Execute tools safely
    - Aggregate results
    """

    def __init__(self, config=None):
        self.config = config

        # Built-in tools
        self._tools: dict[str, Tool] = {}
        self._register_builtin_tools()

        # MCP client
        self.mcp = MCPClient()

        logger.info(f"ToolManager initialized with {len(self._tools)} built-in tools")

    def _register_builtin_tools(self):
        """Register built-in tools."""
        builtins = [
            CalculatorTool(),
            DateTimeTool(),
            FileReadTool(),
            # Search-First tools (priority order)
            WebSearchTool(),
            WebBrowseTool(),
            FactVerificationTool(),
        ]
        for tool in builtins:
            self._tools[tool.name] = tool

    def register(self, tool: Tool):
        """Register a custom tool."""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        """List all available tools."""
        definitions = [tool.get_definition() for tool in self._tools.values()]

        # Add MCP tools
        for mcp_tool in self.mcp.get_all_tools():
            definitions.append(
                ToolDefinition(
                    name=f"mcp:{mcp_tool['server']}:{mcp_tool['name']}",
                    description=mcp_tool.get("description", ""),
                    parameters=mcp_tool.get("inputSchema", {}).get("properties", {}),
                )
            )

        return definitions

    def get_tools_prompt(self) -> str:
        """Get formatted tool list for prompts."""
        lines = ["Available tools:"]
        for defn in self.list_tools():
            lines.append(f"- {defn.to_prompt_format()}")
        return "\n".join(lines)

    async def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        # Check if it's an MCP tool
        if tool_name.startswith("mcp:"):
            parts = tool_name.split(":")
            if len(parts) >= 3:
                server, name = parts[1], parts[2]
                return await self.mcp.call_tool(server, name, kwargs)

        # Built-in tool
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(success=False, output=None, error=f"Tool not found: {tool_name}")

        try:
            return await asyncio.wait_for(tool.execute(**kwargs), timeout=30.0)  # 30 second timeout
        except asyncio.TimeoutError:
            return ToolResult(success=False, output=None, error="Tool execution timed out")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

    async def auto_execute(self, query: str) -> list[ToolResult]:
        """
        Automatically determine and execute relevant tools.

        SEARCH-FIRST PARADIGM: Web search is the PRIMARY action for
        informational queries. We prioritize external data over generation.
        """
        results = []
        query_lower = query.lower()

        # SEARCH-FIRST: Check for informational queries FIRST
        # These trigger web search as the PRIMARY action
        informational_indicators = [
            # Question words
            "what",
            "when",
            "where",
            "who",
            "how",
            "why",
            "which",
            # Information seeking
            "tell me",
            "explain",
            "describe",
            "is it true",
            "define",
            # Current events / facts
            "latest",
            "news",
            "current",
            "today",
            "recent",
            "update",
            # Lookup patterns
            "search",
            "find",
            "look up",
            "what is",
            "who is",
        ]

        # Check if this is an informational query (Search-First trigger)
        is_informational = "?" in query or any(  # Any question
            ind in query_lower for ind in informational_indicators
        )

        if is_informational:
            # PRIMARY ACTION: Web search first
            result = await self.execute("web_search", query=query, num_results=10)
            results.append(result)

            # If search returned results, optionally verify facts
            if result.success and result.output.get("num_results", 0) >= 3:
                # Extract key claim for verification if needed
                pass  # Fact verification happens at a higher level

        # Calculator for math (specific, not informational)
        if any(op in query for op in ["+", "-", "*", "/", "calculate", "compute"]):
            import re

            expr_match = re.search(r"[\d\s\+\-\*\/\.\(\)]+", query)
            if expr_match:
                result = await self.execute("calculator", expression=expr_match.group().strip())
                if result.success:
                    results.append(result)

        # DateTime for time queries
        if any(word in query_lower for word in ["time", "date", "day"]):
            # Only if asking about current time, not general time questions
            if not is_informational or any(
                w in query_lower for w in ["what time", "current time", "now"]
            ):
                result = await self.execute("datetime")
                results.append(result)

        return results

    async def shutdown(self):
        """Cleanup resources."""
        await self.mcp.shutdown()
