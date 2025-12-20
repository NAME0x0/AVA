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
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path

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
    error: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class ToolDefinition:
    """Definition of a tool."""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_params: List[str] = field(default_factory=list)
    
    def to_prompt_format(self) -> str:
        """Format for including in prompts."""
        params = ", ".join(
            f"{name}: {info.get('type', 'any')}"
            for name, info in self.parameters.items()
        )
        return f"{self.name}({params}) - {self.description}"


class Tool(ABC):
    """Base class for all tools."""
    
    name: str = "base_tool"
    description: str = "A tool"
    parameters: Dict[str, Any] = {}
    required_params: List[str] = []
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            required_params=self.required_params
        )


# =============================================================================
# Built-in Tools
# =============================================================================

class CalculatorTool(Tool):
    """Safe math calculator."""
    
    name = "calculator"
    description = "Evaluate mathematical expressions safely"
    parameters = {"expression": {"type": "string", "description": "Math expression to evaluate"}}
    required_params = ["expression"]
    
    async def execute(self, expression: str = "", **kwargs) -> ToolResult:
        import time
        start = time.time()
        
        try:
            # Safe evaluation - only allow math operations
            allowed = set("0123456789+-*/().% ")
            if not all(c in allowed for c in expression):
                return ToolResult(
                    success=False,
                    output=None,
                    error="Expression contains invalid characters"
                )
            
            result = eval(expression)
            return ToolResult(
                success=True,
                output=result,
                execution_time_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class DateTimeTool(Tool):
    """Get current date and time."""
    
    name = "datetime"
    description = "Get current date, time, or perform date calculations"
    parameters = {
        "format": {"type": "string", "description": "Output format (optional)"},
        "timezone": {"type": "string", "description": "Timezone (optional)"}
    }
    
    async def execute(self, format: str = None, **kwargs) -> ToolResult:
        import time
        start = time.time()
        
        now = datetime.now()
        if format:
            try:
                output = now.strftime(format)
            except:
                output = now.isoformat()
        else:
            output = {
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "day": now.strftime("%A"),
                "iso": now.isoformat()
            }
        
        return ToolResult(
            success=True,
            output=output,
            execution_time_ms=(time.time() - start) * 1000
        )


class FileReadTool(Tool):
    """Read files from allowed directories."""
    
    name = "read_file"
    description = "Read contents of a file"
    parameters = {
        "path": {"type": "string", "description": "Path to file"},
        "lines": {"type": "integer", "description": "Max lines to read (optional)"}
    }
    required_params = ["path"]
    
    # Allowed directories (security)
    allowed_dirs: List[str] = ["data", "docs", "config"]
    
    async def execute(self, path: str = "", lines: int = None, **kwargs) -> ToolResult:
        import time
        start = time.time()
        
        try:
            file_path = Path(path)
            
            # Security check
            if not any(file_path.is_relative_to(d) or str(file_path).startswith(d) 
                      for d in self.allowed_dirs):
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Access denied. Allowed directories: {self.allowed_dirs}"
                )
            
            if not file_path.exists():
                return ToolResult(success=False, output=None, error="File not found")
            
            content = file_path.read_text()
            if lines:
                content = "\n".join(content.split("\n")[:lines])
            
            return ToolResult(
                success=True,
                output=content,
                execution_time_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class WebSearchTool(Tool):
    """Search the web using DuckDuckGo."""
    
    name = "web_search"
    description = "Search the web for information"
    parameters = {
        "query": {"type": "string", "description": "Search query"},
        "num_results": {"type": "integer", "description": "Number of results (default 5)"}
    }
    required_params = ["query"]
    
    async def execute(self, query: str = "", num_results: int = 5, **kwargs) -> ToolResult:
        import time
        start = time.time()
        
        try:
            # Use DuckDuckGo Lite API
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=10.0
                )
                
                if response.status_code != 200:
                    return ToolResult(
                        success=False, output=None,
                        error=f"Search failed: {response.status_code}"
                    )
                
                # Parse results (simple extraction)
                results = []
                text = response.text
                
                # Extract result snippets (simplified)
                import re
                snippets = re.findall(r'<a class="result__snippet"[^>]*>([^<]+)</a>', text)
                titles = re.findall(r'<a class="result__a"[^>]*>([^<]+)</a>', text)
                
                for i, (title, snippet) in enumerate(zip(titles[:num_results], snippets[:num_results])):
                    results.append({
                        "title": title.strip(),
                        "snippet": snippet.strip()
                    })
                
                return ToolResult(
                    success=True,
                    output=results,
                    execution_time_ms=(time.time() - start) * 1000
                )
                
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


# =============================================================================
# MCP Client
# =============================================================================

@dataclass
class MCPServer:
    """Configuration for an MCP server."""
    name: str
    command: str  # Command to start the server
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    transport: str = "stdio"  # "stdio" or "http"
    url: Optional[str] = None  # For HTTP transport


class MCPClient:
    """
    Model Context Protocol client.
    
    Connects to MCP servers to access external tools and data sources.
    Follows the MCP specification: https://modelcontextprotocol.io/
    """
    
    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}
        self._processes: Dict[str, subprocess.Popen] = {}
        self._tools: Dict[str, Dict[str, Any]] = {}  # server_name -> tools
        
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
                    env={**dict(subprocess.os.environ), **server.env}
                )
                self._processes[server.name] = process
                
                # Initialize connection
                await self._initialize_server(server.name)
                
            elif server.transport == "http" and server.url:
                # HTTP-based MCP server
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{server.url}/initialize",
                        json={"protocolVersion": "0.1.0"}
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
                "clientInfo": {"name": "AVA", "version": "3.1.0"}
            }
        }
        
        response = await self._send_request(server_name, request)
        if response:
            # Get available tools
            tools_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list"
            }
            tools_response = await self._send_request(server_name, tools_request)
            if tools_response and "result" in tools_response:
                self._tools[server_name] = tools_response["result"].get("tools", [])
    
    async def _send_request(self, server_name: str, request: dict) -> Optional[dict]:
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
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> ToolResult:
        """Call a tool on an MCP server."""
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        response = await self._send_request(server_name, request)
        
        if response and "result" in response:
            return ToolResult(
                success=True,
                output=response["result"]
            )
        elif response and "error" in response:
            return ToolResult(
                success=False,
                output=None,
                error=response["error"].get("message", "Unknown error")
            )
        else:
            return ToolResult(success=False, output=None, error="No response from server")
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get all available tools from all servers."""
        all_tools = []
        for server_name, tools in self._tools.items():
            for tool in tools:
                tool["server"] = server_name
                all_tools.append(tool)
        return all_tools
    
    async def shutdown(self):
        """Shutdown all MCP servers."""
        for name, process in self._processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
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
    
    def __init__(self, config = None):
        self.config = config
        
        # Built-in tools
        self._tools: Dict[str, Tool] = {}
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
            WebSearchTool(),
        ]
        for tool in builtins:
            self._tools[tool.name] = tool
    
    def register(self, tool: Tool):
        """Register a custom tool."""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[ToolDefinition]:
        """List all available tools."""
        definitions = [tool.get_definition() for tool in self._tools.values()]
        
        # Add MCP tools
        for mcp_tool in self.mcp.get_all_tools():
            definitions.append(ToolDefinition(
                name=f"mcp:{mcp_tool['server']}:{mcp_tool['name']}",
                description=mcp_tool.get("description", ""),
                parameters=mcp_tool.get("inputSchema", {}).get("properties", {})
            ))
        
        return definitions
    
    def get_tools_prompt(self) -> str:
        """Get formatted tool list for prompts."""
        lines = ["Available tools:"]
        for defn in self.list_tools():
            lines.append(f"- {defn.to_prompt_format()}")
        return "\n".join(lines)
    
    async def execute(
        self,
        tool_name: str,
        **kwargs
    ) -> ToolResult:
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
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool not found: {tool_name}"
            )
        
        try:
            return await asyncio.wait_for(
                tool.execute(**kwargs),
                timeout=30.0  # 30 second timeout
            )
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                output=None,
                error="Tool execution timed out"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    async def auto_execute(self, query: str) -> List[ToolResult]:
        """
        Automatically determine and execute relevant tools.
        
        This analyzes the query and runs appropriate tools.
        """
        results = []
        query_lower = query.lower()
        
        # Calculator for math
        if any(op in query for op in ["+", "-", "*", "/", "calculate", "compute"]):
            # Extract expression (simplified)
            import re
            expr_match = re.search(r'[\d\s\+\-\*\/\.\(\)]+', query)
            if expr_match:
                result = await self.execute("calculator", expression=expr_match.group().strip())
                if result.success:
                    results.append(result)
        
        # DateTime for time queries
        if any(word in query_lower for word in ["time", "date", "today", "now", "day"]):
            result = await self.execute("datetime")
            results.append(result)
        
        # Web search for information queries
        if any(word in query_lower for word in ["search", "find", "what is", "who is", "look up"]):
            result = await self.execute("web_search", query=query)
            results.append(result)
        
        return results
    
    async def shutdown(self):
        """Cleanup resources."""
        await self.mcp.shutdown()
