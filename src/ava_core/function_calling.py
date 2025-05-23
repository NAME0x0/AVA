#!/usr/bin/env python3
"""
Enhanced Function Calling and Tool Use Logic for AVA
Provides robust tool detection, execution, and structured output parsing
for local agentic AI on RTX A2000 4GB VRAM constraints.
"""

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime
import inspect

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Standard format for tool execution results."""
    success: bool
    result: Any
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None
    tool_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    """Represents a parsed tool call from LLM output."""
    tool_name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None
    confidence: float = 1.0


class BaseTool(ABC):
    """Abstract base class for all AVA tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given arguments."""
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Return the tool's parameter schema."""
        pass


class FunctionCaller:
    """
    Enhanced Function Caller for AVA's agentic capabilities.
    Supports both string parsing and structured JSON function calls.
    """
    
    def __init__(self, available_tools: Optional[Dict[str, Union[Callable, BaseTool]]] = None):
        """
        Initialize the Function Caller.
        
        Args:
            available_tools: Dictionary of tool names to callables or BaseTool instances
        """
        self.available_tools: Dict[str, Union[Callable, BaseTool]] = available_tools or {}
        self._tool_schemas: Dict[str, Dict[str, Any]] = {}
        
        # Compile regex patterns for performance
        self._tool_call_pattern = re.compile(r'\[TOOL_CALL:\s*([^)]+)\]', re.IGNORECASE)
        self._function_pattern = re.compile(r'(\w+)\((.*?)\)$')
        
        self._register_tool_schemas()
        logger.info(f"Function Caller initialized with {len(self.available_tools)} tools: {list(self.available_tools.keys())}")
    
    def _register_tool_schemas(self):
        """Register schemas for all available tools."""
        for tool_name, tool in self.available_tools.items():
            if isinstance(tool, BaseTool):
                self._tool_schemas[tool_name] = tool.get_schema()
            else:
                # Generate schema from function signature
                self._tool_schemas[tool_name] = self._generate_function_schema(tool)
    
    def _generate_function_schema(self, func: Callable) -> Dict[str, Any]:
        """Generate a basic schema from function signature."""
        try:
            sig = inspect.signature(func)
            parameters = {}
            for param_name, param in sig.parameters.items():
                param_info = {"type": "string"}  # Default type
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation in (int, float):
                        param_info["type"] = "number"
                    elif param.annotation == bool:
                        param_info["type"] = "boolean"
                
                if param.default != inspect.Parameter.empty:
                    param_info["default"] = param.default
                
                parameters[param_name] = param_info
            
            return {
                "type": "function",
                "parameters": parameters,
                "description": func.__doc__ or f"Execute {func.__name__}"
            }
        except Exception as e:
            logger.warning(f"Could not generate schema for {func.__name__}: {e}")
            return {"type": "function", "parameters": {}}
    
    def add_tool(self, name: str, tool: Union[Callable, BaseTool]):
        """Add a new tool to the available tools."""
        self.available_tools[name] = tool
        if isinstance(tool, BaseTool):
            self._tool_schemas[name] = tool.get_schema()
        else:
            self._tool_schemas[name] = self._generate_function_schema(tool)
        logger.info(f"Added tool: {name}")
    
    def remove_tool(self, name: str) -> bool:
        """Remove a tool from available tools."""
        if name in self.available_tools:
            del self.available_tools[name]
            if name in self._tool_schemas:
                del self._tool_schemas[name]
            logger.info(f"Removed tool: {name}")
            return True
        return False
    
    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available tools."""
        return {
            name: {
                "schema": schema,
                "type": "BaseTool" if isinstance(tool, BaseTool) else "function"
            }
            for name, (tool, schema) in zip(self.available_tools.keys(), 
                                          zip(self.available_tools.values(), self._tool_schemas.values()))
        }
    
    def needs_function_call(self, llm_output: Union[str, Dict[str, Any]]) -> bool:
        """
        Determine if the LLM output indicates a need for a function call.
        
        Args:
            llm_output: Raw output from the LLM (string or structured dict)
            
        Returns:
            True if a function call is indicated
        """
        try:
            if isinstance(llm_output, str):
                # Check for string-based tool call pattern
                return bool(self._tool_call_pattern.search(llm_output))
            
            elif isinstance(llm_output, dict):
                # Check for structured tool calls
                return bool(llm_output.get("tool_calls")) or bool(llm_output.get("function_call"))
            
        except Exception as e:
            logger.error(f"Error checking for function call: {e}")
        
        return False
    
    def parse_function_calls(self, llm_output: Union[str, Dict[str, Any]]) -> List[ToolCall]:
        """
        Parse tool calls from LLM output.
        
        Args:
            llm_output: Raw output from the LLM
            
        Returns:
            List of parsed ToolCall objects
        """
        tool_calls = []
        
        try:
            if isinstance(llm_output, str):
                tool_calls.extend(self._parse_string_tool_calls(llm_output))
            elif isinstance(llm_output, dict):
                tool_calls.extend(self._parse_structured_tool_calls(llm_output))
                
        except Exception as e:
            logger.error(f"Error parsing function calls: {e}")
        
        return tool_calls
    
    def _parse_string_tool_calls(self, output: str) -> List[ToolCall]:
        """Parse tool calls from string format."""
        tool_calls = []
        
        for match in self._tool_call_pattern.finditer(output):
            call_content = match.group(1).strip()
            
            # Parse function name and arguments
            func_match = self._function_pattern.match(call_content)
            if not func_match:
                logger.warning(f"Could not parse function call: {call_content}")
                continue
            
            tool_name = func_match.group(1)
            args_str = func_match.group(2).strip()
            
            # Parse arguments
            arguments = {}
            if args_str:
                arguments = self._parse_function_arguments(args_str)
            
            if tool_name in self.available_tools:
                tool_calls.append(ToolCall(
                    tool_name=tool_name,
                    arguments=arguments,
                    call_id=f"call_{len(tool_calls)}"
                ))
                logger.debug(f"Parsed tool call: {tool_name} with args: {arguments}")
            else:
                logger.warning(f"Unknown tool in call: {tool_name}")
        
        return tool_calls
    
    def _parse_structured_tool_calls(self, output: Dict[str, Any]) -> List[ToolCall]:
        """Parse tool calls from structured JSON format."""
        tool_calls = []
        
        # Handle OpenAI-style tool_calls
        if "tool_calls" in output:
            for call_data in output["tool_calls"]:
                try:
                    function_data = call_data.get("function", {})
                    tool_name = function_data.get("name")
                    arguments_json = function_data.get("arguments", "{}")
                    
                    if tool_name:
                        arguments = json.loads(arguments_json) if isinstance(arguments_json, str) else arguments_json
                        
                        tool_calls.append(ToolCall(
                            tool_name=tool_name,
                            arguments=arguments,
                            call_id=call_data.get("id")
                        ))
                        logger.debug(f"Parsed structured tool call: {tool_name}")
                    
                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(f"Error parsing structured tool call: {e}")
        
        # Handle legacy function_call format
        elif "function_call" in output:
            function_data = output["function_call"]
            tool_name = function_data.get("name")
            arguments_json = function_data.get("arguments", "{}")
            
            if tool_name:
                try:
                    arguments = json.loads(arguments_json) if isinstance(arguments_json, str) else arguments_json
                    tool_calls.append(ToolCall(
                        tool_name=tool_name,
                        arguments=arguments,
                        call_id="legacy_call"
                    ))
                    logger.debug(f"Parsed legacy function call: {tool_name}")
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing legacy function call arguments: {e}")
        
        return tool_calls
    
    def _parse_function_arguments(self, args_str: str) -> Dict[str, Any]:
        """Parse function arguments from string format."""
        arguments = {}
        
        if not args_str.strip():
            return arguments
        
        try:
            # Try to parse as JSON first
            if args_str.strip().startswith('{') and args_str.strip().endswith('}'):
                return json.loads(args_str)
            
            # Parse key=value pairs
            for arg_pair in args_str.split(','):
                if '=' not in arg_pair:
                    continue
                
                key, value = arg_pair.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes and try to parse value
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                else:
                    # Try to convert to appropriate type
                    try:
                        if value.lower() in ('true', 'false'):
                            value = value.lower() == 'true'
                        elif value.isdigit():
                            value = int(value)
                        elif '.' in value and value.replace('.', '').isdigit():
                            value = float(value)
                    except ValueError:
                        pass  # Keep as string
                
                arguments[key] = value
        
        except Exception as e:
            logger.error(f"Error parsing arguments '{args_str}': {e}")
        
        return arguments
    
    async def execute_function_async(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool call asynchronously.
        
        Args:
            tool_call: The ToolCall object to execute
            
        Returns:
            ToolResult with execution details
        """
        start_time = datetime.now()
        
        try:
            if tool_call.tool_name not in self.available_tools:
                return ToolResult(
                    success=False,
                    result=None,
                    error_message=f"Tool '{tool_call.tool_name}' not found",
                    tool_name=tool_call.tool_name
                )
            
            tool = self.available_tools[tool_call.tool_name]
            logger.info(f"Executing tool '{tool_call.tool_name}' with args: {tool_call.arguments}")
            
            # Execute the tool
            if isinstance(tool, BaseTool):
                result = await tool.execute(**tool_call.arguments)
            else:
                # Handle regular function calls
                if asyncio.iscoroutinefunction(tool):
                    raw_result = await tool(**tool_call.arguments)
                else:
                    raw_result = tool(**tool_call.arguments)
                
                result = ToolResult(
                    success=True,
                    result=raw_result,
                    tool_name=tool_call.tool_name
                )
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            result.execution_time_ms = execution_time
            
            logger.info(f"Tool '{tool_call.tool_name}' executed successfully in {execution_time:.1f}ms")
            return result
        
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            error_msg = f"Error executing tool '{tool_call.tool_name}': {str(e)}"
            logger.error(error_msg)
            
            return ToolResult(
                success=False,
                result=None,
                error_message=error_msg,
                execution_time_ms=execution_time,
                tool_name=tool_call.tool_name
            )
    
    def execute_function(self, tool_call: ToolCall) -> ToolResult:
        """
        Synchronous wrapper for tool execution.
        
        Args:
            tool_call: The ToolCall object to execute
            
        Returns:
            ToolResult with execution details
        """
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.execute_function_async(tool_call))
        except RuntimeError:
            # No event loop running, create a new one
            return asyncio.run(self.execute_function_async(tool_call))


# --- Enhanced Example Tools ---

class WeatherTool(BaseTool):
    """Enhanced weather tool with proper error handling."""
    
    def __init__(self):
        super().__init__(
            name="get_weather",
            description="Get current weather information for a location"
        )
        self._weather_data = {
            "london": {"temperature": 15, "condition": "Cloudy", "humidity": 75},
            "tokyo": {"temperature": 28, "condition": "Sunny", "humidity": 60},
            "new york": {"temperature": 22, "condition": "Partly Cloudy", "humidity": 68},
            "paris": {"temperature": 18, "condition": "Rainy", "humidity": 85}
        }
    
    async def execute(self, location: str, unit: str = "celsius") -> ToolResult:
        """Execute weather lookup."""
        try:
            location_key = location.lower().strip()
            
            if location_key in self._weather_data:
                weather = self._weather_data[location_key].copy()
                weather["location"] = location
                weather["unit"] = unit
                
                # Convert temperature if needed
                if unit.lower() == "fahrenheit":
                    weather["temperature"] = (weather["temperature"] * 9/5) + 32
                
                return ToolResult(
                    success=True,
                    result=weather,
                    metadata={"source": "simulated", "timestamp": datetime.now().isoformat()}
                )
            else:
                return ToolResult(
                    success=False,
                    result=None,
                    error_message=f"Weather data not available for location: {location}"
                )
        
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error_message=f"Weather tool error: {str(e)}"
            )
    
    def get_schema(self) -> Dict[str, Any]:
        """Return the tool's parameter schema."""
        return {
            "type": "function",
            "description": self.description,
            "parameters": {
                "location": {
                    "type": "string",
                    "description": "The city or location to get weather for"
                },
                "unit": {
                    "type": "string",
                    "description": "Temperature unit (celsius or fahrenheit)",
                    "default": "celsius",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }


class CalculatorTool(BaseTool):
    """Enhanced calculator tool with comprehensive operations."""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations"
        )
    
    async def execute(self, operation: str, num1: float, num2: float = 0) -> ToolResult:
        """Execute calculation."""
        try:
            operation = operation.lower().strip()
            
            operations = {
                "add": lambda x, y: x + y,
                "subtract": lambda x, y: x - y,
                "multiply": lambda x, y: x * y,
                "divide": lambda x, y: x / y if y != 0 else None,
                "power": lambda x, y: x ** y,
                "modulo": lambda x, y: x % y if y != 0 else None
            }
            
            if operation not in operations:
                return ToolResult(
                    success=False,
                    result=None,
                    error_message=f"Unsupported operation: {operation}. Available: {list(operations.keys())}"
                )
            
            if operation in ["divide", "modulo"] and num2 == 0:
                return ToolResult(
                    success=False,
                    result=None,
                    error_message=f"Cannot {operation} by zero"
                )
            
            result = operations[operation](num1, num2)
            
            return ToolResult(
                success=True,
                result={
                    "operation": operation,
                    "operands": [num1, num2] if operation != "square" else [num1],
                    "result": result
                },
                metadata={"precision": "float64"}
            )
        
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error_message=f"Calculator error: {str(e)}"
            )
    
    def get_schema(self) -> Dict[str, Any]:
        """Return the tool's parameter schema."""
        return {
            "type": "function",
            "description": self.description,
            "parameters": {
                "operation": {
                    "type": "string",
                    "description": "Mathematical operation to perform",
                    "enum": ["add", "subtract", "multiply", "divide", "power", "modulo"]
                },
                "num1": {
                    "type": "number",
                    "description": "First number"
                },
                "num2": {
                    "type": "number",
                    "description": "Second number (optional for some operations)",
                    "default": 0
                }
            },
            "required": ["operation", "num1"]
        }


def get_default_tools() -> Dict[str, BaseTool]:
    """Get the default set of tools for AVA."""
    return {
        "get_weather": WeatherTool(),
        "calculator": CalculatorTool()
    }


# --- Test Functions ---
async def test_function_caller():
    """Test the enhanced function caller."""
    print("--- Enhanced Function Caller Test ---")
    
    # Initialize with default tools
    fc = FunctionCaller(get_default_tools())
    
    # Test 1: String-based tool call
    test_string = "I need to check the weather. [TOOL_CALL: get_weather(location=\"London\", unit=\"celsius\")] for my trip."
    print(f"\nTest 1 - String parsing: {test_string}")
    
    if fc.needs_function_call(test_string):
        tool_calls = fc.parse_function_calls(test_string)
        for tool_call in tool_calls:
            result = await fc.execute_function_async(tool_call)
            print(f"Result: {result}")
    
    # Test 2: JSON-based tool call
    test_json = {
        "text": "Let me calculate that for you.",
        "tool_calls": [{
            "id": "call_123",
            "type": "function", 
            "function": {
                "name": "calculator",
                "arguments": json.dumps({"operation": "multiply", "num1": 25, "num2": 4})
            }
        }]
    }
    print(f"\nTest 2 - JSON parsing: {test_json}")
    
    if fc.needs_function_call(test_json):
        tool_calls = fc.parse_function_calls(test_json)
        for tool_call in tool_calls:
            result = await fc.execute_function_async(tool_call)
            print(f"Result: {result}")
    
    # Test 3: Error handling
    error_call = ToolCall(tool_name="calculator", arguments={"operation": "divide", "num1": 10, "num2": 0})
    print(f"\nTest 3 - Error handling: Division by zero")
    result = await fc.execute_function_async(error_call)
    print(f"Result: {result}")
    
    print("\n--- Test Complete ---")


if __name__ == "__main__":
    asyncio.run(test_function_caller()) 