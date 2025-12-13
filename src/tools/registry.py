"""
Tool Registry for AVA

Central registry for all tools with safety level management.
Tools are gated by developmental stage to ensure age-appropriate access.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolSafetyLevel(IntEnum):
    """
    Safety levels for tool access.

    Higher levels require more developmental maturity.
    """
    LEVEL_0 = 0  # Baby-safe: echo, simple math, time
    LEVEL_1 = 1  # Toddler: calculator, word count
    LEVEL_2 = 2  # Child: web search, file reading
    LEVEL_3 = 3  # Adolescent: file writing, API calls
    LEVEL_4 = 4  # Young Adult: code execution
    LEVEL_5 = 5  # Mature: system commands, dangerous operations


@dataclass
class ToolDefinition:
    """
    Definition of a registered tool.

    Contains metadata, access requirements, and usage statistics.
    """
    name: str
    description: str
    safety_level: ToolSafetyLevel

    # The actual callable function
    function: Callable

    # Parameter schema
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_params: List[str] = field(default_factory=list)

    # Return type description
    return_type: str = "str"

    # Developmental requirements beyond safety level
    required_milestones: List[str] = field(default_factory=list)
    minimum_competencies: Dict[str, float] = field(default_factory=dict)

    # Usage tracking
    times_used: int = 0
    success_count: int = 0
    failure_count: int = 0
    last_used: Optional[datetime] = None

    # Learning benefits
    teaches_competencies: List[str] = field(default_factory=list)

    # Emotional impact of using this tool
    success_emotions: Dict[str, float] = field(default_factory=dict)
    failure_emotions: Dict[str, float] = field(default_factory=dict)

    # Example usage for LLM context
    example_usage: str = ""

    def record_use(self, success: bool):
        """Record a tool usage."""
        self.times_used += 1
        self.last_used = datetime.now()
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

    def get_success_rate(self) -> float:
        """Get the success rate for this tool."""
        if self.times_used == 0:
            return 0.0
        return self.success_count / self.times_used

    def to_prompt_description(self) -> str:
        """Generate a description for including in LLM prompts."""
        params_str = ", ".join(
            f"{name}: {info.get('type', 'any')}"
            for name, info in self.parameters.items()
        )
        return f"{self.name}({params_str}) - {self.description}"


@dataclass
class ToolAccessDecision:
    """Result of checking tool access."""
    allowed: bool
    tool_name: str
    reason: str
    missing_requirements: List[str] = field(default_factory=list)
    suggested_alternatives: List[str] = field(default_factory=list)


class ToolRegistry:
    """
    Central registry for all AVA tools.

    Manages tool registration, access control based on developmental
    stage, and usage statistics.
    """

    def __init__(self):
        """Initialize the tool registry."""
        self.tools: Dict[str, ToolDefinition] = {}
        self._register_default_tools()
        logger.info(f"ToolRegistry initialized with {len(self.tools)} tools")

    def _register_default_tools(self):
        """Register the default set of tools."""
        # Default tools are registered via base_tools.py
        pass

    def register(
        self,
        name: str,
        function: Callable,
        description: str,
        safety_level: ToolSafetyLevel,
        parameters: Optional[Dict[str, Any]] = None,
        required_params: Optional[List[str]] = None,
        **kwargs
    ) -> ToolDefinition:
        """
        Register a new tool.

        Args:
            name: Unique tool name
            function: The callable to execute
            description: Human-readable description
            safety_level: Required safety level
            parameters: Parameter schema
            required_params: Required parameter names
            **kwargs: Additional ToolDefinition attributes
        """
        tool = ToolDefinition(
            name=name,
            function=function,
            description=description,
            safety_level=safety_level,
            parameters=parameters or {},
            required_params=required_params or [],
            **kwargs
        )

        self.tools[name] = tool
        logger.debug(f"Registered tool: {name} (Level {safety_level})")

        return tool

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self.tools.get(name)

    def check_access(
        self,
        tool_name: str,
        current_stage: int,
        achieved_milestones: Optional[List[str]] = None,
        competencies: Optional[Dict[str, float]] = None,
    ) -> ToolAccessDecision:
        """
        Check if a tool can be accessed at the current developmental stage.

        Args:
            tool_name: Name of the tool
            current_stage: Current developmental stage value
            achieved_milestones: List of achieved milestone IDs
            competencies: Current competency scores

        Returns:
            ToolAccessDecision with access result and reasoning
        """
        tool = self.tools.get(tool_name)

        if tool is None:
            return ToolAccessDecision(
                allowed=False,
                tool_name=tool_name,
                reason=f"Tool '{tool_name}' not found",
            )

        missing = []

        # Check safety level
        if current_stage < tool.safety_level:
            missing.append(f"Stage {tool.safety_level} required (current: {current_stage})")

        # Check milestones
        achieved_milestones = achieved_milestones or []
        for milestone in tool.required_milestones:
            if milestone not in achieved_milestones:
                missing.append(f"Milestone '{milestone}' required")

        # Check competencies
        competencies = competencies or {}
        for comp, min_level in tool.minimum_competencies.items():
            current_level = competencies.get(comp, 0.0)
            if current_level < min_level:
                missing.append(f"Competency '{comp}' needs {min_level} (current: {current_level:.2f})")

        if missing:
            # Find alternative tools at lower levels
            alternatives = self.get_available_tools(current_stage)
            alt_names = [t.name for t in alternatives if t.name != tool_name]

            return ToolAccessDecision(
                allowed=False,
                tool_name=tool_name,
                reason="Requirements not met",
                missing_requirements=missing,
                suggested_alternatives=alt_names[:3],
            )

        return ToolAccessDecision(
            allowed=True,
            tool_name=tool_name,
            reason="Access granted",
        )

    def get_available_tools(
        self,
        max_safety_level: int,
        filter_by_competencies: Optional[Dict[str, float]] = None,
    ) -> List[ToolDefinition]:
        """
        Get all tools available at or below the given safety level.

        Args:
            max_safety_level: Maximum safety level to include
            filter_by_competencies: Optional competency filtering

        Returns:
            List of available tools
        """
        available = []

        for tool in self.tools.values():
            if tool.safety_level <= max_safety_level:
                # Check competencies if provided
                if filter_by_competencies:
                    meets_competencies = all(
                        filter_by_competencies.get(comp, 0) >= min_level
                        for comp, min_level in tool.minimum_competencies.items()
                    )
                    if not meets_competencies:
                        continue

                available.append(tool)

        return available

    def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        validate_access: bool = True,
        current_stage: int = 5,
    ) -> Dict[str, Any]:
        """
        Execute a tool with the given arguments.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
            validate_access: Whether to check access first
            current_stage: Current developmental stage

        Returns:
            Dict with 'success', 'result', and 'error' keys
        """
        tool = self.tools.get(tool_name)

        if tool is None:
            return {
                "success": False,
                "result": None,
                "error": f"Tool '{tool_name}' not found",
            }

        # Validate access if requested
        if validate_access:
            access = self.check_access(tool_name, current_stage)
            if not access.allowed:
                return {
                    "success": False,
                    "result": None,
                    "error": f"Access denied: {access.reason}",
                    "missing": access.missing_requirements,
                }

        # Validate required parameters
        for param in tool.required_params:
            if param not in arguments:
                return {
                    "success": False,
                    "result": None,
                    "error": f"Missing required parameter: {param}",
                }

        # Execute the tool
        try:
            result = tool.function(**arguments)
            tool.record_use(success=True)

            return {
                "success": True,
                "result": result,
                "error": None,
                "teaches": tool.teaches_competencies,
            }

        except Exception as e:
            tool.record_use(success=False)
            logger.error(f"Tool '{tool_name}' failed: {e}")

            return {
                "success": False,
                "result": None,
                "error": str(e),
            }

    def get_tools_for_prompt(
        self,
        max_safety_level: int,
        format: str = "simple",
    ) -> str:
        """
        Generate tool descriptions for inclusion in LLM prompts.

        Args:
            max_safety_level: Maximum safety level to include
            format: Output format ("simple", "detailed", "json")

        Returns:
            Formatted tool descriptions
        """
        tools = self.get_available_tools(max_safety_level)

        if format == "simple":
            lines = ["Available tools:"]
            for tool in tools:
                lines.append(f"- {tool.to_prompt_description()}")
            return "\n".join(lines)

        elif format == "detailed":
            lines = ["# Available Tools\n"]
            for tool in tools:
                lines.append(f"## {tool.name}")
                lines.append(f"**Description**: {tool.description}")
                lines.append(f"**Safety Level**: {tool.safety_level}")
                if tool.parameters:
                    lines.append("**Parameters**:")
                    for name, info in tool.parameters.items():
                        required = "(required)" if name in tool.required_params else "(optional)"
                        lines.append(f"  - {name}: {info.get('type', 'any')} {required}")
                if tool.example_usage:
                    lines.append(f"**Example**: {tool.example_usage}")
                lines.append("")
            return "\n".join(lines)

        elif format == "json":
            import json
            return json.dumps([
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                    "required": t.required_params,
                }
                for t in tools
            ], indent=2)

        return ""

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all tools."""
        stats = {
            "total_tools": len(self.tools),
            "by_level": {},
            "usage": {},
        }

        for level in ToolSafetyLevel:
            tools_at_level = [t for t in self.tools.values() if t.safety_level == level]
            stats["by_level"][level.name] = len(tools_at_level)

        for tool in self.tools.values():
            if tool.times_used > 0:
                stats["usage"][tool.name] = {
                    "times_used": tool.times_used,
                    "success_rate": tool.get_success_rate(),
                    "last_used": tool.last_used.isoformat() if tool.last_used else None,
                }

        return stats
