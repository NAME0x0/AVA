"""
EXECUTIVE - High-Level Decision Making and Orchestration
=========================================================

The Executive module provides high-level coordination between AVA's
cognitive components, managing the flow of information between:
- The Medulla (reflexive processing)
- The Cortex (deep reasoning)
- The Hippocampus (memory systems)
- The Agency (action selection)

Key Responsibilities:
1. Route incoming requests to appropriate cognitive systems
2. Manage resource allocation between fast/slow pathways
3. Orchestrate multi-step reasoning chains
4. Maintain coherent context across interactions

This module acts as the "CEO" of AVA's cognitive architecture,
making strategic decisions about how to process each request.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ExecutiveState(Enum):
    """Operating states for the Executive."""

    IDLE = "idle"
    ROUTING = "routing"
    COORDINATING = "coordinating"
    MONITORING = "monitoring"
    ERROR = "error"


@dataclass
class ExecutiveConfig:
    """
    Configuration for the Executive module.

    Controls routing decisions, resource allocation, and
    coordination strategies.
    """

    # Routing thresholds
    complexity_threshold: float = 0.5  # Route to Cortex above this
    urgency_threshold: float = 0.7  # Fast-path for urgent requests

    # Resource management
    max_concurrent_tasks: int = 3
    timeout_seconds: float = 30.0

    # Coordination settings
    enable_cot: bool = True  # Chain-of-thought reasoning
    enable_tool_use: bool = True  # Tool augmentation
    enable_memory: bool = True  # Memory integration

    # Logging
    log_decisions: bool = True
    decision_log_path: str = "data/memory/executive_decisions.jsonl"

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "complexity_threshold": self.complexity_threshold,
            "urgency_threshold": self.urgency_threshold,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "timeout_seconds": self.timeout_seconds,
            "enable_cot": self.enable_cot,
            "enable_tool_use": self.enable_tool_use,
            "enable_memory": self.enable_memory,
            "log_decisions": self.log_decisions,
        }


@dataclass
class ExecutiveDecision:
    """Represents a routing/coordination decision."""

    route_to: str = "medulla"  # Target component
    priority: float = 0.5  # Task priority (0-1)
    use_tools: bool = False  # Enable tool use
    use_cot: bool = False  # Enable chain-of-thought
    query_memory: bool = False  # Query memory systems
    estimated_complexity: float = 0.0  # Estimated task complexity
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert decision to dictionary."""
        return {
            "route_to": self.route_to,
            "priority": self.priority,
            "use_tools": self.use_tools,
            "use_cot": self.use_cot,
            "query_memory": self.query_memory,
            "estimated_complexity": self.estimated_complexity,
            "timestamp": self.timestamp.isoformat(),
        }


class Executive:
    """
    Executive Module - High-Level Cognitive Orchestration.

    The Executive coordinates AVA's cognitive processes, making
    strategic decisions about how to handle each request. It
    acts as the "conductor" of the cognitive orchestra.

    Key Capabilities:
    - Request classification and routing
    - Resource allocation between fast/slow systems
    - Multi-step reasoning coordination
    - Context and state management
    """

    def __init__(
        self,
        config: ExecutiveConfig | None = None,
    ):
        """
        Initialize the Executive module.

        Args:
            config: Executive configuration
        """
        self.config = config or ExecutiveConfig()
        self.state = ExecutiveState.IDLE
        self.is_initialized = False
        self._active_tasks: list[str] = []

        logger.info("Executive module created")

    async def initialize(self) -> None:
        """Initialize the Executive module."""
        logger.info("Initializing Executive...")
        self.state = ExecutiveState.MONITORING
        self.is_initialized = True
        logger.info("Executive initialization complete")

    async def route_request(
        self,
        request: str,
        context: dict[str, Any] | None = None,
    ) -> ExecutiveDecision:
        """
        Route a request to the appropriate cognitive component.

        Args:
            request: The incoming request/query
            context: Optional context information

        Returns:
            ExecutiveDecision with routing information
        """
        self.state = ExecutiveState.ROUTING
        context = context or {}

        # Estimate complexity
        complexity = self._estimate_complexity(request, context)

        # Make routing decision
        decision = ExecutiveDecision(
            estimated_complexity=complexity,
        )

        if complexity > self.config.complexity_threshold:
            decision.route_to = "cortex"
            decision.use_cot = self.config.enable_cot
            decision.priority = 0.8
        else:
            decision.route_to = "medulla"
            decision.priority = 0.5

        # Check for tool requirements
        if self._needs_tools(request):
            decision.use_tools = self.config.enable_tool_use

        # Check for memory requirements
        if self._needs_memory(request, context):
            decision.query_memory = self.config.enable_memory

        self.state = ExecutiveState.IDLE

        if self.config.log_decisions:
            logger.info(f"Routed request to {decision.route_to}: {decision.to_dict()}")

        return decision

    def _estimate_complexity(
        self,
        request: str,
        context: dict[str, Any],
    ) -> float:
        """
        Estimate the complexity of a request.

        Simple heuristics for now - could be enhanced with ML.
        """
        complexity = 0.0

        # Length-based heuristic
        if len(request) > 200:
            complexity += 0.2

        # Question complexity
        question_words = ["why", "how", "explain", "analyze", "compare"]
        if any(word in request.lower() for word in question_words):
            complexity += 0.3

        # Multi-step indicators
        multi_step = ["then", "after", "first", "next", "finally"]
        if any(word in request.lower() for word in multi_step):
            complexity += 0.2

        # Context size
        if context.get("history_length", 0) > 5:
            complexity += 0.1

        return min(1.0, complexity)

    def _needs_tools(self, request: str) -> bool:
        """Check if request likely needs tool use."""
        tool_indicators = [
            "search",
            "find",
            "look up",
            "calculate",
            "what is",
            "current",
            "latest",
            "today",
        ]
        return any(ind in request.lower() for ind in tool_indicators)

    def _needs_memory(
        self,
        request: str,
        context: dict[str, Any],
    ) -> bool:
        """Check if request needs memory access."""
        memory_indicators = [
            "remember",
            "earlier",
            "before",
            "last time",
            "you said",
            "we discussed",
        ]
        return any(ind in request.lower() for ind in memory_indicators)

    async def shutdown(self) -> None:
        """Clean shutdown of the Executive."""
        logger.info("Shutting down Executive...")
        self.state = ExecutiveState.IDLE
        self.is_initialized = False
        logger.info("Executive shutdown complete")
