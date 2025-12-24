"""
AVA - Adaptive Virtual Agent
=============================

A research-grade AI assistant focused on accuracy over speed.

Core Components:
- Engine: The unified Cortex-Medulla brain
- Tools: Extensible tool system with MCP support
- Memory: Conversation context and learning
- Config: Centralized configuration

Quick Start:
    from ava import AVA

    async def main():
        ava = AVA()
        await ava.start()
        response = await ava.chat("Hello!")
        print(response.text)
"""

from .config import AVAConfig, load_config
from .engine import AVAEngine, EngineConfig
from .memory import ConversationMemory, MemoryConfig
from .tools import MCPClient, Tool, ToolManager

__version__ = "3.1.0"
__all__ = [
    "AVA",
    "AVAEngine",
    "EngineConfig",
    "ToolManager",
    "Tool",
    "MCPClient",
    "ConversationMemory",
    "MemoryConfig",
    "AVAConfig",
    "load_config",
]


class AVA:
    """
    Main AVA interface - simple and clean API.

    Example:
        ava = AVA()
        await ava.start()

        # Simple chat
        response = await ava.chat("What is 2 + 2?")
        print(response.text)

        # Force deep thinking
        response = await ava.think("Explain quantum entanglement")
        print(response.text)

        # Use specific tools
        response = await ava.chat("Search for Python tutorials", tools=["web_search"])
    """

    def __init__(self, config: AVAConfig = None):
        self.config = config or AVAConfig()
        self._engine = None
        self._tools = None
        self._memory = None
        self._started = False

    async def start(self) -> bool:
        """Initialize AVA. Returns True if successful."""
        if self._started:
            return True

        from .engine import AVAEngine
        from .memory import ConversationMemory
        from .tools import ToolManager

        self._engine = AVAEngine(self.config.engine)
        self._tools = ToolManager(self.config.tools)
        self._memory = ConversationMemory(self.config.memory)

        success = await self._engine.initialize()
        if success:
            self._started = True
        return success

    async def chat(self, message: str, **kwargs) -> "Response":
        """Send a message and get a response."""
        if not self._started:
            await self.start()
        return await self._engine.process(message, memory=self._memory, tools=self._tools, **kwargs)

    async def think(self, message: str, **kwargs) -> "Response":
        """Force deep thinking mode (Cortex)."""
        return await self.chat(message, force_cortex=True, **kwargs)

    async def stop(self):
        """Cleanup resources."""
        if self._engine:
            await self._engine.shutdown()
        self._started = False

    @property
    def is_running(self) -> bool:
        return self._started


class Response:
    """Response from AVA."""

    def __init__(
        self,
        text: str,
        used_cortex: bool = False,
        tools_used: list = None,
        thinking_time_ms: float = 0,
        confidence: float = 0.8,
        cognitive_state: str = "FLOW",
    ):
        self.text = text
        self.used_cortex = used_cortex
        self.tools_used = tools_used or []
        self.thinking_time_ms = thinking_time_ms
        self.confidence = confidence
        self.cognitive_state = cognitive_state

    def __str__(self):
        return self.text

    def __repr__(self):
        return f"Response(text='{self.text[:50]}...', cortex={self.used_cortex})"
