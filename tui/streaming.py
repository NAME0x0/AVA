"""
AVA TUI - Streaming Support
============================

WebSocket client for streaming responses from AVA backend.
Provides token-by-token display for better user experience.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class StreamEventType(Enum):
    """Types of streaming events."""
    TOKEN = "token"
    THINKING = "thinking"
    METRICS = "metrics"
    DONE = "done"
    ERROR = "error"


@dataclass
class StreamEvent:
    """A streaming event from the backend."""
    type: StreamEventType
    content: str = ""
    metadata: Optional[dict] = None


class StreamingClient:
    """WebSocket client for streaming AVA responses.
    
    Usage:
        client = StreamingClient("ws://localhost:8085/ws")
        async for event in client.stream("Hello, AVA!"):
            if event.type == StreamEventType.TOKEN:
                print(event.content, end="", flush=True)
            elif event.type == StreamEventType.DONE:
                print()
    """
    
    def __init__(self, ws_url: str = "ws://localhost:8085/ws"):
        """Initialize the streaming client.
        
        Args:
            ws_url: WebSocket endpoint URL
        """
        self.ws_url = ws_url
        self._ws = None
        self._connected = False

    async def connect(self) -> bool:
        """Establish WebSocket connection.
        
        Returns:
            True if connected successfully
        """
        try:
            import websockets
            self._ws = await websockets.connect(self.ws_url)
            self._connected = True
            logger.info(f"Connected to {self.ws_url}")
            return True
        except ImportError:
            logger.warning("websockets not installed, falling back to HTTP")
            return False
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        if self._ws:
            await self._ws.close()
            self._ws = None
            self._connected = False

    async def stream(
        self, 
        message: str,
        force_search: bool = False,
        force_cortex: bool = False,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a response from AVA.
        
        Args:
            message: User message
            force_search: Force search-first mode
            force_cortex: Force deep thinking mode
            
        Yields:
            StreamEvent objects as they arrive
        """
        if not self._connected:
            if not await self.connect():
                # Fallback to HTTP if WebSocket not available
                async for event in self._http_fallback(message, force_search, force_cortex):
                    yield event
                return

        try:
            # Send message
            request = {
                "message": message,
                "force_search": force_search,
                "force_cortex": force_cortex,
            }
            await self._ws.send(json.dumps(request))

            # Receive streaming response
            async for raw_msg in self._ws:
                try:
                    data = json.loads(raw_msg)
                    event_type = StreamEventType(data.get("type", "token"))
                    
                    yield StreamEvent(
                        type=event_type,
                        content=data.get("content", ""),
                        metadata=data.get("metadata"),
                    )
                    
                    if event_type == StreamEventType.DONE:
                        break
                        
                except json.JSONDecodeError:
                    # Plain text token
                    yield StreamEvent(type=StreamEventType.TOKEN, content=raw_msg)

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield StreamEvent(
                type=StreamEventType.ERROR,
                content=str(e),
            )

    async def _http_fallback(
        self,
        message: str,
        force_search: bool = False,
        force_cortex: bool = False,
    ) -> AsyncIterator[StreamEvent]:
        """Fallback to HTTP when WebSocket not available.
        
        This provides a simulated streaming experience by yielding
        the full response at once after receiving it.
        """
        try:
            import httpx
            
            # Yield thinking event
            yield StreamEvent(
                type=StreamEventType.THINKING,
                content="Processing...",
            )
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.ws_url.replace("ws://", "http://").replace("/ws", "/chat"),
                    json={
                        "message": message,
                        "force_search": force_search,
                        "force_cortex": force_cortex,
                    },
                    timeout=300.0,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data.get("response", "")
                    
                    # Simulate streaming by yielding chunks
                    chunk_size = 20  # characters per chunk
                    for i in range(0, len(content), chunk_size):
                        yield StreamEvent(
                            type=StreamEventType.TOKEN,
                            content=content[i:i + chunk_size],
                        )
                        # Small delay for visual effect
                        await asyncio.sleep(0.01)
                    
                    # Done event with metadata
                    yield StreamEvent(
                        type=StreamEventType.DONE,
                        metadata={
                            "used_cortex": data.get("used_cortex", False),
                            "cognitive_state": data.get("cognitive_state", "FLOW"),
                        },
                    )
                else:
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        content=f"HTTP {response.status_code}",
                    )
                    
        except Exception as e:
            yield StreamEvent(
                type=StreamEventType.ERROR,
                content=str(e),
            )


class StreamingWidget:
    """Helper for building streaming content in a widget.
    
    Used by ChatPanel to accumulate streaming tokens and
    update the display progressively.
    """
    
    def __init__(
        self,
        on_token: Callable[[str], None],
        on_complete: Callable[[str, dict], None],
    ):
        """Initialize the streaming widget helper.
        
        Args:
            on_token: Callback for each new token
            on_complete: Callback when streaming completes
        """
        self._buffer = []
        self._on_token = on_token
        self._on_complete = on_complete
        self._metadata = {}

    def append_token(self, token: str) -> None:
        """Add a token to the buffer and notify.
        
        Args:
            token: New token content
        """
        self._buffer.append(token)
        self._on_token(token)

    def complete(self, metadata: Optional[dict] = None) -> str:
        """Mark streaming as complete.
        
        Args:
            metadata: Final response metadata
            
        Returns:
            Complete response text
        """
        full_text = "".join(self._buffer)
        self._metadata = metadata or {}
        self._on_complete(full_text, self._metadata)
        return full_text

    @property
    def current_text(self) -> str:
        """Get current accumulated text."""
        return "".join(self._buffer)

    @property
    def token_count(self) -> int:
        """Get number of tokens received."""
        return len(self._buffer)
