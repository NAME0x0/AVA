"""
AVA Terminal User Interface (TUI)
=================================

A powerful, keyboard-driven terminal interface for AVA using Textual.

Features:
- Full chat interface with markdown support
- Real-time system metrics display
- Command palette (Ctrl+K)
- Keyboard-driven navigation
- Split-pane view
- ASCII thinking animations
- Conversation persistence (SQLite)
- Streaming response support (WebSocket/HTTP fallback)

Usage:
    python -m tui.app
    # or
    from tui import AVATUI
    app = AVATUI()
    app.run()
"""

from .app import AVATUI
from .persistence import ConversationStore, Message, Session
from .streaming import StreamEvent, StreamEventType, StreamingClient

__all__ = [
    "AVATUI",
    "ConversationStore",
    "Session",
    "Message",
    "StreamingClient",
    "StreamEvent",
    "StreamEventType",
]
