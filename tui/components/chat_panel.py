"""
Chat Panel Component
====================

Displays chat messages with markdown rendering and code highlighting.

Accessibility Features:
- Focusable with Tab navigation
- Arrow keys for message navigation
- Page Up/Down for scrolling
- Home/End to jump to top/bottom
"""

from datetime import datetime
from typing import Optional

from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from textual.binding import Binding
from textual.containers import ScrollableContainer
from textual.widgets import Static


class MessageWidget(Static):
    """A single chat message."""

    def __init__(
        self,
        content: str,
        role: str = "assistant",
        used_cortex: bool = False,
        cognitive_state: str = "FLOW",
        timestamp: Optional[datetime] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.content = content
        self.role = role
        self.used_cortex = used_cortex
        self.cognitive_state = cognitive_state
        self.timestamp = timestamp or datetime.now()

    def compose(self):
        """Render the message."""
        # Format based on role
        if self.role == "user":
            style = "bold cyan"
            prefix = "You"
        elif self.role == "error":
            style = "bold red"
            prefix = "Error"
        else:
            style = "bold green" if not self.used_cortex else "bold magenta"
            prefix = "AVA" if not self.used_cortex else "AVA [Cortex]"

        # Create header
        time_str = self.timestamp.strftime("%H:%M")
        header = Text()
        header.append(f"{prefix}", style=style)
        header.append(f" • {time_str}", style="dim")
        if self.role == "assistant" and self.cognitive_state != "FLOW":
            header.append(f" • {self.cognitive_state}", style="italic yellow")

        yield Static(header)

        # Render content as markdown
        try:
            md = Markdown(self.content)
            yield Static(md, classes="message-content")
        except Exception:
            yield Static(self.content, classes="message-content")


class ChatPanel(ScrollableContainer):
    """Panel displaying chat messages with scrolling."""

    # Enable focus for keyboard navigation
    can_focus = True

    # Keyboard bindings for accessibility
    BINDINGS = [
        Binding("up", "scroll_up", "Scroll up", show=False),
        Binding("down", "scroll_down", "Scroll down", show=False),
        Binding("j", "scroll_down", "Scroll down", show=False),  # Vim-style
        Binding("k", "scroll_up", "Scroll up", show=False),  # Vim-style
    ]

    DEFAULT_CSS = """
    ChatPanel {
        height: 1fr;
        border: solid $primary;
        border-title-color: $primary;
        padding: 1;
    }

    ChatPanel:focus {
        border: double $primary;
    }

    ChatPanel:focus-within {
        border: solid $accent;
    }

    ChatPanel > MessageWidget {
        margin-bottom: 1;
        padding: 1;
    }

    ChatPanel > .user-message {
        background: $surface-lighten-1;
    }

    ChatPanel > .assistant-message {
        background: $surface;
    }

    ChatPanel > .error-message {
        background: $error 20%;
    }

    .message-content {
        margin-left: 2;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.border_title = "Conversation"

    def action_scroll_up(self) -> None:
        """Scroll up by one line."""
        self.scroll_up(animate=False)

    def action_scroll_down(self) -> None:
        """Scroll down by one line."""
        self.scroll_down(animate=False)

    def add_user_message(self, content: str) -> None:
        """Add a user message to the chat."""
        widget = MessageWidget(
            content=content,
            role="user",
            classes="user-message",
        )
        self.mount(widget)
        self.scroll_end(animate=False)

    def add_assistant_message(
        self,
        content: str,
        used_cortex: bool = False,
        cognitive_state: str = "FLOW",
    ) -> None:
        """Add an assistant message to the chat."""
        widget = MessageWidget(
            content=content,
            role="assistant",
            used_cortex=used_cortex,
            cognitive_state=cognitive_state,
            classes="assistant-message",
        )
        self.mount(widget)
        self.scroll_end(animate=False)

    def add_error_message(self, content: str) -> None:
        """Add an error message to the chat."""
        widget = MessageWidget(
            content=content,
            role="error",
            classes="error-message",
        )
        self.mount(widget)
        self.scroll_end(animate=False)

    def clear(self) -> None:
        """Clear all messages."""
        self.remove_children()
