"""
Chat Panel Component
====================

Displays chat messages with markdown rendering and code highlighting.

Accessibility Features:
- Focusable with Tab navigation
- Arrow keys for message navigation
- Page Up/Down for scrolling
- Home/End to jump to top/bottom

Streaming Features:
- StreamingMessageWidget for real-time token display
- Blinking cursor animation during streaming

Syntax Highlighting:
- Full Pygments-powered syntax highlighting for code blocks
- Support for 100+ programming languages
- Custom theme optimized for terminal display
"""

from datetime import datetime
from typing import Any

from rich.markdown import Markdown
from rich.text import Text
from textual.binding import Binding
from textual.containers import ScrollableContainer
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import Static


class MessageWidget(Static):
    """A single chat message."""

    def __init__(
        self,
        content: str,
        role: str = "assistant",
        used_cortex: bool = False,
        cognitive_state: str = "FLOW",
        timestamp: datetime | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.content = content
        self.role = role
        self.used_cortex = used_cortex
        self.cognitive_state = cognitive_state
        self.timestamp = timestamp or datetime.now()

    def compose(self):
        """Render the message with syntax highlighting for code blocks."""
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

        # Render content as markdown with syntax highlighting
        # Rich's Markdown class automatically applies Pygments highlighting
        # to fenced code blocks (```python, ```javascript, etc.)
        try:
            md = Markdown(
                self.content,
                code_theme="monokai",  # Use monokai theme for code blocks
                inline_code_theme="monokai",
            )
            yield Static(md, classes="message-content")
        except Exception:
            yield Static(self.content, classes="message-content")


class StreamingMessageWidget(Static):
    """
    Message widget that updates as tokens arrive.

    Features:
    - Blinking cursor during streaming
    - Incremental content updates
    - Metadata display on completion
    """

    content = reactive("")
    is_complete = reactive(False)

    DEFAULT_CSS = """
    StreamingMessageWidget {
        padding: 1;
        margin-bottom: 1;
        background: $surface;
    }

    StreamingMessageWidget .streaming-cursor {
        background: $accent;
    }
    """

    def __init__(
        self,
        role: str = "assistant",
        used_cortex: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.role = role
        self.used_cortex = used_cortex
        self.cognitive_state = "FLOW"
        self.timestamp = datetime.now()
        self._cursor_timer: Timer | None = None
        self._cursor_visible = True
        self._metadata: dict[str, Any] = {}

    def on_mount(self) -> None:
        """Start cursor blink timer."""
        self._cursor_timer = self.set_interval(0.5, self._toggle_cursor)

    def _toggle_cursor(self) -> None:
        """Toggle cursor visibility for blink effect."""
        if not self.is_complete:
            self._cursor_visible = not self._cursor_visible
            self.refresh()

    def append_token(self, token: str) -> None:
        """Append a streaming token to the content."""
        self.content += token
        self.refresh()

    def complete(self, metadata: dict[str, Any] | None = None) -> None:
        """Mark streaming as complete with optional metadata."""
        self.is_complete = True
        if metadata:
            self._metadata = metadata
            self.used_cortex = metadata.get("used_cortex", self.used_cortex)
            self.cognitive_state = metadata.get("cognitive_state", self.cognitive_state)
        if self._cursor_timer:
            self._cursor_timer.stop()
        self.refresh()

    def render(self) -> Text:
        """Render the streaming message with cursor."""
        result = Text()

        # Header
        style = "bold green" if not self.used_cortex else "bold magenta"
        prefix = "AVA" if not self.used_cortex else "AVA [Cortex]"
        time_str = self.timestamp.strftime("%H:%M")

        result.append(f"{prefix}", style=style)
        result.append(f" • {time_str}", style="dim")

        if not self.is_complete:
            result.append(" • streaming...", style="italic cyan")
        elif self.cognitive_state != "FLOW":
            result.append(f" • {self.cognitive_state}", style="italic yellow")

        result.append("\n\n")

        # Content
        result.append(self.content)

        # Cursor (blink effect)
        if not self.is_complete and self._cursor_visible:
            result.append("█", style="cyan")

        return result

    def to_message_widget(self) -> MessageWidget:
        """Convert to a static MessageWidget after streaming completes."""
        return MessageWidget(
            content=self.content,
            role=self.role,
            used_cortex=self.used_cortex,
            cognitive_state=self.cognitive_state,
            timestamp=self.timestamp,
            classes="assistant-message",
        )


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

    def start_streaming_message(self) -> StreamingMessageWidget:
        """
        Start a new streaming message.

        Returns:
            StreamingMessageWidget that can be updated with append_token()
        """
        widget = StreamingMessageWidget(
            role="assistant",
            classes="assistant-message streaming",
        )
        self.mount(widget)
        self.scroll_end(animate=False)
        return widget

    def finalize_streaming_message(
        self,
        streaming_widget: StreamingMessageWidget,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Finalize a streaming message, replacing with Markdown-rendered widget.

        This enables full syntax highlighting for code blocks after streaming
        completes, providing better readability for code-heavy responses.

        Args:
            streaming_widget: The streaming widget to finalize
            metadata: Optional metadata (used_cortex, cognitive_state, etc.)
        """
        streaming_widget.complete(metadata)

        # Replace with static MessageWidget for full Markdown/syntax highlighting
        # This renders code blocks with proper syntax highlighting via Rich
        static_widget = streaming_widget.to_message_widget()
        streaming_widget.remove()
        self.mount(static_widget)

        self.scroll_end(animate=False)
