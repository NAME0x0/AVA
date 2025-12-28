"""
Chat Input Component
====================

Multi-line input with command history and auto-complete.

Accessibility Features:
- Auto-focused on app start
- Up/Down arrows for command history
- Clear placeholder text for screen readers
- Ctrl+1 quick access
"""

from textual.binding import Binding
from textual.message import Message
from textual.widgets import Input


class ChatInput(Input):
    """Chat input with history support."""

    # Keyboard bindings for history navigation
    BINDINGS = [
        Binding("up", "history_prev", "Previous history", show=False),
        Binding("down", "history_next", "Next history", show=False),
    ]

    DEFAULT_CSS = """
    ChatInput {
        dock: bottom;
        height: auto;
        min-height: 3;
        max-height: 10;
        border: solid $accent;
        padding: 1;
    }

    ChatInput:focus {
        border: double $primary;
        background: $surface-lighten-1;
    }
    """

    class Submitted(Message):
        """Message sent when user submits input."""

        def __init__(self, value: str):
            super().__init__()
            self.value = value

    def __init__(self, **kwargs):
        super().__init__(
            placeholder="Type your message... (Enter to send, Ctrl+Enter for new line)",
            **kwargs,
        )
        self.history = []
        self.history_index = -1

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        value = event.value.strip()
        if value:
            # Add to history
            if not self.history or self.history[-1] != value:
                self.history.append(value)
            self.history_index = -1

            # Clear input
            self.value = ""

            # Send message to app
            self.post_message(self.Submitted(value))

    def action_history_prev(self) -> None:
        """Navigate to previous history item."""
        if self.history and self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.value = self.history[-(self.history_index + 1)]

    def action_history_next(self) -> None:
        """Navigate to next history item."""
        if self.history_index > 0:
            self.history_index -= 1
            self.value = self.history[-(self.history_index + 1)]
        elif self.history_index == 0:
            self.history_index = -1
            self.value = ""
