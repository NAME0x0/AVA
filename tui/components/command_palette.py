"""
AVA TUI Command Palette
=======================

A modal command palette with fuzzy search for quick command access.
Triggered by Ctrl+K.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static


@dataclass
class Command:
    """A command that can be executed from the palette."""

    id: str
    label: str
    description: str
    shortcut: str = ""
    category: str = "General"
    action: str = ""  # Action method name to call on app


# Available commands
COMMANDS = [
    Command("clear", "Clear Chat", "Clear all messages from chat", "Ctrl+L", "Chat", "clear_chat"),
    Command("search", "Force Search", "Use web search for next query", "Ctrl+S", "Mode", "force_search"),
    Command("think", "Deep Think", "Force Cortex deep thinking", "Ctrl+D", "Mode", "deep_think"),
    Command("metrics", "Toggle Metrics", "Show/hide metrics panel", "Ctrl+T", "View", "toggle_metrics"),
    Command("export", "Export Chat", "Save chat history to file", "", "Chat", "export_chat"),
    Command("help", "Help", "Show keyboard shortcuts", "F1", "Help", "help"),
    Command("quit", "Quit", "Exit application", "Ctrl+Q", "System", "quit"),
]


class CommandItem(Static):
    """A single command item in the palette list."""

    DEFAULT_CSS = """
    CommandItem {
        height: 3;
        padding: 0 2;
        background: transparent;
    }

    CommandItem:hover {
        background: $primary 15%;
    }

    CommandItem.-selected {
        background: $primary 25%;
        border-left: thick $primary;
    }
    """

    def __init__(self, command: Command, **kwargs) -> None:
        super().__init__(**kwargs)
        self.command = command

    def render(self) -> Text:
        """Render the command item with label, description, and shortcut."""
        text = Text()

        # Label (bold)
        text.append(self.command.label, style="bold")
        text.append("  ")

        # Description (dim)
        text.append(self.command.description, style="dim")

        # Shortcut (right-aligned, cyan)
        if self.command.shortcut:
            text.append("  ")
            text.append(f"[{self.command.shortcut}]", style="cyan dim")

        return text


class CommandPalette(ModalScreen[str | None]):
    """
    Modal command palette with fuzzy search.

    Press Ctrl+K to open, type to filter, Enter to execute.
    """

    BINDINGS = [
        Binding("escape", "close", "Close", show=False),
        Binding("up", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("enter", "execute", "Execute", show=False),
    ]

    DEFAULT_CSS = """
    CommandPalette {
        align: center top;
        padding-top: 5;
    }

    #palette-container {
        width: 60;
        max-width: 80%;
        height: auto;
        max-height: 70%;
        background: $surface;
        border: solid $primary;
        border-title-color: $primary;
        border-title-style: bold;
    }

    #palette-search {
        margin: 1 1 0 1;
        border: solid $accent;
    }

    #palette-search:focus {
        border: solid $primary;
    }

    #palette-results {
        height: auto;
        max-height: 15;
        margin: 1;
        overflow-y: auto;
    }

    #palette-hint {
        height: 1;
        margin: 0 1 1 1;
        text-align: center;
        color: $text-muted;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._commands = COMMANDS.copy()
        self._filtered_commands: list[Command] = COMMANDS.copy()
        self._selected_index = 0

    def compose(self) -> ComposeResult:
        """Compose the command palette UI."""
        with Container(id="palette-container"):
            yield Input(
                placeholder="Type a command...",
                id="palette-search",
            )
            with Vertical(id="palette-results"):
                for cmd in self._commands:
                    yield CommandItem(cmd, id=f"cmd-{cmd.id}")
            yield Static("↑↓ Navigate  Enter Select  Esc Close", id="palette-hint")

    def on_mount(self) -> None:
        """Focus the search input on mount."""
        self.query_one("#palette-search", Input).focus()
        self._update_selection()

    @on(Input.Changed, "#palette-search")
    def on_search_changed(self, event: Input.Changed) -> None:
        """Filter commands based on search input."""
        query = event.value.lower().strip()

        if not query:
            self._filtered_commands = self._commands.copy()
        else:
            # Fuzzy matching: check if query chars appear in order
            self._filtered_commands = []
            for cmd in self._commands:
                label_lower = cmd.label.lower()
                desc_lower = cmd.description.lower()

                # Exact match in label or description
                if query in label_lower or query in desc_lower:
                    self._filtered_commands.append(cmd)
                    continue

                # Initials match (e.g., "dt" matches "Deep Think")
                initials = "".join(w[0] for w in cmd.label.split()).lower()
                if query in initials:
                    self._filtered_commands.append(cmd)
                    continue

        # Update visibility
        results = self.query_one("#palette-results", Vertical)
        for child in results.children:
            if isinstance(child, CommandItem):
                visible = child.command in self._filtered_commands
                child.display = visible

        # Reset selection
        self._selected_index = 0
        self._update_selection()

    def _update_selection(self) -> None:
        """Update the visual selection state."""
        results = self.query_one("#palette-results", Vertical)
        visible_items = [
            c for c in results.children
            if isinstance(c, CommandItem) and c.display
        ]

        for i, item in enumerate(visible_items):
            if i == self._selected_index:
                item.add_class("-selected")
            else:
                item.remove_class("-selected")

    def action_move_up(self) -> None:
        """Move selection up."""
        if self._selected_index > 0:
            self._selected_index -= 1
            self._update_selection()

    def action_move_down(self) -> None:
        """Move selection down."""
        max_index = len(self._filtered_commands) - 1
        if self._selected_index < max_index:
            self._selected_index += 1
            self._update_selection()

    def action_execute(self) -> None:
        """Execute the selected command."""
        if self._filtered_commands and 0 <= self._selected_index < len(self._filtered_commands):
            command = self._filtered_commands[self._selected_index]
            self.dismiss(command.action)

    def action_close(self) -> None:
        """Close the palette without executing."""
        self.dismiss(None)
