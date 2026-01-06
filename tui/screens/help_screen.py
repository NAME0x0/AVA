"""
Help Screen
===========

Modal screen displaying keyboard shortcuts and help information.
Press F1 or ? to open, Escape to close.
"""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Label, Static

HELP_CONTENT = """
╭───────────────────────────────────────────────────────────────────────────────╮
│                          AVA TUI - Keyboard Shortcuts                         │
╰───────────────────────────────────────────────────────────────────────────────╯

┌─────────────────────────────────────┬─────────────────────────────────────────┐
│         NAVIGATION                  │           CHAT SCROLLING                │
├─────────────────────────────────────┼─────────────────────────────────────────┤
│  Tab          Focus next panel      │  ↑/↓ or j/k    Scroll by line          │
│  Shift+Tab    Focus previous panel  │  Page Up/Down  Scroll by page          │
│  Ctrl+1       Focus input box       │  Home          Jump to top             │
│  Ctrl+2       Focus chat history    │  End           Jump to bottom          │
│  Ctrl+3       Focus metrics panel   │                                        │
│  Ctrl+4       Toggle settings       │                                        │
│  Ctrl+5       Toggle tools panel    │                                        │
└─────────────────────────────────────┴─────────────────────────────────────────┘

┌─────────────────────────────────────┬─────────────────────────────────────────┐
│         COMMANDS                    │           SPECIAL MODES                 │
├─────────────────────────────────────┼─────────────────────────────────────────┤
│  Ctrl+K       Command palette       │  Ctrl+S        Force search mode       │
│  Ctrl+L       Clear chat history    │  Ctrl+D        Deep think mode         │
│  Ctrl+T       Toggle metrics        │                (uses Cortex)           │
│  Ctrl+E       Export conversation   │                                        │
│  F1 or ?      Show this help        │                                        │
│  Escape       Close overlay         │                                        │
│  Ctrl+Q       Quit AVA              │                                        │
└─────────────────────────────────────┴─────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                              INPUT SHORTCUTS                                  │
├───────────────────────────────────────────────────────────────────────────────┤
│  Enter          Send message                                                  │
│  Shift+Enter    New line (multiline input)                                    │
│  Ctrl+C         Cancel current operation                                      │
│  ↑/↓            Navigate input history                                        │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                              COGNITIVE STATES                                 │
├───────────────────────────────────────────────────────────────────────────────┤
│  FLOW      Normal operation, quick responses                                  │
│  CURIOUS   Model is exploring, may ask clarifying questions                   │
│  FOCUSED   Deep analysis mode, longer processing                              │
│  CREATIVE  Generating novel content                                           │
│  CONFUSED  May need more context or clarification                             │
└───────────────────────────────────────────────────────────────────────────────┘

                              Press Escape or q to close
"""


class HelpScreen(ModalScreen[None]):
    """Modal screen showing keyboard shortcuts and help."""

    CSS = """
    HelpScreen {
        align: center middle;
    }

    HelpScreen > Container {
        width: 90%;
        height: 90%;
        max-width: 90;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    HelpScreen #help-title {
        text-align: center;
        text-style: bold;
        color: $primary;
        padding-bottom: 1;
    }

    HelpScreen #help-content {
        height: 1fr;
        overflow-y: auto;
    }

    HelpScreen #help-footer {
        text-align: center;
        color: $text-muted;
        padding-top: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
        Binding("q", "dismiss", "Close", show=False),
    ]

    def compose(self) -> ComposeResult:
        """Create the help screen layout."""
        with Container():
            yield Label("📚 AVA Help", id="help-title")
            yield Static(HELP_CONTENT, id="help-content")
            yield Label(
                "Press [bold cyan]Escape[/bold cyan] or [bold cyan]q[/bold cyan] to close",
                id="help-footer",
            )

    def action_dismiss(self) -> None:
        """Close the help screen."""
        self.dismiss()
