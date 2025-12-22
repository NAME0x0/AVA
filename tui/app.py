"""
AVA TUI - Main Application
==========================

The main Textual application for AVA's terminal interface.
"""

import asyncio
import logging
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Static, Input, RichLog
from textual.reactive import reactive

from .components.chat_panel import ChatPanel
from .components.input_box import ChatInput
from .components.metrics_panel import MetricsPanel
from .components.status_bar import StatusBar
from .components.thinking_indicator import ThinkingIndicator

logger = logging.getLogger(__name__)


class AVATUI(App):
    """
    AVA Terminal User Interface - Power User Mode.

    A full-featured terminal interface for interacting with AVA,
    featuring real-time metrics, command palette, and keyboard navigation.
    """

    CSS_PATH = "styles/base.tcss"
    TITLE = "AVA v3 - Neural Interface"
    SUB_TITLE = "Cortex-Medulla Architecture"

    BINDINGS = [
        Binding("ctrl+k", "command_palette", "Commands", show=True),
        Binding("ctrl+l", "clear_chat", "Clear", show=True),
        Binding("ctrl+t", "toggle_metrics", "Metrics", show=True),
        Binding("ctrl+s", "force_search", "Search", show=False),
        Binding("ctrl+d", "deep_think", "Think", show=False),
        Binding("f1", "help", "Help", show=True),
        Binding("escape", "close_overlay", "Close", show=False),
        Binding("ctrl+q", "quit", "Quit", show=True),
    ]

    # Reactive state
    connected = reactive(False)
    thinking = reactive(False)
    thinking_stage = reactive("idle")
    active_component = reactive("medulla")

    def __init__(self, backend_url: str = "http://localhost:8085"):
        super().__init__()
        self.backend_url = backend_url
        self.debug_mode = False
        self.offline_mode = False
        self._ava = None
        self._force_search = False
        self._force_cortex = False

    def compose(self) -> ComposeResult:
        """Create the UI layout."""
        yield Header()
        with Container(id="main"):
            with Horizontal(id="split-view"):
                with Vertical(id="chat-container"):
                    yield ChatPanel(id="chat")
                    yield ThinkingIndicator(id="thinking")
                    yield ChatInput(id="input")
                yield MetricsPanel(id="metrics")
        yield StatusBar(id="status")
        yield Footer()

    async def on_mount(self) -> None:
        """Initialize on mount."""
        self.title = "AVA v3"
        self.sub_title = "Connecting..."

        # Handle offline mode
        if self.offline_mode:
            self.sub_title = "Offline Mode"
            self.notify("Running in offline mode", severity="warning")
            return

        # Try to connect to backend
        await self._connect_backend()

        # Start polling for system state
        self.set_interval(2.0, self._poll_system_state)

    def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        """Handle chat input submission."""
        asyncio.create_task(self.send_message(event.value))

    async def _connect_backend(self) -> None:
        """Connect to AVA backend."""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.backend_url}/health", timeout=5.0)
                if response.status_code == 200:
                    self.connected = True
                    self.sub_title = "Connected"
                    self.notify("Connected to AVA backend", severity="information")
                else:
                    self.sub_title = "Connection failed"
                    self.notify("Backend not responding", severity="warning")
        except Exception as e:
            self.sub_title = "Offline"
            self.notify(f"Backend unavailable: {e}", severity="error")
            logger.warning(f"Could not connect to backend: {e}")

    async def _poll_system_state(self) -> None:
        """Poll backend for system state updates."""
        if not self.connected:
            return

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.backend_url}/status", timeout=2.0)
                if response.status_code == 200:
                    data = response.json()
                    # Update metrics panel
                    metrics = self.query_one("#metrics", MetricsPanel)
                    metrics.update_state(data)
        except Exception as e:
            logger.debug(f"Polling error: {e}")

    async def send_message(self, message: str) -> None:
        """Send a message to AVA."""
        if not message.strip():
            return

        chat = self.query_one("#chat", ChatPanel)
        thinking = self.query_one("#thinking", ThinkingIndicator)

        # Add user message
        chat.add_user_message(message)

        # Show thinking indicator
        self.thinking = True
        thinking.start()

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.backend_url}/chat",
                    json={"message": message},
                    timeout=300.0,  # 5 minute timeout for deep thinking
                )

                if response.status_code == 200:
                    data = response.json()
                    chat.add_assistant_message(
                        data.get("response", ""),
                        used_cortex=data.get("used_cortex", False),
                        cognitive_state=data.get("cognitive_state", "FLOW"),
                    )
                else:
                    chat.add_error_message(f"Error: {response.status_code}")

        except Exception as e:
            chat.add_error_message(f"Error: {str(e)}")
            logger.error(f"Send message error: {e}")
        finally:
            self.thinking = False
            thinking.stop()

    def action_command_palette(self) -> None:
        """Show command palette."""
        self.notify("Command palette coming soon!", severity="information")

    def action_toggle_metrics(self) -> None:
        """Toggle metrics panel visibility."""
        metrics = self.query_one("#metrics", MetricsPanel)
        metrics.display = not metrics.display

    def action_clear_chat(self) -> None:
        """Clear chat history."""
        chat = self.query_one("#chat", ChatPanel)
        chat.clear()
        self.notify("Chat cleared", severity="information")

    def action_force_search(self) -> None:
        """Force search mode for next query."""
        self._force_search = True
        self._force_cortex = False
        self.notify("Search mode enabled for next query", severity="information")

    def action_deep_think(self) -> None:
        """Force Cortex (deep thinking) for next query."""
        self._force_cortex = True
        self._force_search = False
        self.notify("Deep thinking enabled for next query", severity="information")

    def action_help(self) -> None:
        """Show help screen."""
        help_text = """
# AVA TUI Keybindings

| Key | Action |
|-----|--------|
| Ctrl+K | Command palette |
| Ctrl+L | Clear chat |
| Ctrl+T | Toggle metrics |
| Ctrl+S | Force search |
| Ctrl+D | Deep think |
| F1 | Help |
| Ctrl+Q | Quit |
        """
        self.notify(help_text, timeout=10)

    def action_close_overlay(self) -> None:
        """Close any open overlay."""
        pass  # Placeholder for future overlays


def main():
    """Entry point for TUI."""
    app = AVATUI()
    app.run()


if __name__ == "__main__":
    main()
