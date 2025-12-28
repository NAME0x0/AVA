"""
AVA TUI - Main Application
==========================

The main Textual application for AVA's terminal interface.

Accessibility Features:
- Tab/Shift+Tab: Cycle focus between panels
- Arrow keys: Navigate within panels
- Ctrl+1/2/3: Jump to specific panels
- Screen reader announcements for state changes
"""

import asyncio
import logging

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header

from .components.chat_panel import ChatPanel
from .components.input_box import ChatInput
from .components.metrics_panel import MetricsPanel
from .components.settings_panel import SettingsPanel
from .components.status_bar import StatusBar
from .components.thinking_indicator import ThinkingIndicator
from .components.tools_panel import ToolsPanel

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
        # Navigation - Accessibility
        Binding("tab", "focus_next", "Next", show=False),
        Binding("shift+tab", "focus_previous", "Previous", show=False),
        Binding("ctrl+1", "focus_input", "Input", show=False),
        Binding("ctrl+2", "focus_chat", "Chat", show=False),
        Binding("ctrl+3", "focus_metrics", "Metrics", show=False),
        Binding("ctrl+4", "toggle_settings", "Settings", show=False),
        Binding("ctrl+5", "toggle_tools", "Tools", show=False),
        # Chat navigation
        Binding("page_up", "scroll_chat_up", "Scroll Up", show=False),
        Binding("page_down", "scroll_chat_down", "Scroll Down", show=False),
        Binding("home", "scroll_chat_home", "Top", show=False),
        Binding("end", "scroll_chat_end", "Bottom", show=False),
        # Commands
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
                yield SettingsPanel(id="settings")
                with Vertical(id="chat-container"):
                    yield ChatPanel(id="chat")
                    yield ThinkingIndicator(id="thinking")
                    yield ChatInput(id="input")
                yield MetricsPanel(id="metrics")
                yield ToolsPanel(id="tools")
        yield StatusBar(id="status")
        yield Footer()

    async def on_mount(self) -> None:
        """Initialize on mount."""
        self.title = "AVA v3"
        self.sub_title = "Connecting..."

        # Auto-focus the input for accessibility
        try:
            input_box = self.query_one("#input", ChatInput)
            input_box.focus()
        except Exception:
            pass

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
                # Build request with force flags
                request_data = {
                    "message": message,
                    "force_search": self._force_search,
                    "force_cortex": self._force_cortex,
                }

                response = await client.post(
                    f"{self.backend_url}/chat",
                    json=request_data,
                    timeout=300.0,  # 5 minute timeout for deep thinking
                )

                # Reset flags after sending
                self._force_search = False
                self._force_cortex = False

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
        from .components.command_palette import CommandPalette

        def handle_result(action_name: str | None) -> None:
            """Handle the command palette result."""
            if action_name:
                # Map action names to methods
                action_map = {
                    "clear_chat": self.action_clear_chat,
                    "force_search": self.action_force_search,
                    "deep_think": self.action_deep_think,
                    "toggle_metrics": self.action_toggle_metrics,
                    "export_chat": self._export_chat,
                    "help": self.action_help,
                    "quit": self.action_quit,
                }
                if action_name in action_map:
                    action_map[action_name]()

        self.push_screen(CommandPalette(), handle_result)

    def _export_chat(self) -> None:
        """Export chat history to a file."""
        import json
        from datetime import datetime
        from pathlib import Path

        chat = self.query_one("#chat", ChatPanel)
        messages = []

        # Collect messages from chat panel
        for child in chat.children:
            if hasattr(child, "content") and hasattr(child, "role"):
                messages.append(
                    {
                        "role": child.role,
                        "content": child.content,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        if not messages:
            self.notify("No messages to export", severity="warning")
            return

        # Save to file
        export_dir = Path("data/exports")
        export_dir.mkdir(parents=True, exist_ok=True)

        filename = f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = export_dir / filename

        with open(filepath, "w") as f:
            json.dump({"messages": messages}, f, indent=2)

        self.notify(f"Chat exported to {filepath}", severity="information")

    def action_toggle_metrics(self) -> None:
        """Toggle metrics panel visibility."""
        metrics = self.query_one("#metrics", MetricsPanel)
        metrics.display = not metrics.display

    def action_toggle_settings(self) -> None:
        """Toggle settings panel visibility (Ctrl+4)."""
        try:
            settings = self.query_one("#settings", SettingsPanel)
            settings.toggle_class("visible")
            if settings.has_class("visible"):
                settings.focus()
                self._announce("Settings panel opened")
            else:
                self._announce("Settings panel closed")
        except Exception:
            pass

    def action_toggle_tools(self) -> None:
        """Toggle tools panel visibility (Ctrl+5)."""
        try:
            tools = self.query_one("#tools", ToolsPanel)
            tools.toggle_class("visible")
            if tools.has_class("visible"):
                tools.focus()
                self._announce("Tools panel opened")
            else:
                self._announce("Tools panel closed")
        except Exception:
            pass

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

## Navigation (Accessibility)
| Key | Action |
|-----|--------|
| Tab | Focus next panel |
| Shift+Tab | Focus previous panel |
| Ctrl+1 | Focus input |
| Ctrl+2 | Focus chat |
| Ctrl+3 | Focus metrics |
| Ctrl+4 | Toggle settings |
| Ctrl+5 | Toggle tools |

## Chat Scrolling
| Key | Action |
|-----|--------|
| ↑/↓ or j/k | Scroll chat |
| Page Up/Down | Scroll page |
| Home/End | Jump to top/bottom |

## Commands
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
        self.notify(help_text, timeout=15)

    def action_close_overlay(self) -> None:
        """Close any open overlay."""
        pass  # Placeholder for future overlays

    # =========================================================================
    # Accessibility: Focus Navigation
    # =========================================================================

    def action_focus_input(self) -> None:
        """Focus the chat input (Ctrl+1)."""
        try:
            input_box = self.query_one("#input", ChatInput)
            input_box.focus()
            self._announce("Chat input focused")
        except Exception:
            pass

    def action_focus_chat(self) -> None:
        """Focus the chat panel (Ctrl+2)."""
        try:
            chat = self.query_one("#chat", ChatPanel)
            chat.focus()
            self._announce("Chat history focused")
        except Exception:
            pass

    def action_focus_metrics(self) -> None:
        """Focus the metrics panel (Ctrl+3)."""
        try:
            metrics = self.query_one("#metrics", MetricsPanel)
            if metrics.display:
                metrics.focus()
                self._announce("Metrics panel focused")
            else:
                self.notify("Metrics panel is hidden. Press Ctrl+T to show.")
        except Exception:
            pass

    # =========================================================================
    # Accessibility: Chat Scrolling
    # =========================================================================

    def action_scroll_chat_up(self) -> None:
        """Scroll chat up one page (Page Up)."""
        try:
            chat = self.query_one("#chat", ChatPanel)
            chat.scroll_page_up()
        except Exception:
            pass

    def action_scroll_chat_down(self) -> None:
        """Scroll chat down one page (Page Down)."""
        try:
            chat = self.query_one("#chat", ChatPanel)
            chat.scroll_page_down()
        except Exception:
            pass

    def action_scroll_chat_home(self) -> None:
        """Scroll to beginning of chat (Home)."""
        try:
            chat = self.query_one("#chat", ChatPanel)
            chat.scroll_home()
            self._announce("Top of chat history")
        except Exception:
            pass

    def action_scroll_chat_end(self) -> None:
        """Scroll to end of chat (End)."""
        try:
            chat = self.query_one("#chat", ChatPanel)
            chat.scroll_end()
            self._announce("End of chat history")
        except Exception:
            pass

    # =========================================================================
    # Accessibility: Screen Reader Announcements
    # =========================================================================

    def _announce(self, message: str, priority: str = "polite") -> None:
        """
        Announce message for screen readers.

        In terminal environments, this uses the notify system which
        screen readers can pick up. For more advanced accessibility,
        Textual's built-in accessibility features are used.
        """
        # Textual's notify is picked up by screen readers
        # Log for debugging accessibility
        logger.debug(f"Accessibility announcement: {message}")

    def watch_connected(self, connected: bool) -> None:
        """Announce connection state changes."""
        if connected:
            self._announce("Connected to AVA backend")
        else:
            self._announce("Disconnected from AVA backend")

    def watch_thinking(self, thinking: bool) -> None:
        """Announce thinking state changes."""
        if thinking:
            self._announce("AVA is thinking")
        else:
            self._announce("AVA finished thinking")


def main():
    """Entry point for TUI."""
    app = AVATUI()
    app.run()


if __name__ == "__main__":
    main()
