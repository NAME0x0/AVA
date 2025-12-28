"""
Tools Panel Component
=====================

Display and manage available tools for AVA.

Accessibility Features:
- Focusable with Tab navigation
- Ctrl+5 quick access
- Screen reader friendly labels
- Arrow key navigation between tools
"""

from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Static


class ToolsPanel(Static):
    """Display and manage available AVA tools."""

    # Enable focus for keyboard navigation
    can_focus = True

    DEFAULT_CSS = """
    ToolsPanel {
        width: 40;
        height: 100%;
        border: solid $success;
        border-title-color: $success;
        padding: 1;
        display: none;
    }

    ToolsPanel.visible {
        display: block;
    }

    ToolsPanel:focus {
        border: double $success;
    }
    """

    # Reactive state
    selected_index = reactive(0)

    # Built-in tools definition
    TOOLS = [
        {
            "name": "web_search",
            "icon": "🔍",
            "label": "Web Search",
            "description": "Search the web for information",
            "enabled": True,
        },
        {
            "name": "calculator",
            "icon": "🔢",
            "label": "Calculator",
            "description": "Perform mathematical calculations",
            "enabled": True,
        },
        {
            "name": "code_exec",
            "icon": "💻",
            "label": "Code Executor",
            "description": "Execute Python code safely",
            "enabled": False,
        },
        {
            "name": "file_reader",
            "icon": "📄",
            "label": "File Reader",
            "description": "Read local files",
            "enabled": False,
        },
        {
            "name": "memory",
            "icon": "🧠",
            "label": "Memory Store",
            "description": "Store and retrieve memories",
            "enabled": True,
        },
        {
            "name": "system_cmd",
            "icon": "⚙️",
            "label": "System Commands",
            "description": "Execute system commands",
            "enabled": False,
        },
    ]

    class ToolToggled(Message):
        """Message sent when a tool is toggled."""

        def __init__(self, name: str, enabled: bool):
            self.name = name
            self.enabled = enabled
            super().__init__()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.border_title = "Tools (Ctrl+5)"
        self._tool_states = {t["name"]: t["enabled"] for t in self.TOOLS}

    def toggle_current(self) -> None:
        """Toggle the currently selected tool."""
        tool = self.TOOLS[self.selected_index]
        name = tool["name"]

        # Some tools cannot be toggled for security
        if name in ("system_cmd", "code_exec"):
            return  # Blocked tools cannot be enabled via TUI

        self._tool_states[name] = not self._tool_states[name]
        self.post_message(self.ToolToggled(name, self._tool_states[name]))

    def move_selection(self, delta: int) -> None:
        """Move selection up or down."""
        new_index = (self.selected_index + delta) % len(self.TOOLS)
        self.selected_index = new_index

    def get_enabled_tools(self) -> list[str]:
        """Get list of enabled tool names."""
        return [name for name, enabled in self._tool_states.items() if enabled]

    def update_tools(self, tools_data: list[dict]) -> None:
        """Update tools from backend data."""
        for tool_data in tools_data:
            name = tool_data.get("name")
            if name in self._tool_states:
                self._tool_states[name] = tool_data.get("enabled", False)

    def render(self) -> str:
        """Render the tools panel."""
        lines = [
            "╭─ Available Tools ──────────────╮",
            "│ Use ↑↓ to select, Enter to    │",
            "│ toggle enabled/disabled       │",
            "╰────────────────────────────────╯",
            "",
        ]

        for i, tool in enumerate(self.TOOLS):
            name = tool["name"]
            enabled = self._tool_states.get(name, False)
            is_selected = i == self.selected_index

            # Status indicator
            if enabled:
                status = "[green]●[/]"
            else:
                status = "[red]○[/]"

            # Check if tool is locked
            is_locked = name in ("system_cmd", "code_exec")
            lock_icon = "🔒" if is_locked else "  "

            # Selection indicator
            prefix = "►" if is_selected else " "
            highlight = "reverse" if is_selected else ""

            icon = tool["icon"]
            label = tool["label"]

            lines.append(f"[{highlight}]{prefix} {icon} {label:<16} {status} {lock_icon}[/]")
            lines.append(f"    [dim]{tool['description']}[/]")
            lines.append("")

        # Summary
        enabled_count = sum(1 for e in self._tool_states.values() if e)
        total_count = len(self.TOOLS)

        lines.extend(
            [
                "╭─ Summary ──────────────────────╮",
                f"│ Enabled: {enabled_count}/{total_count} tools              │",
                "│ 🔒 = Security locked           │",
                "╰────────────────────────────────╯",
            ]
        )

        return "\n".join(lines)
