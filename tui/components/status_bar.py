"""
Status Bar Component
====================

Bottom status bar showing system state and quick stats.
"""

from datetime import datetime
from typing import Any

from textual.reactive import reactive
from textual.widgets import Static


class StatusBar(Static):
    """Status bar with system information."""

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 1;
        background: $surface-darken-1;
        color: $text-muted;
        padding: 0 1;
    }
    """

    # Reactive state
    system_state = reactive("running")
    cognitive_state = reactive("FLOW")
    active_component = reactive("medulla")
    interaction_count = reactive(0)
    response_time = reactive(0.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._start_time = datetime.now()

    def update_state(self, data: dict[str, Any]) -> None:
        """Update status from backend data."""
        if "system_state" in data:
            self.system_state = data["system_state"]
        if "cognitive_state" in data:
            cog = data["cognitive_state"]
            if isinstance(cog, dict):
                self.cognitive_state = cog.get("label", "FLOW")
            else:
                self.cognitive_state = str(cog)
        if "active_component" in data:
            self.active_component = data["active_component"]
        if "interaction_count" in data:
            self.interaction_count = data["interaction_count"]
        if "avg_response_time" in data:
            self.response_time = data["avg_response_time"]

    def render(self) -> str:
        """Render the status bar."""
        # Calculate uptime
        uptime = datetime.now() - self._start_time
        hours = int(uptime.total_seconds() // 3600)
        minutes = int((uptime.total_seconds() % 3600) // 60)
        uptime_str = f"{hours}h {minutes}m"

        # State indicator
        state_icon = "●" if self.system_state == "running" else "○"
        state_color = "green" if self.system_state == "running" else "yellow"

        # Component indicator
        comp_icon = "⚡" if self.active_component == "medulla" else "🧠"

        parts = [
            f"[{state_color}]{state_icon}[/] {self.system_state.upper()}",
            f"⏱ {uptime_str}",
            f"{comp_icon} {self.active_component}",
            f"🧠 {self.cognitive_state}",
            f"💬 {self.interaction_count}",
            f"⚡ {self.response_time:.0f}ms",
        ]

        return " │ ".join(parts)
