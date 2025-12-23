"""
Metrics Panel Component
=======================

Real-time display of system metrics and cognitive state.

Accessibility Features:
- Focusable with Tab navigation
- Ctrl+3 quick access
- Screen reader friendly labels
"""

from typing import Any, Dict

from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static


class MetricsPanel(Static):
    """Real-time system metrics display."""

    # Enable focus for keyboard navigation
    can_focus = True

    DEFAULT_CSS = """
    MetricsPanel {
        width: 35;
        height: 100%;
        border: solid $secondary;
        border-title-color: $secondary;
        padding: 1;
    }

    MetricsPanel:focus {
        border: double $secondary;
    }
    """

    # Reactive properties
    cognitive_state = reactive("FLOW")
    entropy = reactive(0.0)
    varentropy = reactive(0.0)
    surprise = reactive(0.0)
    confidence = reactive(0.8)
    active_component = reactive("medulla")
    memory_count = reactive(0)
    interaction_count = reactive(0)
    cortex_invocations = reactive(0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.border_title = "System Metrics"

    def update_state(self, data: Dict[str, Any]) -> None:
        """Update metrics from backend data."""
        if "cognitive_state" in data:
            cog = data["cognitive_state"]
            if isinstance(cog, dict):
                self.cognitive_state = cog.get("label", "FLOW")
                self.entropy = cog.get("entropy", 0.0)
                self.varentropy = cog.get("varentropy", 0.0)
                self.confidence = cog.get("confidence", 0.8)
            else:
                self.cognitive_state = str(cog)

        if "surprise" in data:
            self.surprise = data["surprise"]

        if "active_component" in data:
            self.active_component = data["active_component"]

        if "memory_stats" in data:
            mem = data["memory_stats"]
            self.memory_count = mem.get("total_memories", 0)

        if "interaction_count" in data:
            self.interaction_count = data["interaction_count"]

        if "cortex_invocations" in data:
            self.cortex_invocations = data["cortex_invocations"]

    def render(self) -> str:
        """Render the metrics display."""
        # Active component indicator
        comp_color = "green" if self.active_component == "medulla" else "magenta"
        comp_icon = "⚡" if self.active_component == "medulla" else "🧠"

        # Cognitive state color
        state_colors = {
            "FLOW": "green",
            "HESITATION": "yellow",
            "CONFUSION": "red",
            "CREATIVE": "cyan",
            "VERIFYING": "blue",
        }
        state_color = state_colors.get(self.cognitive_state, "white")

        lines = [
            f"╭─ Active Component ─────────╮",
            f"│ {comp_icon} [{comp_color}]{self.active_component.upper():^22}[/] │",
            f"╰────────────────────────────╯",
            f"",
            f"╭─ Cognitive State ──────────╮",
            f"│ State:    [{state_color}]{self.cognitive_state:>14}[/] │",
            f"│ Entropy:        {self.entropy:>10.4f} │",
            f"│ Varentropy:     {self.varentropy:>10.4f} │",
            f"│ Confidence:     {self.confidence:>10.2%} │",
            f"│ Surprise:       {self.surprise:>10.4f} │",
            f"╰────────────────────────────╯",
            f"",
            f"╭─ Statistics ───────────────╮",
            f"│ Interactions:   {self.interaction_count:>10} │",
            f"│ Cortex Calls:   {self.cortex_invocations:>10} │",
            f"│ Memories:       {self.memory_count:>10} │",
            f"╰────────────────────────────╯",
        ]

        return "\n".join(lines)
