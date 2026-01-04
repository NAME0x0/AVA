"""
Settings Panel Component
========================

Interactive settings panel for configuring AVA behavior.

Accessibility Features:
- Focusable with Tab navigation
- Ctrl+4 quick access
- Screen reader friendly labels
- Arrow key navigation between options
"""

from pathlib import Path
from typing import Any

import yaml
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Static


class SettingsPanel(Static):
    """Interactive settings configuration panel."""

    # Enable focus for keyboard navigation
    can_focus = True

    DEFAULT_CSS = """
    SettingsPanel {
        width: 40;
        height: 100%;
        border: solid $primary;
        border-title-color: $primary;
        padding: 1;
        display: none;
    }

    SettingsPanel.visible {
        display: block;
    }

    SettingsPanel:focus {
        border: double $primary;
    }
    """

    # Reactive settings
    simulation_mode = reactive(True)
    search_first = reactive(True)
    thermal_limit = reactive(75)
    cortex_threshold = reactive(0.7)
    selected_index = reactive(0)

    # Settings definition
    SETTINGS = [
        {
            "key": "simulation_mode",
            "label": "Simulation Mode",
            "description": "Run without real models",
            "type": "bool",
        },
        {
            "key": "search_first",
            "label": "Search First",
            "description": "Web search for questions",
            "type": "bool",
        },
        {
            "key": "thermal_limit",
            "label": "Thermal Limit",
            "description": "GPU temperature limit (C)",
            "type": "int",
            "min": 60,
            "max": 90,
            "step": 5,
        },
        {
            "key": "cortex_threshold",
            "label": "Cortex Threshold",
            "description": "Surprise level to trigger",
            "type": "float",
            "min": 0.3,
            "max": 0.9,
            "step": 0.1,
        },
    ]

    class SettingChanged(Message):
        """Message sent when a setting is changed."""

        def __init__(self, key: str, value):
            self.key = key
            self.value = value
            super().__init__()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.border_title = "Settings (Ctrl+4)"
        self._load_settings()

    def _load_settings(self) -> None:
        """Load settings from config file."""
        config_path = Path("config/cortex_medulla.yaml")
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f) or {}

                # Extract settings
                dev = config.get("development", {})
                self.simulation_mode = dev.get("simulation_mode", True)

                search = config.get("search_first", {})
                self.search_first = search.get("enabled", True)

                thermal = config.get("thermal", {})
                self.thermal_limit = thermal.get("warning_temp", 75)

                medulla = config.get("medulla", {})
                self.cortex_threshold = medulla.get("high_surprise_threshold", 0.7)
            except Exception:
                pass  # Use defaults on error

    def _save_settings(self) -> None:
        """Save settings to config file."""
        config_path = Path("config/cortex_medulla.yaml")

        # Load existing config
        config: dict[str, Any] = {}
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f) or {}
            except Exception:
                pass

        # Update config
        if "development" not in config:
            config["development"] = {}
        config["development"]["simulation_mode"] = self.simulation_mode

        if "search_first" not in config:
            config["search_first"] = {}
        config["search_first"]["enabled"] = self.search_first

        if "thermal" not in config:
            config["thermal"] = {}
        config["thermal"]["warning_temp"] = self.thermal_limit

        if "medulla" not in config:
            config["medulla"] = {}
        config["medulla"]["high_surprise_threshold"] = self.cortex_threshold

        # Write config
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
        except Exception:
            pass

    def _get_value(self, key: str):
        """Get current value for a setting key."""
        return getattr(self, key, None)

    def _set_value(self, key: str, value) -> None:
        """Set value for a setting key."""
        setattr(self, key, value)
        self.post_message(self.SettingChanged(key, value))
        self._save_settings()

    def toggle_current(self) -> None:
        """Toggle or increment the currently selected setting."""
        setting = self.SETTINGS[self.selected_index]
        key = setting["key"]
        current = self._get_value(key)

        if setting["type"] == "bool":
            self._set_value(key, not current)
        elif setting["type"] in ("int", "float"):
            step = setting.get("step", 1)
            max_val = setting.get("max", 100)
            new_val = current + step
            if new_val > max_val:
                new_val = setting.get("min", 0)
            self._set_value(key, new_val)

    def decrement_current(self) -> None:
        """Decrement the currently selected setting."""
        setting = self.SETTINGS[self.selected_index]
        key = setting["key"]
        current = self._get_value(key)

        if setting["type"] == "bool":
            self._set_value(key, not current)
        elif setting["type"] in ("int", "float"):
            step = setting.get("step", 1)
            min_val = setting.get("min", 0)
            new_val = current - step
            if new_val < min_val:
                new_val = setting.get("max", 100)
            self._set_value(key, new_val)

    def move_selection(self, delta: int) -> None:
        """Move selection up or down."""
        new_index = (self.selected_index + delta) % len(self.SETTINGS)
        self.selected_index = new_index

    def render(self) -> str:
        """Render the settings panel."""
        lines = [
            "╭─ Configuration ────────────────╮",
            "│ Use ↑↓ to select, Enter/←→    │",
            "│ to change values              │",
            "╰────────────────────────────────╯",
            "",
        ]

        for i, setting in enumerate(self.SETTINGS):
            key = setting["key"]
            label = setting["label"]
            value = self._get_value(key)
            is_selected = i == self.selected_index

            # Format value display
            if setting["type"] == "bool":
                val_str = "[green]ON [/]" if value else "[red]OFF[/]"
            elif setting["type"] == "int":
                val_str = f"{value:>3}"
            else:
                val_str = f"{value:.1f}"

            # Selection indicator
            prefix = "►" if is_selected else " "
            highlight = "reverse" if is_selected else ""

            lines.append(f"[{highlight}]{prefix} {label:<18} {val_str:>8}[/]")
            lines.append(f"  [dim]{setting['description']}[/]")
            lines.append("")

        lines.extend(
            [
                "╭─ Actions ──────────────────────╮",
                "│ [S] Save    [R] Reset defaults │",
                "│ [Esc] Close panel              │",
                "╰────────────────────────────────╯",
            ]
        )

        return "\n".join(lines)
