"""
Thinking Indicator Component
============================

Stunning ASCII art animation showing when AVA is thinking.
Features a neural network visualization with synaptic activity.
"""

from textual.reactive import reactive
from textual.widgets import Static


# Detailed neural brain animation frames with synaptic activity
BRAIN_FRAMES = [
    """
[bold cyan]    ╭──────────────────────────────────────╮[/]
[cyan]    │[/]     [dim white]∴[/]    [bold magenta]◉[/]    [dim white]∴[/]     [bold cyan]◉[/]    [dim white]∴[/]     [cyan]│[/]
[cyan]    │[/]   [dim cyan]╱[/] [dim white]·[/] [bold cyan]━━━[/] [dim white]·[/] [bold magenta]━━━[/] [dim white]·[/] [dim cyan]╲[/]   [cyan]│[/]
[cyan]    │[/]  [bold cyan]◉[/][dim cyan]━━[/][dim white]∙[/][bold white]◯[/][dim white]∙[/][dim cyan]━━[/][bold magenta]◉[/][dim cyan]━━[/][dim white]∙[/][bold white]◯[/][dim white]∙[/][dim cyan]━━[/][bold cyan]◉[/]  [cyan]│[/]
[cyan]    │[/]   [dim cyan]╲[/] [dim white]·[/] [bold magenta]━━━[/] [dim white]·[/] [bold cyan]━━━[/] [dim white]·[/] [dim cyan]╱[/]   [cyan]│[/]
[cyan]    │[/]     [dim white]∴[/]    [bold cyan]◉[/]    [dim white]∴[/]     [bold magenta]◉[/]    [dim white]∴[/]     [cyan]│[/]
[bold cyan]    ╰──────────────────────────────────────╯[/]
""",
    """
[bold cyan]    ╭──────────────────────────────────────╮[/]
[cyan]    │[/]     [bold white]∴[/]    [dim magenta]◉[/]    [bold white]∴[/]     [dim cyan]◉[/]    [bold white]∴[/]     [cyan]│[/]
[cyan]    │[/]   [dim cyan]╱[/] [bold white]·[/] [dim cyan]━━━[/] [bold white]·[/] [dim magenta]━━━[/] [bold white]·[/] [dim cyan]╲[/]   [cyan]│[/]
[cyan]    │[/]  [dim cyan]◉[/][bold cyan]━━[/][bold white]∙[/][bold cyan]◯[/][bold white]∙[/][bold cyan]━━[/][dim magenta]◉[/][bold cyan]━━[/][bold white]∙[/][bold cyan]◯[/][bold white]∙[/][bold cyan]━━[/][dim cyan]◉[/]  [cyan]│[/]
[cyan]    │[/]   [dim cyan]╲[/] [bold white]·[/] [dim magenta]━━━[/] [bold white]·[/] [dim cyan]━━━[/] [bold white]·[/] [dim cyan]╱[/]   [cyan]│[/]
[cyan]    │[/]     [bold white]∴[/]    [dim cyan]◉[/]    [bold white]∴[/]     [dim magenta]◉[/]    [bold white]∴[/]     [cyan]│[/]
[bold cyan]    ╰──────────────────────────────────────╯[/]
""",
    """
[bold cyan]    ╭──────────────────────────────────────╮[/]
[cyan]    │[/]     [dim white]∴[/]    [bold cyan]◉[/]    [dim white]∴[/]     [bold magenta]◉[/]    [dim white]∴[/]     [cyan]│[/]
[cyan]    │[/]   [bold cyan]╱[/] [dim white]·[/] [bold cyan]━━━[/] [dim white]·[/] [bold magenta]━━━[/] [dim white]·[/] [bold cyan]╲[/]   [cyan]│[/]
[cyan]    │[/]  [bold magenta]◉[/][dim cyan]━━[/][bold white]∙[/][bold magenta]◯[/][bold white]∙[/][dim cyan]━━[/][bold cyan]◉[/][dim cyan]━━[/][bold white]∙[/][bold magenta]◯[/][bold white]∙[/][dim cyan]━━[/][bold magenta]◉[/]  [cyan]│[/]
[cyan]    │[/]   [bold cyan]╲[/] [dim white]·[/] [bold magenta]━━━[/] [dim white]·[/] [bold cyan]━━━[/] [dim white]·[/] [bold cyan]╱[/]   [cyan]│[/]
[cyan]    │[/]     [dim white]∴[/]    [bold magenta]◉[/]    [dim white]∴[/]     [bold cyan]◉[/]    [dim white]∴[/]     [cyan]│[/]
[bold cyan]    ╰──────────────────────────────────────╯[/]
""",
    """
[bold cyan]    ╭──────────────────────────────────────╮[/]
[cyan]    │[/]     [bold cyan]∴[/]    [bold white]◉[/]    [bold cyan]∴[/]     [bold white]◉[/]    [bold cyan]∴[/]     [cyan]│[/]
[cyan]    │[/]   [bold magenta]╱[/] [bold cyan]·[/] [bold white]━━━[/] [bold cyan]·[/] [bold white]━━━[/] [bold cyan]·[/] [bold magenta]╲[/]   [cyan]│[/]
[cyan]    │[/]  [bold white]◉[/][bold magenta]━━[/][bold cyan]∙[/][bold white]◯[/][bold cyan]∙[/][bold magenta]━━[/][bold white]◉[/][bold magenta]━━[/][bold cyan]∙[/][bold white]◯[/][bold cyan]∙[/][bold magenta]━━[/][bold white]◉[/]  [cyan]│[/]
[cyan]    │[/]   [bold magenta]╲[/] [bold cyan]·[/] [bold white]━━━[/] [bold cyan]·[/] [bold white]━━━[/] [bold cyan]·[/] [bold magenta]╱[/]   [cyan]│[/]
[cyan]    │[/]     [bold cyan]∴[/]    [bold white]◉[/]    [bold cyan]∴[/]     [bold white]◉[/]    [bold cyan]∴[/]     [cyan]│[/]
[bold cyan]    ╰──────────────────────────────────────╯[/]
""",
]

# Stage descriptions with icons
STAGES = {
    "idle": "",
    "perceiving": "👁  Perceiving input...",
    "routing": "🔀 Routing to processor...",
    "searching": "🔍 Searching for information...",
    "generating": "⚡ Generating response...",
    "verifying": "✓  Verifying accuracy...",
    "thinking": "🧠 Deep thinking...",
}

# Progress bar characters
PROGRESS_CHARS = ["▱", "▰"]


class ThinkingIndicator(Static):
    """
    Animated thinking indicator with neural network visualization.

    Features:
    - Animated ASCII neural network
    - Stage-aware status messages
    - Progress indication
    - Color-coded cognitive states
    """

    DEFAULT_CSS = """
    ThinkingIndicator {
        height: auto;
        padding: 1 2;
        display: none;
        background: $surface-lighten-1;
        border: round $primary;
        margin: 1 0;
    }

    ThinkingIndicator.active {
        display: block;
    }
    """

    # Reactive state
    is_active = reactive(False)
    stage = reactive("idle")
    frame_index = reactive(0)
    progress = reactive(0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._timer = None
        self._progress_timer = None

    def start(self, stage: str = "generating") -> None:
        """Start the thinking animation."""
        self.is_active = True
        self.stage = stage
        self.progress = 0
        self.add_class("active")

        # Start animation timer
        if self._timer is None:
            self._timer = self.set_interval(0.15, self._next_frame)

        # Start progress timer
        if self._progress_timer is None:
            self._progress_timer = self.set_interval(0.1, self._update_progress)

    def stop(self) -> None:
        """Stop the thinking animation."""
        self.is_active = False
        self.stage = "idle"
        self.progress = 0
        self.remove_class("active")

        # Stop timers
        if self._timer:
            self._timer.stop()
            self._timer = None
        if self._progress_timer:
            self._progress_timer.stop()
            self._progress_timer = None

    def _next_frame(self) -> None:
        """Advance to next animation frame."""
        self.frame_index = (self.frame_index + 1) % len(BRAIN_FRAMES)
        self.refresh()

    def _update_progress(self) -> None:
        """Update progress indicator."""
        self.progress = (self.progress + 1) % 20
        self.refresh()

    def _render_progress_bar(self) -> str:
        """Render an animated progress bar."""
        total = 20
        filled = self.progress % total
        bar = ""
        for i in range(total):
            if i == filled:
                bar += "[bold cyan]▰[/]"
            elif i < filled:
                bar += "[dim cyan]▰[/]"
            else:
                bar += "[dim white]▱[/]"
        return bar

    def render(self) -> str:
        """Render the thinking indicator."""
        if not self.is_active:
            return ""

        frame = BRAIN_FRAMES[self.frame_index]
        stage_text = STAGES.get(self.stage, "🧠 Thinking...")
        progress_bar = self._render_progress_bar()

        # Build the display
        output = f"{frame}\n"
        output += f"    [bold cyan]{stage_text}[/]\n"
        output += f"    {progress_bar}\n"

        return output
