# AVA Terminal User Interface (TUI)

The AVA TUI is a full-featured terminal interface for power users, built with [Textual](https://textual.textualize.io/).

## Quick Start

```bash
# Start the TUI
python run_tui.py

# With options
python run_tui.py --backend http://localhost:8085
python run_tui.py --debug
python run_tui.py --no-connect  # Offline mode
```

## Layout

```
┌─────────────────────────────────────────────────────────────┐
│  AVA v3 - Neural Interface                      [Connected] │
├────────────────────────────────────────┬────────────────────┤
│                                        │ System Metrics     │
│  Chat Messages                         │ ──────────────     │
│  ─────────────                         │ Component: MEDULLA │
│                                        │ State: FLOW        │
│  [User] Hello, AVA!                    │ Entropy: 1.234     │
│                                        │ Varentropy: 0.567  │
│  [AVA] Hello! How can I help you       │ Confidence: 89%    │
│  today?                                │ Surprise: 0.234    │
│                                        │                    │
│  ┌─ Thinking ──────────────────────┐   │ Statistics         │
│  │      ◦  ·  ◦  ·                 │   │ ──────────         │
│  │     ╱ ─ · ─ ─ ╲                 │   │ Interactions: 42   │
│  │    │  ·  ◦  ·  │                │   │ Cortex Calls: 7    │
│  │     ╲ ─ ─ · ─ ╱                 │   │ Memories: 156      │
│  │      ◦  ·  ◦  ·                 │   │                    │
│  │   Generating response...        │   │                    │
│  └─────────────────────────────────┘   │                    │
│                                        │                    │
├────────────────────────────────────────┴────────────────────┤
│ Type your message... (Enter to send)                        │
├─────────────────────────────────────────────────────────────┤
│ ● RUNNING │ ⏱ 2h 15m │ ⚡ medulla │ 🧠 FLOW │ 💬 42 │ ⚡ 150ms│
└─────────────────────────────────────────────────────────────┘
```

## Keybindings

| Key | Action | Description |
|-----|--------|-------------|
| `Ctrl+K` | Command Palette | Open quick actions menu |
| `Ctrl+L` | Clear Chat | Clear all messages |
| `Ctrl+T` | Toggle Metrics | Show/hide metrics panel |
| `Ctrl+S` | Force Search | Enable search mode for next query |
| `Ctrl+D` | Deep Think | Force Cortex for next query |
| `F1` | Help | Show help screen |
| `Ctrl+Q` | Quit | Exit the TUI |
| `↑` / `↓` | History | Navigate command history |
| `Enter` | Send | Send current message |
| `Escape` | Close | Close overlays/popups |

## Components

### Chat Panel

The main chat area displays conversation history with:
- **User messages**: Your inputs with timestamp
- **Assistant messages**: AVA's responses with metadata
  - Cognitive state indicator (FLOW, HESITATION, CONFUSION)
  - Active component (Medulla or Cortex)
  - Response time

### Thinking Indicator

When AVA is processing, an animated ASCII brain shows:
- Current processing stage
- Visual neural activity animation
- Stage labels: Perceiving, Routing, Searching, Generating, Verifying

### Metrics Panel

Real-time system metrics including:
- **Active Component**: Currently active brain (Medulla/Cortex)
- **Cognitive State**: Current processing mode
  - FLOW: Normal operation
  - HESITATION: Uncertainty detected
  - CONFUSION: High entropy
  - CREATIVE: Exploratory mode
  - VERIFYING: Fact-checking
- **Entropy/Varentropy**: Information-theoretic metrics
- **Confidence**: Response certainty
- **Surprise**: Novelty of current input
- **Statistics**: Interaction counts, memory usage

### Status Bar

Bottom status bar showing:
- System state (RUNNING, PAUSED, SLEEPING)
- Uptime
- Active component
- Cognitive state
- Interaction count
- Average response time

### Input Box

Multi-line input with features:
- Command history (↑/↓ navigation)
- Auto-growing height
- Enter to send, Ctrl+Enter for new line

## Themes

The TUI uses a neural-themed color palette:

| Color | Variable | Usage |
|-------|----------|-------|
| `#00D4C8` | Primary | Highlights, borders |
| `#8B5CF6` | Secondary | Cortex indicators |
| `#00B4A8` | Accent | Focus states |
| `#0E0E14` | Surface | Background |
| `#F0F0F5` | Text | Main text |
| `#64647B` | Muted | Secondary text |
| `#10B981` | Success | Positive states |
| `#F59E0B` | Warning | Caution states |
| `#EF4444` | Error | Error states |

## Configuration

The TUI can be configured via command line arguments:

```bash
# Specify backend URL
python run_tui.py --backend http://192.168.1.100:8085

# Enable debug mode (verbose logging)
python run_tui.py --debug

# Start in offline mode (no backend connection)
python run_tui.py --no-connect
```

## Offline Mode

When running with `--no-connect`, the TUI operates without a backend:
- Chat messages are displayed but not sent
- Metrics show placeholder values
- Useful for testing UI or when backend is unavailable

## Extending the TUI

### Adding Custom Components

```python
# In tui/components/my_component.py
from textual.widgets import Static

class MyComponent(Static):
    DEFAULT_CSS = """
    MyComponent {
        border: solid $primary;
        padding: 1;
    }
    """

    def render(self) -> str:
        return "My custom content"
```

### Adding New Keybindings

```python
# In tui/app.py
BINDINGS = [
    # ... existing bindings
    Binding("ctrl+m", "my_action", "My Action", show=True),
]

def action_my_action(self) -> None:
    """Handle my custom action."""
    self.notify("Custom action triggered!")
```

### Custom Styles

Create a new stylesheet in `tui/styles/custom.tcss`:

```css
/* Custom theme */
$primary: #FF6B6B;
$secondary: #4ECDC4;

Screen {
    background: #1A1A2E;
}
```

## Troubleshooting

### TUI Not Starting

```bash
# Check Textual is installed
pip install textual>=0.40.0

# Run with debug output
python run_tui.py --debug
```

### Backend Connection Failed

```bash
# Verify backend is running
curl http://localhost:8085/health

# Start backend first
python server.py
```

### Display Issues

```bash
# Ensure terminal supports 256 colors
export TERM=xterm-256color

# Check terminal size (minimum 80x24)
stty size
```

### Input Not Working

- Ensure terminal is in raw mode
- Try a different terminal emulator
- Check for conflicting key bindings

## Architecture

```
tui/
├── __init__.py          # Package exports
├── app.py               # Main AVATUI application
├── components/
│   ├── __init__.py      # Component exports
│   ├── chat_panel.py    # Chat message display
│   ├── input_box.py     # User input handling
│   ├── metrics_panel.py # Real-time metrics
│   ├── status_bar.py    # Bottom status bar
│   └── thinking_indicator.py  # ASCII brain animation
└── styles/
    └── base.tcss        # Neural-themed styles
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install textual-dev

# Run TUI in dev mode
textual run --dev tui.app:AVATUI
```

### Console Mode

For debugging, run with console:

```bash
textual console
# In another terminal:
textual run --dev tui.app:AVATUI
```

This opens a debug console showing all events and messages.
