# AVA Terminal User Interface (TUI)

> **Version**: 4.2.3  
> Built with [Textual](https://textual.textualize.io/)

The AVA TUI provides a powerful command-line interface for power users who prefer keyboard-driven workflows.

---

## Quick Start

```bash
# Basic launch
python run_tui.py

# With options
python run_tui.py --backend http://localhost:8085
python run_tui.py --debug
python run_tui.py --no-connect  # Offline mode
```

---

## Interface Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  AVA v4.2 - Neural Interface                        [Connected] │
├──────────────────────┬──────────────────────────────────────────┤
│                      │                                          │
│     Settings         │              Chat Panel                  │
│       Panel          │                                          │
│                      │  You • 10:30                             │
│  Simulation Mode: ✓  │  Hello, AVA!                             │
│  Search First: ✓     │                                          │
│  Cortex Threshold:   │  AVA • 10:30 • FLOW                      │
│        0.7           │  Hello! How can I help you today?       │
│                      │                                          │
│                      │  ┌─ Thinking ───────────────────────┐   │
│                      │  │      ◦  ·  ◦  ·                  │   │
│                      │  │     ╱ ─ · ─ ─ ╲                  │   │
│                      │  │    │  ·  ◦  ·  │  Generating...  │   │
│                      │  │     ╲ ─ ─ · ─ ╱                  │   │
│                      │  └──────────────────────────────────┘   │
├──────────────────────┼──────────────────────────────────────────┤
│   System Metrics     │              Tools Panel                 │
│                      │                                          │
│  Entropy: 0.45       │  🧮 Calculator                           │
│  Surprise: 0.23      │  🔍 Web Search                           │
│  Varentropy: 0.12    │  📅 Date/Time                            │
│  Confidence: 0.87    │                                          │
└──────────────────────┴──────────────────────────────────────────┘
│ ● RUNNING │ ⏱ 1h 23m │ ⚡ medulla │ 🧠 FLOW │ 💬 42 │ ⚡ 234ms │
└─────────────────────────────────────────────────────────────────┘
```

---

## Keyboard Shortcuts

### Navigation

| Key | Action | Description |
|-----|--------|-------------|
| `Tab` | Focus Next | Cycle to next panel |
| `Shift+Tab` | Focus Previous | Cycle to previous panel |
| `Ctrl+1` | Focus Input | Jump to chat input |
| `Ctrl+2` | Focus Chat | Jump to chat history |
| `Ctrl+3` | Focus Metrics | Jump to metrics panel |
| `Ctrl+4` | Toggle Settings | Show/hide settings |
| `Ctrl+5` | Toggle Tools | Show/hide tools |

### Chat Scrolling

| Key | Action |
|-----|--------|
| `↑` / `↓` or `j` / `k` | Scroll by line |
| `Page Up` / `Page Down` | Scroll by page |
| `Home` | Jump to top |
| `End` | Jump to bottom |

### Commands

| Key | Action |
|-----|--------|
| `Ctrl+K` | Command palette |
| `Ctrl+L` | Clear chat history |
| `Ctrl+T` | Toggle metrics panel |
| `Ctrl+S` | Force search mode |
| `Ctrl+D` | Deep think (Cortex) |
| `Ctrl+E` | Export conversation |
| `Ctrl+M` | Cycle models |
| `F1` or `?` | Show help |
| `Escape` | Close overlay |
| `Ctrl+Q` | Quit |

### Input

| Key | Action |
|-----|--------|
| `Enter` | Send message |
| `Shift+Enter` | New line |
| `↑` / `↓` | Navigate history |

---

## Cognitive States

AVA displays its state in message headers:

| State | Description |
|-------|-------------|
| `FLOW` | Normal operation |
| `HESITATION` | Uncertainty detected |
| `CONFUSION` | High entropy |
| `CREATIVE` | Exploratory mode |
| `VERIFYING` | Fact-checking |

---

## Special Modes

### Search Mode (`Ctrl+S`)
Forces web search for the next query. Use for questions requiring current information.

### Deep Think Mode (`Ctrl+D`)
Forces Cortex (larger model) for complex reasoning tasks.

---

## Accessibility

### Keyboard Navigation
- Full keyboard control - no mouse required
- Tab cycling between panels
- Vim-style navigation (`j/k` for scrolling)

### Screen Reader Support
- State changes announced (connection, thinking)
- Panels have descriptive titles
- Clear focus indicators

### Visual Accessibility
- High contrast color scheme
- Double borders on focused elements
- Distinct colors for user vs assistant

---

## Theme Colors

| Color | Hex | Usage |
|-------|-----|-------|
| Primary | `#00D4C8` | Highlights, borders |
| Secondary | `#8B5CF6` | Cortex indicators |
| Surface | `#0E0E14` | Background |
| Text | `#F0F0F5` | Main text |
| Success | `#10B981` | Positive states |
| Warning | `#F59E0B` | Caution |
| Error | `#EF4444` | Errors |

---

## Persistence

- **Conversations**: Auto-saved to SQLite
- **Sessions**: Persist across restarts
- **Settings**: Saved to `config/user_config.yaml`

### Export Formats

Press `Ctrl+E` to export:
- `data/exports/conversation_*.md` - Markdown
- `data/exports/conversation_*.json` - JSON

---

## Configuration

```yaml
# config/user_config.yaml
backend:
  url: "http://localhost:8085"
  timeout: 30

ui:
  theme: "dark"
  show_metrics: true
  
cognitive:
  cortex_threshold: 0.7
  search_first: true
```

---

## Troubleshooting

### TUI Not Starting
```bash
pip install textual>=0.40.0
python run_tui.py --debug
```

### Backend Connection Failed
```bash
# Verify backend is running
curl http://localhost:8085/health
```

### Display Issues
```bash
export TERM=xterm-256color
stty size  # Minimum 80x24
```

---

## Architecture

```
tui/
├── app.py               # Main application
├── components/
│   ├── chat_panel.py    # Message display
│   ├── input_box.py     # User input
│   ├── metrics_panel.py # Real-time metrics
│   ├── status_bar.py    # Bottom bar
│   └── thinking_indicator.py  # Brain animation
└── styles/
    └── base.tcss        # Neural theme
```

---

## Tips

1. **Quick Nav**: `Ctrl+1/2/3` to jump to panels
2. **Vim Style**: Use `j/k` for scrolling
3. **Command Palette**: `Ctrl+K` for all commands
4. **Export First**: Export before clearing important chats
5. **Model Switch**: Smaller = faster, larger = smarter
