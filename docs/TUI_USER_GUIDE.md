# AVA TUI User Guide

The AVA Terminal User Interface (TUI) provides a powerful command-line interface for interacting with AVA. It's designed for power users who prefer keyboard-driven workflows.

## Getting Started

### Launch the TUI

```bash
# From the project root
python run_tui.py

# Or with custom backend URL
python run_tui.py --backend-url http://localhost:8085
```

### First Launch

On first launch, the TUI will:
1. Connect to the AVA backend server
2. Create a new conversation session (persisted to SQLite)
3. Display the main interface

## Interface Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  AVA v4 - Neural Interface                          [Connected] │
├──────────────────────┬──────────────────────────────────────────┤
│                      │                                          │
│     Settings         │              Chat Panel                  │
│       Panel          │                                          │
│                      │  You • 10:30                             │
│  Simulation Mode: ✓  │  Hello, AVA!                             │
│  Search First: ✓     │                                          │
│  Thermal Limit: 75°C │  AVA • 10:30 • FLOW                      │
│  Cortex Threshold:   │  Hello! How can I help you today?       │
│        0.7           │                                          │
│                      ├──────────────────────────────────────────┤
│                      │  [Thinking indicator]                    │
│                      ├──────────────────────────────────────────┤
│                      │  Type your message...                    │
├──────────────────────┼──────────────────────────────────────────┤
│   System Metrics     │              Tools Panel                 │
│                      │                                          │
│  Entropy: 0.45       │  🧮 Calculator                           │
│  Surprise: 0.23      │  🔍 Web Search                           │
│  Confidence: 0.87    │  📅 Date/Time                            │
└──────────────────────┴──────────────────────────────────────────┘
│ ● RUNNING │ ⏱ 1h 23m │ ⚡ medulla │ 🧠 FLOW │ 💬 42 │ ⚡ 234ms │
└─────────────────────────────────────────────────────────────────┘
```

## Keyboard Shortcuts

### Navigation
| Key | Action |
|-----|--------|
| `Tab` | Focus next panel |
| `Shift+Tab` | Focus previous panel |
| `Ctrl+1` | Focus input box |
| `Ctrl+2` | Focus chat history |
| `Ctrl+3` | Focus metrics panel |
| `Ctrl+4` | Toggle settings panel |
| `Ctrl+5` | Toggle tools panel |

### Chat Scrolling
| Key | Action |
|-----|--------|
| `↑/↓` or `j/k` | Scroll by line |
| `Page Up/Down` | Scroll by page |
| `Home` | Jump to top |
| `End` | Jump to bottom |

### Commands
| Key | Action |
|-----|--------|
| `Ctrl+K` | Open command palette |
| `Ctrl+L` | Clear chat history |
| `Ctrl+T` | Toggle metrics panel |
| `Ctrl+E` | Export conversation |
| `Ctrl+M` | Cycle through models |
| `Ctrl+S` | Force search mode |
| `Ctrl+D` | Deep think mode (Cortex) |
| `F1` or `?` | Show help screen |
| `Escape` | Close overlay |
| `Ctrl+Q` | Quit AVA |

### Input
| Key | Action |
|-----|--------|
| `Enter` | Send message |
| `Shift+Enter` | New line |
| `↑/↓` | Navigate input history |

## Features

### Syntax Highlighting

Code blocks in AVA's responses are automatically syntax-highlighted using Pygments. Supports 100+ programming languages.

### Conversation Export

Press `Ctrl+E` to export the current conversation. Exports are saved to:
- `data/exports/conversation_YYYYMMDD_HHMMSS.md` - Human-readable Markdown
- `data/exports/conversation_YYYYMMDD_HHMMSS.json` - Machine-readable JSON

### Model Switching

Press `Ctrl+M` to cycle through available Ollama models. The current model is displayed in the status bar.

### Cognitive States

AVA displays its current cognitive state in message headers:

| State | Description |
|-------|-------------|
| `FLOW` | Normal operation, quick responses |
| `CURIOUS` | Exploring, may ask clarifying questions |
| `FOCUSED` | Deep analysis mode |
| `CREATIVE` | Generating novel content |
| `CONFUSED` | May need more context |

### Special Modes

#### Search Mode (`Ctrl+S`)
Enables web search for the next query. Useful for questions requiring current information.

#### Deep Think Mode (`Ctrl+D`)
Forces AVA to use the Cortex (larger model) for the next query. Use for complex reasoning tasks.

## Persistence

### Session Management
- Conversations are automatically saved to SQLite
- Sessions persist across restarts
- Each session has a unique ID

### Settings
- Settings are saved to `config/user_config.yaml`
- Changes apply immediately

## Troubleshooting

### Connection Issues
If you see "Offline" in the status:
1. Check that the AVA backend is running (`python ava_server.py`)
2. Verify Ollama is running (`ollama serve`)
3. Check the backend URL matches your configuration

### Performance
- The TUI uses minimal resources
- For slow responses, try a smaller model (`Ctrl+M`)
- Clear chat history (`Ctrl+L`) if it gets too long

### Display Issues
- Ensure your terminal supports Unicode
- Use a terminal with at least 80x24 dimensions
- For best results, use a modern terminal emulator

## Configuration

The TUI reads configuration from:
- `config/base_config.yaml` - Default settings
- `config/user_config.yaml` - User overrides

Key settings:
```yaml
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

## Tips & Tricks

1. **Quick Navigation**: Use `Ctrl+1/2/3` to jump directly to panels
2. **Vim-Style Scrolling**: Use `j/k` for line-by-line scrolling
3. **Command Palette**: `Ctrl+K` gives quick access to all commands
4. **Export Before Clearing**: Export important conversations before clearing
5. **Model Selection**: Smaller models are faster, larger models are smarter
