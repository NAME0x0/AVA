# AVA Quick Start Guide

Get AVA running in under 5 minutes.

> **New to programming?** Check out the [Beginner's Guide](BEGINNER_GUIDE.md) for a gentler introduction.

## Prerequisites

- **Python 3.9+** ([Download](https://python.org))
- **Ollama** ([Download](https://ollama.ai))
- **Git** (optional, [Download](https://git-scm.com))

## Step 1: One-Command Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/NAME0x0/AVA.git
cd AVA

# Run automated setup
python setup_ava.py
```

The setup script automatically:
- Creates a virtual environment
- Installs all dependencies
- Downloads required AI models
- Verifies the installation

You'll see a live progress bar:
```
[Step 3/7] Installing Dependencies
  [████████████░░░░░░░░] 60%  Installing httpx
```

### Setup Options

```bash
python setup_ava.py              # Standard setup
python setup_ava.py --minimal    # Minimal models (faster)
python setup_ava.py --full       # All models (best quality)
python setup_ava.py --verbose    # Show all details
python setup_ava.py --check      # Verify existing install
```

## Step 2: Start Ollama

```bash
# In a new terminal, start Ollama
ollama serve

# Pull the default model (first time only)
ollama pull gemma3:4b
```

## Step 3: Run AVA (30 seconds)

Choose your preferred interface:

### Option A: Web API + GUI

```bash
python server.py
```

Open http://localhost:8085 in your browser.

### Option B: Terminal UI

```bash
python run_tui.py
```

### Option C: Quick Chat

```bash
python -c "
from ava import AVA
import asyncio

async def chat():
    ava = AVA()
    await ava.start()
    result = await ava.chat('Hello, AVA!')
    print(result.text)
    await ava.stop()

asyncio.run(chat())
"
```

## Verify Installation

Run the doctor command to check your setup:

```bash
python scripts/verify_install.py
```

You should see all green checkmarks.

## First Conversation

Try these prompts to explore AVA's capabilities:

| Prompt | Expected Behavior |
|--------|------------------|
| "Hello!" | Quick Medulla response |
| "What is Python?" | May trigger web search |
| "Explain quantum computing in depth" | Deep Cortex thinking |

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+K` | Open command palette |
| `Ctrl+D` | Force deep thinking |
| `Ctrl+L` | Clear chat |
| `Ctrl+T` | Toggle metrics panel |
| `Ctrl+4` | Toggle settings panel |
| `Ctrl+5` | Toggle tools panel |
| `F1` | Show help |
| `Ctrl+Q` | Quit |

## Next Steps

- Read the [Beginner's Guide](BEGINNER_GUIDE.md) for detailed explanations
- Check [Troubleshooting](#common-issues) below if you hit issues
- Explore the [Architecture docs](CORTEX_MEDULLA.md) for technical details

## Common Issues

### "Ollama not running"

```bash
ollama serve
```

### "No models available"

```bash
ollama pull gemma3:4b
```

### Python version error

AVA requires Python 3.10+. Check with:

```bash
python --version
```

---

*That's it! You're ready to use AVA.*
