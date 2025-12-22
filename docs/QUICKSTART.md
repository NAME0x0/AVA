# AVA Quick Start Guide

Get AVA running in under 5 minutes.

## Prerequisites

- **Python 3.10+** ([Download](https://python.org))
- **Ollama** ([Download](https://ollama.ai))
- **Git** ([Download](https://git-scm.com))

## Step 1: Clone and Install (2 minutes)

```bash
# Clone the repository
git clone https://github.com/NAME0x0/AVA.git
cd AVA

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Start Ollama (1 minute)

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
| `Ctrl+B` | Toggle sidebar |

## Next Steps

- Read the [User Guide](USER_GUIDE.md) for detailed usage
- Check [Troubleshooting](TROUBLESHOOTING.md) if you hit issues
- Explore [API Examples](API_EXAMPLES.md) for programmatic use

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
