# AVA User Guide

Welcome to AVA (Autonomous Virtual Assistant) - a research-grade AI assistant with a biomimetic dual-brain architecture. This guide will help you get the most out of AVA.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding AVA's Brain](#understanding-avas-brain)
3. [Using the Interfaces](#using-the-interfaces)
4. [Tips & Tricks](#tips--tricks)
5. [Frequently Asked Questions](#frequently-asked-questions)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/NAME0x0/AVA.git
cd AVA

# Run setup (Windows)
.\setup_ava.ps1

# Or manually
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Download the AI model
ollama pull gemma3:4b
```

### 2. Start AVA

```bash
# Start the API server
python server.py

# Or use the Terminal UI (recommended for power users)
python run_tui.py

# Or launch the Desktop App
cd ui && npm run tauri dev
```

### 3. Your First Conversation

Open `http://localhost:8085` in your browser, or use the TUI/Desktop app. Try these:

- **Simple question**: "What is Python?"
- **Complex analysis**: "Explain how neural networks learn"
- **Web search**: "What's the weather in Tokyo today?"

---

## Understanding AVA's Brain

AVA uses a **dual-brain architecture** inspired by the human nervous system:

### The Medulla (Fast Brain)
- **What it does**: Handles quick, routine questions
- **Response time**: < 200ms
- **When it's used**: Greetings, simple facts, common queries
- **Visual indicator**: Cyan/teal glow in the UI

### The Cortex (Deep Brain)
- **What it does**: Complex reasoning and analysis
- **Response time**: 5-60 seconds
- **When it's used**: Philosophy, coding, creative writing
- **Visual indicator**: Purple/magenta glow in the UI

### How AVA Decides

AVA calculates a "**surprise score**" for each input:

| Surprise Level | Brain Used | Example Queries |
|---------------|------------|-----------------|
| 0.0 - 0.3 | Medulla | "Hello", "What time is it?" |
| 0.3 - 0.7 | Agency decides | "Explain Python decorators" |
| 0.7 - 1.0 | Cortex | "Analyze the philosophy of Kant" |

### Cognitive States

Watch the metrics panel to see AVA's mental state:

| State | Meaning | Color |
|-------|---------|-------|
| **FLOW** | Confident, processing smoothly | Green |
| **HESITATION** | Uncertain, considering options | Yellow |
| **CONFUSION** | High uncertainty, may need clarification | Red |
| **CREATIVE** | Exploring novel ideas | Blue |
| **VERIFYING** | Fact-checking the response | Purple |

---

## Using the Interfaces

### Desktop App (GUI)

The desktop app provides a beautiful, modern interface:

**Layout:**
- **Left sidebar**: Real-time 3D neural visualization, metrics
- **Center**: Chat area with message history
- **Bottom**: Input field

**Keyboard Shortcuts:**
| Key | Action |
|-----|--------|
| `Ctrl+K` | Open command palette |
| `Ctrl+B` | Toggle sidebar |
| `Ctrl+L` | Clear chat |
| `Ctrl+D` | Force deep thinking |

**Features:**
- Real-time 3D brain visualization
- Cognitive state indicators
- Streaming responses
- Dark neural theme

### Terminal UI (TUI)

For power users who prefer the command line:

```bash
python run_tui.py
```

**Keyboard Shortcuts:**
| Key | Action |
|-----|--------|
| `Ctrl+K` | Command palette |
| `Ctrl+L` | Clear chat |
| `Ctrl+T` | Toggle metrics |
| `Ctrl+S` | Force search mode |
| `Ctrl+D` | Force deep thinking |
| `F1` | Help |
| `Ctrl+Q` | Quit |

**Features:**
- ASCII neural network animation
- Split-pane view
- Real-time metrics
- Command history (↑/↓)

### HTTP API

For programmatic access:

```bash
# Simple chat
curl -X POST http://localhost:8085/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello AVA!"}'

# Force deep thinking
curl -X POST http://localhost:8085/think \
  -H "Content-Type: application/json" \
  -d '{"message": "Analyze the ethics of AI"}'
```

### Python API

```python
from ava import AVA
import asyncio

async def main():
    ava = AVA()
    await ava.start()

    # Regular chat (auto-routes to Medulla or Cortex)
    response = await ava.chat("What is machine learning?")
    print(response.text)

    # Force deep thinking
    response = await ava.think("Compare philosophy of mind theories")
    print(response.text)

    await ava.stop()

asyncio.run(main())
```

---

## Tips & Tricks

### Getting Better Responses

1. **Be specific**: Instead of "Tell me about AI", try "Explain how transformers work in NLP"

2. **Use question words for search**: Starting with "What", "When", "Where", "Who", "How", "Why" triggers web search

3. **Force deep thinking**: For complex topics, use `Ctrl+D` before asking or prefix with "Think deeply about..."

4. **Provide context**: "Given that I'm a Python developer, explain..."

### Performance Tips

1. **First response is slower**: The model needs to load initially

2. **Keep Ollama running**: Start `ollama serve` and leave it running

3. **Monitor temperature**: AVA throttles when GPU gets hot (check status bar)

4. **Use simulation mode for testing**: Add `--simulation` flag for faster responses without real AI

### Power User Tricks

1. **Chain queries**: AVA remembers context within a session

2. **Force specific modes**:
   - `Ctrl+S` then query = Force web search
   - `Ctrl+D` then query = Force Cortex (deep thinking)

3. **Use the command palette**: `Ctrl+K` gives you quick access to all actions

4. **Watch the metrics**: Entropy and varentropy tell you about uncertainty

5. **Check response time**: Status bar shows average response time

---

## Frequently Asked Questions

### General Questions

**Q: What models does AVA use?**

A: AVA uses models via Ollama. Default is `gemma3:4b` for the Medulla (fast responses). For Cortex (deep thinking), it can use larger models like `qwen2.5:32b` via layer-wise paging.

**Q: Does AVA work offline?**

A: Yes! AVA runs entirely on your computer. However, web search features require internet access.

**Q: How much VRAM do I need?**

A: Minimum 4GB. AVA is optimized for RTX A2000 (4GB) but works with any NVIDIA GPU with 4GB+.

**Q: Can AVA run on CPU only?**

A: Yes, but it will be slower. Ollama supports CPU-only mode.

### Usage Questions

**Q: Why is the first response slow?**

A: The AI model needs to load into memory. Subsequent responses are faster.

**Q: How do I make AVA think deeper?**

A: Press `Ctrl+D` before your query, or start with "Think deeply about..." or use the `/think` endpoint.

**Q: Why does AVA search the web for some questions?**

A: AVA uses a "Search-First" approach for factual questions (starting with what, when, where, who, how, why) to provide accurate, up-to-date information.

**Q: Can I disable web search?**

A: Yes, in `config/cortex_medulla.yaml`, set `search_first.enabled: false`.

**Q: How do I clear the conversation?**

A: Press `Ctrl+L` or use the command palette (`Ctrl+K` → "Clear Chat").

### Technical Questions

**Q: What are entropy and varentropy?**

A: These are information-theoretic measures:
- **Entropy**: How uncertain AVA is (higher = more uncertain)
- **Varentropy**: Variance in uncertainty (higher = more variable confidence)

**Q: What does "surprise" mean?**

A: Surprise measures how novel your input is compared to what AVA has seen. High surprise triggers deeper thinking.

**Q: Why does my GPU get hot?**

A: AI inference is compute-intensive. AVA monitors temperature and automatically throttles to protect your GPU. The default power cap is 15%.

**Q: Can I use different AI providers?**

A: Currently AVA uses Ollama. OpenAI/Anthropic support may be added in future versions.

### Troubleshooting Questions

**Q: "Ollama is not running" error?**

A: Start Ollama with `ollama serve` in a terminal.

**Q: "No models available" error?**

A: Download a model: `ollama pull gemma3:4b`

**Q: "Connection refused" error?**

A: Make sure:
1. Ollama is running (`ollama serve`)
2. Port 8085 is not in use
3. Firewall isn't blocking connections

**Q: Responses are too slow?**

A: Try:
1. Use a smaller model: `ollama pull phi3:mini`
2. Enable simulation mode for testing
3. Check GPU temperature (might be thermal throttling)

---

## Troubleshooting

### Common Issues

#### AVA Won't Start

```bash
# Check Python version (need 3.10+)
python --version

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check if port is in use
netstat -an | findstr 8085
```

#### Slow Responses

1. **Check GPU usage**: Open Task Manager → Performance → GPU
2. **Check temperature**: Look at status bar or use `nvidia-smi`
3. **Use smaller model**: Try `phi3:mini` instead of `gemma3:4b`
4. **Restart Ollama**: Sometimes it helps to restart the model server

#### Memory Issues

```bash
# Check available VRAM
nvidia-smi

# Use CPU-only mode (slower but less memory)
# Set CUDA_VISIBLE_DEVICES="" before running
```

#### TUI Display Issues

```bash
# Ensure terminal supports 256 colors
export TERM=xterm-256color

# Check terminal size (minimum 80x24)
stty size

# Try a different terminal emulator
```

### Getting Help

1. Check the [GitHub Issues](https://github.com/NAME0x0/AVA/issues)
2. Read the [Architecture docs](./ARCHITECTURE.md)
3. Review [Configuration guide](./CONFIGURATION.md)

---

## Advanced Usage

### Custom Configuration

Edit `config/cortex_medulla.yaml`:

```yaml
# Adjust when Cortex activates
medulla:
  low_surprise_threshold: 0.3   # Below this: always Medulla
  high_surprise_threshold: 0.7  # Above this: always Cortex

# Web search behavior
search_first:
  enabled: true
  min_sources: 3
  agreement_threshold: 0.7

# GPU protection
thermal:
  max_gpu_power_percent: 15
  warning_temp_c: 75
  throttle_temp_c: 80

# Curiosity level
agency:
  epistemic_weight: 0.6  # Higher = more exploratory
```

### Adding Custom Tools

```python
# In src/ava/tools.py
class MyTool(Tool):
    name = "my_tool"
    description = "Does something useful"

    async def execute(self, args: dict) -> ToolResult:
        # Your implementation
        return ToolResult(output="Result")

# Register it
tool_manager.register(MyTool())
```

### MCP Server Integration

AVA supports Model Context Protocol for external tools:

```yaml
# In config/ava.yaml
tools:
  mcp_servers:
    - name: "filesystem"
      command: "mcp-server-filesystem"
      args: ["--root", "/path/to/files"]
```

### Running in Production

```bash
# Use a process manager
pip install supervisor

# Or with systemd (Linux)
sudo systemctl enable ava
sudo systemctl start ava
```

---

## Glossary

| Term | Definition |
|------|------------|
| **Medulla** | The fast-response component of AVA's brain |
| **Cortex** | The deep-thinking component for complex queries |
| **Bridge** | Connects Medulla and Cortex representations |
| **Titans** | Test-time learning memory system |
| **Agency** | Autonomous decision-making module |
| **Surprise** | How novel/unexpected an input is |
| **Entropy** | Measure of uncertainty in responses |
| **Varentropy** | Variance in confidence levels |
| **VRAM** | Video RAM (GPU memory) |
| **Ollama** | Local LLM inference server |

---

*Last updated: December 2024*
