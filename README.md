# AVA - Your Personal AI Research Assistant

<p align="center">
  <img src="docs/assets/ava_logo.png" alt="AVA Logo" width="150" />
</p>

**AVA** is a research-grade AI assistant that runs completely on your computer. It's designed for accuracy over speed, using a "think-before-speaking" architecture inspired by the human brain.

## ✨ What Makes AVA Special

- 🧠 **Dual-Brain Architecture**: Quick responses for simple questions, deep thinking for complex ones
- 🎯 **Accuracy First**: Verifies important responses before answering
- 🔧 **Tool Use**: Can search the web, do calculations, read files, and more
- 🔌 **MCP Support**: Connect to external tools via Model Context Protocol
- 🔒 **Private**: Everything runs locally on your machine
- 💻 **Works on 4GB VRAM**: Optimized for consumer GPUs

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites

1. **Python 3.10+** - [Download Python](https://www.python.org/downloads/)
2. **Ollama** - [Download Ollama](https://ollama.ai/)

### Step 1: Install Ollama and Download a Model

```bash
# After installing Ollama, open a terminal and run:
ollama pull gemma3:4b
```

This downloads a 4B parameter model (~3GB). For better quality, try `ollama pull llama3:8b` (~5GB).

### Step 2: Download AVA

```bash
git clone https://github.com/NAME0x0/AVA.git
cd AVA
```

Or download as ZIP from GitHub and extract.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Start AVA

```bash
# Make sure Ollama is running first (it usually starts automatically)
python server.py
```

You should see:
```
AVA API SERVER
============================================================
Starting on http://127.0.0.1:8085
Server ready. Press Ctrl+C to stop.
```

### Step 5: Chat with AVA

Open another terminal:

```bash
# Simple chat
curl -X POST http://localhost:8085/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello! What can you help me with?"}'

# Force deep thinking
curl -X POST http://localhost:8085/think \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain how transformers work in AI"}'
```

Or open `http://localhost:8085/health` in your browser to verify it's running.

---

## 💬 Using AVA

### Python API

```python
import asyncio
from src.ava import AVA

async def main():
    ava = AVA()
    await ava.start()
    
    # Simple chat
    response = await ava.chat("What is the capital of France?")
    print(response.text)
    
    # Force deep thinking for complex questions
    response = await ava.think("Compare Python and Rust for systems programming")
    print(response.text)
    
    await ava.stop()

asyncio.run(main())
```

### HTTP API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check if AVA is running |
| `/status` | GET | Get system status |
| `/chat` | POST | Send a message |
| `/think` | POST | Force deep thinking |
| `/tools` | GET | List available tools |
| `/ws` | WebSocket | Streaming chat |

### Example: Chat with JSON

```python
import requests

response = requests.post(
    "http://localhost:8085/chat",
    json={"message": "What is 25 * 17?"}
)
print(response.json()["text"])
```

---

## ⚙️ Configuration

Create `config/ava.yaml` to customize:

```yaml
engine:
  ollama:
    host: "http://localhost:11434"
    fast_model: "gemma3:4b"    # For quick responses
    deep_model: "llama3:8b"    # For deep thinking (if available)
  
  # When to use deep thinking
  cortex_surprise_threshold: 0.5
  cortex_complexity_threshold: 0.4
  
  # Enable response verification for accuracy
  verify_responses: true

tools:
  enable_calculator: true
  enable_web_search: true
  enable_file_access: true

ui:
  port: 8085
```

---

## 🔧 Available Tools

AVA can use these tools automatically:

| Tool | Description |
|------|-------------|
| `calculator` | Safe math evaluation |
| `datetime` | Current date/time |
| `web_search` | Search the web |
| `read_file` | Read files (restricted directories) |

### Adding MCP Tools

AVA supports [Model Context Protocol](https://modelcontextprotocol.io/) for external tools:

```yaml
# In config/ava.yaml
tools:
  mcp_servers:
    - name: "filesystem"
      command: "mcp-server-filesystem"
      args: ["--root", "/path/to/files"]
```

---

## 🧪 For Researchers

AVA implements several cutting-edge concepts:

- **Cortex-Medulla Architecture**: Inspired by human brain structure
- **Surprise-Based Routing**: Uses embedding distance to detect novel queries
- **Response Verification**: Double-checks factual claims
- **Test-Time Adaptation**: Adjusts processing based on query complexity

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for technical details.

---

## 📁 Project Structure

```
AVA/
├── server.py           # Main API server
├── src/
│   ├── ava/           # New clean API
│   │   ├── engine.py  # Core brain
│   │   ├── tools.py   # Tool system + MCP
│   │   ├── memory.py  # Conversation memory
│   │   └── config.py  # Configuration
│   └── core/          # Cortex-Medulla (legacy)
├── config/            # Configuration files
├── data/              # Data storage
└── docs/              # Documentation
```

---

## 🐛 Troubleshooting

### "Ollama is not running"
```bash
# Start Ollama
ollama serve
```

### "No models available"
```bash
# Download a model
ollama pull gemma3:4b
```

### "Connection refused"
- Make sure Ollama is running: `ollama serve`
- Check if the port is available: `netstat -an | findstr 11434`

### Slow Responses
- Responses take 5-30 seconds depending on your hardware
- First response is slower (model loading)
- Use a smaller model: `ollama pull phi3:mini`

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📄 License

MIT License - see [LICENSE](LICENSE).

---

## 🙏 Credits

Built with:
- [Ollama](https://ollama.ai/) - Local LLM inference
- [aiohttp](https://docs.aiohttp.org/) - Async HTTP
- Research papers: Titans (2025), Entropix (2024), Active Inference

---

<p align="center">
Made with 💜 for the research community
</p>
