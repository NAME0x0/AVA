# Getting Started with AVA

> **Version**: 4.2.3 (Sentinel Architecture)  
> **Audience**: New users and developers

This guide covers everything you need to get AVA running on your machine.

---

## Quick Start (5 Minutes)

### Prerequisites

| Requirement | Description |
|-------------|-------------|
| **Ollama** | Local LLM server - [Download](https://ollama.ai) |
| **Python 3.10+** | [Download](https://python.org) (for TUI only) |
| **4GB+ RAM** | System memory |
| **~10GB disk** | For AI models |

### Step 1: Install Ollama & Pull Model

```bash
# After installing Ollama from https://ollama.ai
ollama serve  # Start the server

# Pull the default model (in another terminal)
ollama pull gemma3:4b
```

### Step 2: Download AVA

**Option A: Download Release** (Recommended)
- Go to [Releases](https://github.com/NAME0x0/AVA/releases)
- Download the latest `.exe` installer for Windows
- Run the installer

**Option B: Clone Repository** (For developers)
```bash
git clone https://github.com/NAME0x0/AVA.git
cd AVA
```

### Step 3: Run AVA

**Desktop App** (Windows):
- Double-click the AVA icon
- The app will connect to Ollama automatically

**Terminal UI** (All platforms):
```bash
cd AVA
pip install -r requirements.txt
python run_tui.py
```

**Python API**:
```bash
cd AVA
python server.py
# Open http://localhost:8085 in browser
```

### Step 4: Verify Installation

```bash
python scripts/verify_install.py
```

You should see all green checkmarks ✅

---

## Understanding AVA's Architecture

AVA uses a **Sentinel Architecture** with four stages:

```
Query → Perception → Appraisal → Execution → Learning → Response
           │            │            │           │
        Surprise    Policy       Route to    Update
        Calculation Selection    Medulla/    Titans
                                 Cortex      Memory
```

### Two Brains (Cortex-Medulla)

| Brain | Model | Use Case |
|-------|-------|----------|
| **Medulla** | gemma3:4b | Fast, simple queries |
| **Cortex** | qwen2.5:32b | Complex reasoning |

AVA automatically decides which brain to use based on query complexity and surprise.

---

## Interface Options

### 1. Desktop App (Tauri + Next.js)

Best for: General users who want a native GUI experience.

```
┌─────────────────────────────────────────┐
│  AVA Desktop                    [─][□][×]│
├─────────────────────────────────────────┤
│  💬 Chat panel with streaming          │
│  📊 Real-time metrics (entropy, etc.)  │
│  ⚙️ Settings panel                      │
│  🔧 Tools panel                         │
└─────────────────────────────────────────┘
```

### 2. Terminal UI (TUI)

Best for: Power users who prefer keyboard-driven workflows.

```bash
python run_tui.py
```

**Key Shortcuts:**
| Key | Action |
|-----|--------|
| `Ctrl+K` | Command palette |
| `Ctrl+L` | Clear chat |
| `Ctrl+T` | Toggle metrics |
| `Ctrl+D` | Force deep thinking |
| `Ctrl+S` | Force search mode |
| `F1` | Help |
| `Ctrl+Q` | Quit |

### 3. HTTP API

Best for: Developers integrating AVA into other applications.

```bash
python server.py
# API available at http://localhost:8085
```

```python
import requests

response = requests.post("http://localhost:8085/chat", json={
    "message": "Hello, AVA!"
})
print(response.json()["response"])
```

---

## Configuration

### Quick Config (Environment Variables)

```bash
# Minimal setup
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=gemma3:4b
export AVA_PORT=8085
```

### Full Config (YAML)

Edit `config/cortex_medulla.yaml`:

```yaml
server:
  port: 8085
  host: "127.0.0.1"

cognitive:
  fast_model: "gemma3:4b"      # Medulla
  deep_model: "qwen2.5:32b"    # Cortex
  temperature: 0.7
  max_tokens: 2048

search:
  enabled: true
  min_sources: 3
```

See [CONFIGURATION.md](CONFIGURATION.md) for all options.

---

## Common Issues

### "Ollama not running"

```bash
# Start Ollama server
ollama serve
```

### "No models available"

```bash
# Pull the required model
ollama pull gemma3:4b
```

### "Connection refused" on port 8085

```bash
# Check if something else is using the port
netstat -ano | findstr :8085

# Use a different port
python server.py --port 8086
```

### "Out of GPU memory"

```bash
# Use smaller model
ollama pull gemma2:2b

# Or set memory limit
export AVA_GPU_MEMORY_LIMIT=3000
```

### Python version error

AVA requires Python 3.10+:
```bash
python --version  # Should be 3.10+
```

---

## Next Steps

| Goal | Resource |
|------|----------|
| Configure AVA | [CONFIGURATION.md](CONFIGURATION.md) |
| Learn architecture | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Use the TUI | [TUI_USER_GUIDE.md](TUI_USER_GUIDE.md) |
| API integration | [API_EXAMPLES.md](API_EXAMPLES.md) |
| Troubleshooting | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| Environment vars | [ENVIRONMENT_VARIABLES.md](ENVIRONMENT_VARIABLES.md) |

---

## Hardware Recommendations

| VRAM | Recommended Models |
|------|-------------------|
| 4GB | gemma3:4b (Medulla only) |
| 8GB | llama3.2:8b / llama3.1:8b |
| 16GB | qwen2.5:14b / qwen2.5:32b-q4 |
| 24GB+ | qwen2.5:72b-q4 |

For RTX A2000 (4GB):
```bash
export AVA_GPU_MEMORY_LIMIT=3500
export AVA_MAX_GPU_POWER_PERCENT=15
```

---

*Welcome to AVA! For questions, open an issue on [GitHub](https://github.com/NAME0x0/AVA/issues).*
