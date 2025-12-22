# AVA - Autonomous Virtual Assistant

<p align="center">
  <img src="docs/assets/ava_logo.png" alt="AVA Logo" width="150" />
</p>

**AVA v3** is a research-grade AI assistant with a **biomimetic dual-brain architecture** inspired by the human nervous system. It runs locally on constrained hardware (4GB VRAM) and prioritizes accuracy over speed.

## What's New in v3

- **Cortex-Medulla Architecture**: Fast reflexive responses for simple queries, deep reasoning for complex ones
- **Desktop App**: Native Tauri + Next.js GUI with real-time neural activity visualization
- **Terminal UI**: Phenomenal TUI for power users built with Textual
- **Search-First Paradigm**: Web search as default for informational queries
- **Titans Neural Memory**: Infinite context through test-time learning
- **Active Inference**: Autonomous behavior using Free Energy Principle

---

## Quick Start

### Prerequisites

1. **Python 3.10+** - [Download Python](https://www.python.org/downloads/)
2. **Ollama** - [Download Ollama](https://ollama.ai/)
3. **Node.js 18+** (for GUI) - [Download Node.js](https://nodejs.org/)

### One-Click Setup

**Windows:**
```powershell
.\setup_ava.ps1
```

**macOS/Linux:**
```bash
chmod +x start.sh && ./start.sh
```

### Manual Setup

```bash
# Clone repository
git clone https://github.com/NAME0x0/AVA.git
cd AVA

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download model
ollama pull gemma3:4b

# Start server
python server.py
```

---

## Running AVA

### HTTP API Server (Primary)
```bash
python server.py
# Runs on http://localhost:8085
```

### Terminal UI (Power Users)
```bash
python run_tui.py
# Full-featured terminal interface with keybindings
```

### Desktop App (GUI)
```bash
cd ui
npm install
npm run tauri dev
```

### Core System (Direct)
```bash
python run_core.py --simulation
```

---

## Architecture

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  MEDULLA (Reflexive Core) - Always On                  │
│  - Mamba SSM for O(1) memory sensing                   │
│  - 1-bit BitNet for quick responses                    │
│  - Calculates "surprise" signal                        │
│  - VRAM: ~800 MB (resident)                            │
└─────────────────────────────────────────────────────────┘
    │                              │
    │ Low Surprise                 │ High Surprise
    ▼                              ▼
┌─────────────┐             ┌─────────────────────────────┐
│Quick Reply  │             │  CORTEX (Reflective Core)   │
│(<200ms)     │             │  - 70B model via AirLLM     │
└─────────────┘             │  - Layer-wise paging        │
                            │  - ~3.3s per token          │
                            │  - VRAM: ~1.6 GB (paged)    │
                            └─────────────────────────────┘
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **Medulla** | `src/core/medulla.py` | Always-on sensory processing |
| **Cortex** | `src/core/cortex.py` | Deep reasoning (70B on 4GB) |
| **Bridge** | `src/core/bridge.py` | Projects Medulla → Cortex |
| **Agency** | `src/core/agency.py` | Active Inference |
| **Titans** | `src/hippocampus/titans.py` | Test-time learning |
| **System** | `src/core/system.py` | Orchestration |

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/status` | GET | System status with metrics |
| `/chat` | POST | Send message (auto-routes Medulla/Cortex) |
| `/think` | POST | Force deep thinking (Cortex) |
| `/tools` | GET | List available tools |
| `/ws` | WebSocket | Streaming chat |

### Python API

```python
from ava import AVA
import asyncio

async def main():
    ava = AVA()
    await ava.start()

    # Auto-routes based on complexity
    response = await ava.chat("What is Python?")
    print(response.text)

    # Force deep thinking
    response = await ava.think("Explain quantum computing")
    print(response.text)

    await ava.stop()

asyncio.run(main())
```

---

## TUI Keybindings

| Key | Action |
|-----|--------|
| `Ctrl+K` | Command palette |
| `Ctrl+L` | Clear chat |
| `Ctrl+T` | Toggle metrics |
| `Ctrl+S` | Force search |
| `Ctrl+D` | Deep think |
| `F1` | Help |
| `Ctrl+Q` | Quit |

---

## Configuration

Main config: `config/cortex_medulla.yaml`

```yaml
development:
  simulation_mode: true  # For testing without models

search_first:
  enabled: true
  min_sources: 3
  agreement_threshold: 0.7

thermal:
  max_gpu_power_percent: 15  # RTX A2000 safe limit
  warning_temp_c: 75
  throttle_temp_c: 80

agency:
  epistemic_weight: 0.6  # High curiosity
```

---

## Project Structure

```
AVA/
├── config/              # Configuration files
├── data/                # Runtime data
├── docs/                # Documentation
├── legacy/              # Archived v2 code
├── models/              # Model adapters
├── src/
│   ├── ava/             # Clean public API
│   ├── core/            # Cortex-Medulla system
│   ├── hippocampus/     # Titans memory
│   ├── cortex/          # Utilities
│   ├── inference/       # LLM inference
│   ├── learning/        # QLoRA training
│   ├── subconscious/    # Background processing
│   └── tools/           # Tool implementations
├── tests/               # Test suite
├── tui/                 # Terminal UI (Textual)
├── ui/                  # Desktop GUI (Next.js + Tauri)
├── server.py            # HTTP API server
├── run_core.py          # Direct core CLI
└── run_tui.py           # TUI entry point
```

---

## VRAM Budget (RTX A2000 4GB)

```
System Overhead:    300 MB
Medulla (Mamba):    800 MB
Titans Memory:      200 MB
Bridge Adapter:      50 MB
Cortex Buffer:    1,600 MB (paged on-demand)
────────────────────────────
Total Resident:   2,050 MB
Peak (Cortex):    3,650 MB
Headroom:           446 MB
```

---

## Troubleshooting

### "Ollama is not running"
```bash
ollama serve
```

### "No models available"
```bash
ollama pull gemma3:4b
```

### Slow Responses
- First response is slower (model loading)
- Deep thinking takes 5-30 seconds
- Use `--simulation` flag for testing

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT License - see [LICENSE](LICENSE).

---

## Credits

Built with:
- [Ollama](https://ollama.ai/) - Local LLM inference
- [Textual](https://textual.textualize.io/) - TUI framework
- [Tauri](https://tauri.app/) - Desktop apps
- [Next.js](https://nextjs.org/) - React framework

Research papers:
- Titans (2025) - Test-time learning
- Entropix (2024) - Entropy-guided routing
- Active Inference - Free Energy Principle

---

<p align="center">
Made with care for the research community
</p>
