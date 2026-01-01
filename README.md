# AVA - Autonomous Virtual Assistant

<p align="center">
  <img src="docs/assets/ava_logo.png" alt="AVA Logo" width="150" />
</p>

<p align="center">
  <a href="https://github.com/NAME0x0/AVA/actions"><img src="https://img.shields.io/github/actions/workflow/status/NAME0x0/AVA/ci.yml?branch=main&style=flat-square&logo=github&label=CI" alt="CI Status"></a>
  <a href="https://github.com/NAME0x0/AVA"><img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://github.com/NAME0x0/AVA/blob/main/LICENSE"><img src="https://img.shields.io/github/license/NAME0x0/AVA?style=flat-square" alt="License"></a>
  <a href="https://github.com/NAME0x0/AVA/releases"><img src="https://img.shields.io/github/v/release/NAME0x0/AVA?style=flat-square&include_prereleases" alt="Release"></a>
  <a href="https://github.com/NAME0x0/AVA/releases/latest"><img src="https://img.shields.io/github/downloads/NAME0x0/AVA/total?style=flat-square&logo=windows&label=Downloads" alt="Downloads"></a>
</p>

<p align="center">
  <a href="https://github.com/NAME0x0/AVA/releases/latest">
    <img src="https://img.shields.io/badge/Download-Latest%20Release-0078D4?style=for-the-badge&logo=windows&logoColor=white" alt="Download Latest Release">
  </a>
</p>

**AVA v3** is a research-grade AI assistant with a **biomimetic dual-brain architecture** inspired by the human nervous system. It runs locally on constrained hardware (4GB VRAM) and prioritizes accuracy over speed.

## What's New in v3

- **Cortex-Medulla Architecture**: Fast reflexive responses for simple queries, deep reasoning for complex ones
- **Desktop App**: Native Tauri + Next.js GUI with real-time neural activity visualization
- **System Tray**: Run in background with minimal resource usage
- **Terminal UI**: Phenomenal TUI for power users built with Textual
- **Search-First Paradigm**: Web search as default for informational queries
- **Titans Neural Memory**: Infinite context through test-time learning
- **Active Inference**: Autonomous behavior using Free Energy Principle
- **Automated Bug Reporting**: One-click bug reports with system info

---

## Two-App Architecture

AVA v3.3+ uses a **two-app architecture** for improved reliability:

```
┌─────────────────────────────────────────────────────────────────┐
│  AVA SERVER (Backend)                                           │
│  - Standalone executable or Python script                       │
│  - Handles all AI processing via Ollama                         │
│  - Runs on http://localhost:8085                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP / WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  AVA UI (Frontend)                                              │
│  - Desktop app (Tauri) or browser                               │
│  - Connects to the server automatically                         │
│  - Shows diagnostics if server isn't running                    │
└─────────────────────────────────────────────────────────────────┘
```

**Start the server first**, then launch the UI. This separation ensures:
- Clear error messages when something goes wrong
- Ability to run server and UI on different machines
- Independent updates for backend and frontend

---

## Installation

### Prerequisites

1. **Ollama** - [Download Ollama](https://ollama.ai/) - **Required**
   ```bash
   ollama pull gemma3:4b
   ollama serve
   ```
2. **Python 3.10+** - [Download Python](https://www.python.org/downloads/) (for pip install or source)
3. **Node.js 20+** (for UI development) - [Download Node.js](https://nodejs.org/)

### Quick Start

**Step 1: Start the Backend**

```bash
# Option A: Standalone executable (download from Releases)
./ava-server.exe

# Option B: Python package
pip install ava-agent
ava-server --port 8085

# Option C: From source
python ava_server.py
```

**Step 2: Launch the UI**

Download and run the desktop app from [Releases](https://github.com/NAME0x0/AVA/releases/latest), or open http://localhost:3000 in your browser (after running `npm run dev` in the ui/ directory).

### Windows Installer

<p align="center">
  <a href="https://github.com/NAME0x0/AVA/releases/latest">
    <img src="https://img.shields.io/badge/Download-Latest%20Release-28a745?style=for-the-badge&logo=github&logoColor=white" alt="Latest Release">
  </a>
</p>

The release includes:
- `ava-server.exe` - Standalone backend server
- `AVA_x.x.x_x64-setup.exe` - Desktop UI installer

---

## Quick Start

> **New to open source or AI projects?** See our [Beginner's Guide](docs/BEGINNER_GUIDE.md) for step-by-step instructions.

### One-Command Setup (From Source)

```bash
git clone https://github.com/NAME0x0/AVA.git
cd AVA
python setup_ava.py
```

The setup script shows live progress and handles everything:
```
[Step 3/7] Installing Dependencies
  [████████████░░░░░░░░] 60%  Installing httpx
  Done: 15 OK
```

**Setup options:**
- `python setup_ava.py --minimal` - Faster setup with small models
- `python setup_ava.py --full` - Complete setup with all models
- `python setup_ava.py --verbose` - Show detailed output
- `python setup_ava.py --check` - Verify existing installation

### Start AVA

```bash
python server.py        # Start the API server
# Then open http://localhost:8085
```

---

## Running AVA

### 1. Start the Server (Required)

```bash
# With preflight checks and verbose logging (recommended)
python ava_server.py

# Or the simpler server.py
python server.py

# Check dependencies without starting
python ava_server.py --check

# Run on a different port
python ava_server.py --port 8080
```

### 2. Connect with UI

**Desktop App (GUI)**
```bash
cd ui
npm install
npm run tauri dev
```

**Terminal UI (Power Users)**
```bash
python run_tui.py
# Full-featured terminal interface with keybindings
```

**Browser**
Open http://localhost:3000 after starting the UI dev server.

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

## Implementation Status

> **Note**: This section clarifies what is currently implemented vs. planned for future development.

### ✅ Fully Implemented

| Feature | Description |
|---------|-------------|
| **Ollama Integration** | Full LLM inference via Ollama (gemma3:4b default) |
| **HTTP API Server** | REST + WebSocket endpoints for chat, tools, status |
| **Search-First Workflow** | Web search as default for informational queries |
| **Active Inference** | Autonomous policy selection using Free Energy Principle |
| **Entropy-Based Routing** | Query complexity analysis via Entropix |
| **Command Safety** | Blocking dangerous system commands |
| **Thermal Monitoring** | GPU temperature tracking and throttling |
| **Terminal UI** | Full-featured TUI with Textual |
| **Desktop GUI** | Tauri + Next.js with neural visualization |
| **Memory System** | Episodic memory with conversation storage |

### 🚧 Designed but Not Yet Implemented

| Feature | Description | Status |
|---------|-------------|--------|
| **AirLLM (70B)** | Layer-wise paging for large models | Architecture ready |
| **BitNet 3B** | 1.58-bit quantized Medulla talker | Not integrated |
| **Slender-Mamba** | 1-bit SSM for Medulla monitor | Not integrated |
| **Titans Test-Time Learning** | Online memory weight updates | Architecture ready |
| **Bridge Adapter Training** | MLP projection training pipeline | Not implemented |
| **Expert Adapters** | DeepSeek-Coder, Butler adapters | Not created |

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
├── installer/           # Installer build system
│   ├── config/          # Installer configuration
│   ├── nsis/            # NSIS scripts (Windows)
│   └── scripts/         # Build automation
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
│   └── src-tauri/       # Rust backend (system tray, bug reports)
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
