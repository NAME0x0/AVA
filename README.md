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

**AVA v4** is a research-grade AI assistant with a **biomimetic dual-brain architecture** inspired by the human nervous system. It runs locally on constrained hardware (4GB VRAM) and prioritizes accuracy over speed.

## What's New in v4

- **Unified Rust Backend**: Single portable executable with embedded HTTP server (no Python required)
- **Cortex-Medulla Architecture**: Fast reflexive responses for simple queries, deep reasoning for complex ones
- **Desktop App**: Native Tauri + Next.js GUI with real-time neural activity visualization
- **Active Inference Metrics**: Free Energy calculation and belief state visualization
- **System Tray**: Run in background with minimal resource usage
- **Terminal UI**: Power-user TUI built with Textual (streaming support coming soon)
- **Search-First Paradigm**: Web search as default for informational queries
- **Titans Neural Memory**: Infinite context through test-time learning
- **Active Inference**: Autonomous behavior using Free Energy Principle
- **Automated Bug Reporting**: One-click bug reports with system info

---

## Architecture

AVA v4 uses a **unified single-app architecture** for maximum portability:

```
┌─────────────────────────────────────────────────────────────────┐
│  AVA Desktop App (Single Executable)                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Embedded Rust Backend (Axum HTTP Server)               │   │
│  │  - All AI processing via Ollama                         │   │
│  │  - Runs on http://127.0.0.1:8085                        │   │
│  │  - Active Inference metrics calculation                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              │ Internal HTTP                    │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Next.js Frontend                                        │   │
│  │  - Real-time neural activity visualization               │   │
│  │  - Metrics dashboard with Free Energy display            │   │
│  │  - Chat interface with streaming responses               │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP API
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Ollama (Local LLM Server)                                      │
│  - gemma3:4b (fast mode)                                       │
│  - llama3.2:latest (deep thinking mode)                        │
└─────────────────────────────────────────────────────────────────┘
```

**Single executable** - no Python installation required for end users.

---

## Installation

### Prerequisites

1. **Ollama** - [Download Ollama](https://ollama.ai/) - **Required**
   ```bash
   ollama pull gemma3:4b
   ollama pull llama3.2:latest  # For deep thinking mode
   ollama serve
   ```

### Quick Start

**Option A: Download Pre-built App (Recommended)**

Download and run the desktop app from [Releases](https://github.com/NAME0x0/AVA/releases/latest).
- `AVA_4.0.0_x64-setup.exe` - Windows installer
- `AVA_4.0.0_x64_en-US.msi` - Windows MSI package

**Option B: Build from Source**

```bash
git clone https://github.com/NAME0x0/AVA.git
cd AVA/ui
npm install
npm run tauri build
```

**Option C: Development Mode**

```bash
cd AVA/ui
npm install
npm run tauri dev
```

### Windows Installer

<p align="center">
  <a href="https://github.com/NAME0x0/AVA/releases/latest">
    <img src="https://img.shields.io/badge/Download-Latest%20Release-28a745?style=for-the-badge&logo=github&logoColor=white" alt="Latest Release">
  </a>
</p>

The release includes:
- `AVA_4.0.0_x64-setup.exe` - Desktop app installer (single executable, no Python needed)
- `AVA_4.0.0_x64_en-US.msi` - Windows MSI package

---

## Quick Start

> **New to open source or AI projects?** See our [Beginner's Guide](docs/BEGINNER_GUIDE.md) for step-by-step instructions.

### Running AVA

**Desktop App (Recommended)**
```bash
# Download from Releases, or build from source:
cd ui
npm install
npm run tauri dev
```

**Terminal UI (Power Users)**
```bash
# Requires Python environment
pip install -e .
python -m tui.app
```

The TUI provides a keyboard-driven interface with:
- Real-time metrics display
- Command palette (Ctrl+K)
- Force modes (Ctrl+S for search, Ctrl+D for deep thinking)
- Settings management

---

## API Endpoints

The embedded server exposes these endpoints on `http://127.0.0.1:8085`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/chat` | POST | Send message, get AI response |
| `/cognitive` | GET | Current cognitive state (entropy, surprise, varentropy) |
| `/memory` | GET | Memory statistics |
| `/belief` | GET | Active Inference belief state and free energy |
| `/stats` | GET | System statistics |

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
| `/chat` | POST | Send message (auto-routes based on force mode) |
| `/cognitive` | GET | Cognitive state (entropy, surprise, varentropy) |
| `/memory` | GET | Memory statistics |
| `/belief` | GET | Active Inference belief state and free energy |
| `/stats` | GET | System statistics |

### Using the API

```bash
# Health check
curl http://127.0.0.1:8085/health

# Send a message
curl -X POST http://127.0.0.1:8085/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is Python?"}'

# Get cognitive state
curl http://127.0.0.1:8085/cognitive

# Get belief state with free energy
curl http://127.0.0.1:8085/belief
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
├── legacy/              # Archived Python server code
├── models/              # Model adapters
├── src/
│   ├── ava/             # Python API (for TUI/development)
│   ├── core/            # Cortex-Medulla system
│   ├── hippocampus/     # Titans memory
│   ├── cortex/          # Utilities
│   ├── inference/       # LLM inference
│   ├── learning/        # QLoRA training
│   ├── subconscious/    # Background processing
│   └── tools/           # Tool implementations
├── tests/               # Test suite
├── tui/                 # Terminal UI (Textual)
└── ui/                  # Desktop GUI (Next.js + Tauri)
    └── src-tauri/       # Rust backend (embedded server)
        └── src/
            ├── main.rs      # Application entry point
            └── engine/      # HTTP server (Axum)
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
ollama pull llama3.2:latest
```

### "Port 8085 already in use"
```bash
# Windows
netstat -ano | findstr :8085
taskkill /F /PID <pid>

# Linux/macOS
lsof -i :8085
kill -9 <pid>
```

### Slow Responses
- First response is slower (model loading)
- Deep thinking (Cortex mode) takes 5-30 seconds
- Use simulation mode for testing without models

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
