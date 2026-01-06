# AVA - Autonomous Virtual Assistant

<p align="center">
  <img src="docs/assets/ava_logo.png" alt="AVA Logo" width="180" />
</p>

<h3 align="center">
  <strong>🧠 Research-Grade AI Assistant with Verified Reasoning</strong>
</h3>

<p align="center">
  <em>Accuracy over Speed • Local-First • Privacy-Preserving</em>
</p>

<p align="center">
  <a href="https://github.com/NAME0x0/AVA/actions"><img src="https://img.shields.io/github/actions/workflow/status/NAME0x0/AVA/ci.yml?branch=main&style=flat-square&logo=github&label=CI" alt="CI Status"></a>
  <a href="https://github.com/NAME0x0/AVA"><img src="https://img.shields.io/badge/rust-1.75%2B-orange?style=flat-square&logo=rust&logoColor=white" alt="Rust 1.75+"></a>
  <a href="https://github.com/NAME0x0/AVA"><img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://github.com/NAME0x0/AVA/blob/main/LICENSE"><img src="https://img.shields.io/github/license/NAME0x0/AVA?style=flat-square&color=green" alt="License"></a>
  <a href="https://github.com/NAME0x0/AVA/releases"><img src="https://img.shields.io/github/v/release/NAME0x0/AVA?style=flat-square&include_prereleases&color=purple" alt="Release"></a>
</p>

<p align="center">
  <a href="https://github.com/NAME0x0/AVA/stargazers"><img src="https://img.shields.io/github/stars/NAME0x0/AVA?style=flat-square&logo=github" alt="Stars"></a>
  <a href="https://github.com/NAME0x0/AVA/network/members"><img src="https://img.shields.io/github/forks/NAME0x0/AVA?style=flat-square&logo=github" alt="Forks"></a>
  <a href="https://github.com/NAME0x0/AVA/issues"><img src="https://img.shields.io/github/issues/NAME0x0/AVA?style=flat-square" alt="Issues"></a>
  <a href="https://github.com/NAME0x0/AVA/releases/latest"><img src="https://img.shields.io/github/downloads/NAME0x0/AVA/total?style=flat-square&logo=windows&label=Downloads" alt="Downloads"></a>
</p>

<p align="center">
  <a href="https://github.com/NAME0x0/AVA/releases/latest">
    <img src="https://img.shields.io/badge/⬇_Download-Latest_Release-0078D4?style=for-the-badge&logo=windows&logoColor=white" alt="Download Latest Release">
  </a>
  &nbsp;
  <a href="docs/GETTING_STARTED.md">
    <img src="https://img.shields.io/badge/📖_Read-Documentation-28a745?style=for-the-badge" alt="Documentation">
  </a>
</p>

---

<p align="center">
  <strong>AVA v4.2</strong> implements the <strong>Sentinel Architecture</strong> — a state-of-the-art cognitive system<br>
  that prioritizes <em>verified accuracy</em> over probabilistic token generation.
</p>

---

## ✨ Why AVA?

<table>
<tr>
<td width="50%">

### 🎯 **Accuracy-First Design**
Unlike standard LLMs that "guess" tokens, AVA implements:
- **Active Inference** for autonomous decision-making
- **Search-First Verification** for factual queries
- **Test-Time Learning** that improves during use

</td>
<td width="50%">

### 🔒 **100% Local & Private**
Your data never leaves your machine:
- Runs entirely on your hardware
- No cloud dependencies
- No telemetry or tracking

</td>
</tr>
<tr>
<td width="50%">

### ⚡ **Optimized for Consumer Hardware**
Designed for 4GB VRAM GPUs:
- Layer-wise paging for large models
- Intelligent routing (fast vs deep)
- Thermal-aware processing

</td>
<td width="50%">

### 🧪 **Research-Grade Architecture**
Built on cutting-edge research:
- **Titans** (Test-Time Learning)
- **Entropix** (Entropy-Based Routing)
- **Free Energy Principle** (Active Inference)

</td>
</tr>
</table>

---

## 🏗️ Sentinel Architecture

AVA's four-stage cognitive loop ensures accurate, verified responses:

```
                              ┌─────────────────┐
                              │   USER QUERY    │
                              └────────┬────────┘
                                       │
                    ╔══════════════════╧══════════════════╗
                    ║      STAGE 1: PERCEPTION            ║
                    ║  ┌─────────────────────────────┐    ║
                    ║  │ Embedding → KL Divergence   │    ║
                    ║  │ → Surprise Score            │    ║
                    ║  └─────────────────────────────┘    ║
                    ╚══════════════════╤══════════════════╝
                                       │
                    ╔══════════════════╧══════════════════╗
                    ║      STAGE 2: APPRAISAL             ║
                    ║  ┌─────────────────────────────┐    ║
                    ║  │ Active Inference Engine     │    ║
                    ║  │ G(π) = -Pragmatic           │    ║
                    ║  │      - Epistemic + Effort   │    ║
                    ║  └─────────────────────────────┘    ║
                    ╚══════════════════╤══════════════════╝
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ▼                        ▼                        ▼
    ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
    │    MEDULLA      │      │     SEARCH      │      │     CORTEX      │
    │   Fast Path     │      │     Tools       │      │    Deep Path    │
    │   ─────────     │      │   ─────────     │      │   ─────────     │
    │   gemma3:4b     │      │   DDG/Google    │      │   qwen2.5:32b   │
    │   <200ms        │      │   Bing/Brave    │      │   3-30s         │
    └─────────────────┘      └─────────────────┘      └─────────────────┘
              │                        │                        │
              └────────────────────────┼────────────────────────┘
                                       │
                    ╔══════════════════╧══════════════════╗
                    ║      STAGE 4: LEARNING              ║
                    ║  ┌─────────────────────────────┐    ║
                    ║  │ Titans Memory Update        │    ║
                    ║  │ M_t = M_{t-1} - η∇θL       │    ║
                    ║  │ (Surprise-Weighted)         │    ║
                    ║  └─────────────────────────────┘    ║
                    ╚══════════════════╤══════════════════╝
                                       │
                              ┌────────┴────────┐
                              │ VERIFIED OUTPUT │
                              └─────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# 1. Install Ollama (required)
# Download from: https://ollama.ai

# 2. Pull models
ollama pull gemma3:4b              # Fast responses
ollama pull nomic-embed-text       # Surprise calculation
ollama serve                       # Start server
```

### Installation

<table>
<tr>
<td width="50%">

**📦 Option A: Download Release** *(Recommended)*

Download the installer from [Releases](https://github.com/NAME0x0/AVA/releases/latest):
- `AVA_x64-setup.exe` — Windows Installer
- `AVA_x64_en-US.msi` — MSI Package

</td>
<td width="50%">

**🔧 Option B: Build from Source**

```bash
git clone https://github.com/NAME0x0/AVA.git
cd AVA/ui
npm install
npm run tauri build
```

</td>
</tr>
</table>

### Run AVA

```bash
# Desktop App (GUI)
./AVA.exe                    # or double-click

# Terminal UI (Power Users)
cd AVA && pip install -e .
python -m tui.app

# API Server Only
python server.py             # http://127.0.0.1:8085
```

---

## 🎮 Features

<table>
<tr>
<td align="center" width="25%">
<img src="https://img.shields.io/badge/🧠-Active_Inference-purple?style=for-the-badge" /><br>
<strong>Policy Selection</strong><br>
<sub>Free Energy minimization for autonomous behavior</sub>
</td>
<td align="center" width="25%">
<img src="https://img.shields.io/badge/📚-Titans_Memory-blue?style=for-the-badge" /><br>
<strong>Test-Time Learning</strong><br>
<sub>Neural memory updates during inference</sub>
</td>
<td align="center" width="25%">
<img src="https://img.shields.io/badge/🔍-Search_First-green?style=for-the-badge" /><br>
<strong>Verified Facts</strong><br>
<sub>Web search before generation</sub>
</td>
<td align="center" width="25%">
<img src="https://img.shields.io/badge/⚡-Real_Surprise-orange?style=for-the-badge" /><br>
<strong>Embedding-Based</strong><br>
<sub>KL divergence, not heuristics</sub>
</td>
</tr>
</table>

### Interfaces

| Interface | Description | Launch |
|-----------|-------------|--------|
| 🖥️ **Desktop App** | Native GUI with neural visualization | `AVA.exe` |
| ⌨️ **Terminal UI** | Keyboard-driven power-user interface | `python -m tui.app` |
| 🌐 **HTTP API** | REST + WebSocket for integrations | `http://127.0.0.1:8085` |

### TUI Keybindings

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| `Ctrl+K` | Command palette | `Ctrl+S` | Force search |
| `Ctrl+L` | Clear chat | `Ctrl+D` | Deep thinking |
| `Ctrl+T` | Toggle metrics | `Ctrl+E` | Export chat |
| `F1` | Help | `Ctrl+Q` | Quit |

---

## 🔌 API Reference

```bash
# Health check
curl http://127.0.0.1:8085/health

# Send message
curl -X POST http://127.0.0.1:8085/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain quantum computing"}'

# Get cognitive state (entropy, surprise, varentropy)
curl http://127.0.0.1:8085/cognitive

# WebSocket streaming
wscat -c ws://127.0.0.1:8085/ws
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health & Ollama status |
| `/chat` | POST | Send message, get response |
| `/ws` | WS | Real-time bidirectional chat |
| `/cognitive` | GET | Entropy, surprise, confidence |
| `/belief` | GET | Active Inference belief state |
| `/memory` | GET | Memory statistics |

---

## 📁 Project Structure

```
AVA/
├── 📂 config/               # Configuration files
│   ├── cortex_medulla.yaml  # Main config
│   └── tools.yaml           # Tool definitions
├── 📂 docs/                 # Documentation
│   ├── GETTING_STARTED.md   # Quick start guide
│   ├── ARCHITECTURE.md      # Sentinel architecture
│   └── API_EXAMPLES.md      # API reference
├── 📂 src/                  # Python source (TUI/tools)
│   ├── core/                # Cortex-Medulla system
│   ├── hippocampus/         # Titans memory
│   └── tools/               # Tool implementations
├── 📂 tui/                  # Terminal UI (Textual)
├── 📂 ui/                   # Desktop GUI (Tauri + Next.js)
│   └── src-tauri/           # Rust backend
│       └── src/engine/      # Cognitive engine
├── 📂 tests/                # Test suite
└── 📄 README.md             # You are here
```

---

## ⚙️ Configuration

Edit `config/cortex_medulla.yaml`:

```yaml
cognitive:
  fast_model: "gemma3:4b"        # Medulla (fast)
  deep_model: "qwen2.5:32b"      # Cortex (deep)
  surprise_threshold: 0.5        # Routing threshold

search:
  enabled: true
  min_sources: 3                 # Verify with N sources

agency:
  epistemic_weight: 0.6          # Curiosity level
  pragmatic_weight: 0.4          # Goal focus

thermal:
  max_gpu_power_percent: 15      # Safe for laptops
```

See [CONFIGURATION.md](docs/CONFIGURATION.md) for all options.

---

## 📊 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU VRAM** | 4GB | 8GB+ |
| **System RAM** | 8GB | 16GB+ |
| **Storage** | 10GB | 50GB |
| **OS** | Windows 10 / Linux | Windows 11 / Ubuntu 22.04 |

### VRAM Budget (4GB GPU)

```
Component           │ Resident  │ Peak
────────────────────┼───────────┼──────────
System Overhead     │   300 MB  │   300 MB
Medulla (gemma3:4b) │ 2,000 MB  │ 2,000 MB
Embedding Model     │   200 MB  │   200 MB
Titans Memory       │   100 MB  │   100 MB
────────────────────┼───────────┼──────────
Total               │ 2,600 MB  │ 2,600 MB
Headroom            │ 1,400 MB  │ 1,400 MB
```

---

## 🛠️ Troubleshooting

<details>
<summary><strong>❌ "Ollama is not running"</strong></summary>

```bash
# Start Ollama server
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
```
</details>

<details>
<summary><strong>❌ "No models available"</strong></summary>

```bash
# Pull required models
ollama pull gemma3:4b
ollama pull nomic-embed-text

# Verify models
ollama list
```
</details>

<details>
<summary><strong>❌ "Port 8085 already in use"</strong></summary>

```bash
# Windows
netstat -ano | findstr :8085
taskkill /F /PID <pid>

# Linux/macOS
lsof -i :8085
kill -9 <pid>
```
</details>

<details>
<summary><strong>❌ "Out of GPU memory"</strong></summary>

```bash
# Use smaller model
export OLLAMA_MODEL=gemma2:2b

# Or limit GPU memory
export AVA_GPU_MEMORY_LIMIT=3000
```
</details>

See [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for more solutions.

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [**Getting Started**](docs/GETTING_STARTED.md) | Installation & first steps |
| [**Architecture**](docs/ARCHITECTURE.md) | Sentinel architecture deep-dive |
| [**Configuration**](docs/CONFIGURATION.md) | All configuration options |
| [**API Examples**](docs/API_EXAMPLES.md) | HTTP/WebSocket examples |
| [**TUI Guide**](docs/TUI_USER_GUIDE.md) | Terminal UI reference |
| [**Environment Variables**](docs/ENVIRONMENT_VARIABLES.md) | All env vars |
| [**Troubleshooting**](docs/TROUBLESHOOTING.md) | Common issues |

---

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) first.

```bash
# Fork the repo, then:
git clone https://github.com/YOUR_USERNAME/AVA.git
cd AVA
pip install -e ".[dev]"
pre-commit install

# Make changes, then:
pytest                    # Run tests
cargo test               # Rust tests
git commit -m "feat: your feature"
git push origin your-branch
```

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

<table>
<tr>
<td align="center"><strong>Research</strong></td>
<td align="center"><strong>Technology</strong></td>
</tr>
<tr>
<td>

- **Titans** — Test-Time Learning (Google, 2025)
- **Entropix** — Entropy-Based Routing
- **Active Inference** — Free Energy Principle (Friston)
- **Mamba** — State Space Models

</td>
<td>

- [Ollama](https://ollama.ai/) — Local LLM inference
- [Tauri](https://tauri.app/) — Desktop framework
- [Textual](https://textual.textualize.io/) — TUI framework
- [Next.js](https://nextjs.org/) — React framework

</td>
</tr>
</table>

---

<p align="center">
  <sub>Built with ❤️ for the research community</sub>
</p>

<p align="center">
  <a href="https://github.com/NAME0x0/AVA/stargazers">⭐ Star this repo</a> •
  <a href="https://github.com/NAME0x0/AVA/issues">🐛 Report Bug</a> •
  <a href="https://github.com/NAME0x0/AVA/issues">💡 Request Feature</a>
</p>
