# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AVA is a **research-grade AI assistant** with a **biomimetic dual-brain architecture** inspired by the human nervous system. It runs locally on constrained hardware (4GB VRAM) and prioritizes accuracy over speed.

**Core Paradigm:** Cortex-Medulla Architecture - fast reflexive responses for simple queries, deep reasoning for complex ones.

## Architecture (v4) - Sentinel

### Sentinel Cognitive Loop

AVA v4 implements the **Sentinel** architecture - a deliberate inference system that prioritizes accuracy over speed through formal verification and multi-agent simulation.

```
┌─────────────────────────────────────────────────────────────────┐
│                    SENTINEL REASONING LOOP                       │
├─────────────────────────────────────────────────────────────────┤
│  Stage 1: PERCEPTION (Medulla)                                  │
│  ├─ Fast 8B model captures query                                │
│  └─ Calculates Predictive Entropy & Surprise                    │
│                           ↓                                      │
│  Stage 2: APPRAISAL (Active Inference)                          │
│  ├─ If entropy > threshold → enter "Thinking" state             │
│  └─ No tokens emitted yet                                       │
│                           ↓                                      │
│  Stage 3: SIMULATION (Mental Sandbox)                           │
│  ├─ Agent A (Thinker): Drafts solution                          │
│  ├─ Agent B (Verifier): Z3 formal logic check                   │
│  └─ Agent C (Butler): User context personalization              │
│                           ↓                                      │
│  Stage 4: CONSENSUS (Formal Output)                             │
│  ├─ Loop until all agents agree (max 5 cycles)                  │
│  └─ Emit only verified, personalized output                     │
└─────────────────────────────────────────────────────────────────┘
```

**Key Principles:**
- **Adaptive Verification:** Z3 formal checks triggered by entropy/complexity
- **Consensus-Driven:** Output only after multi-agent agreement
- **Accuracy Priority:** No latency budget - correctness over speed

### Dual System Architecture

**Primary: Unified Rust Backend (v4)**
```
┌─────────────────────────────────────────────────────────────────┐
│  AVA Desktop App (Single Executable)                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Embedded Rust Backend (Axum HTTP on 127.0.0.1:8085)        ││
│  │  - Ollama integration (gemma3:4b, llama3.2)                 ││
│  │  - Active Inference metrics                                  ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Next.js Frontend (Tauri WebView)                           ││
│  │  - Neural activity visualization                            ││
│  │  - Free Energy metrics dashboard                            ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

**Secondary: Python Core System (research/development)**
```
User Input → Medulla (surprise calc) → Low Surprise → Quick Reply (<200ms)
                                     → High Surprise → Cortex (70B via AirLLM, ~3.3s/token)
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **Medulla** | `src/core/medulla.py` | Always-on sensory processing, surprise calculation |
| **Cortex** | `src/core/cortex.py` | Deep reasoning via 70B models on 4GB GPU |
| **Bridge** | `src/core/bridge.py` | Projects Medulla state to Cortex embeddings |
| **Agency** | `src/core/agency.py` | Active Inference (Free Energy Principle) |
| **Titans** | `src/hippocampus/titans.py` | Test-time learning, infinite context |
| **System** | `src/core/system.py` | Orchestrates all components (AVACoreSystem) |
| **Verification** | `src/core/verification.py` | Z3 formal verifier (MATH-VF) |
| **Sandbox** | `src/core/sandbox.py` | Mental Sandbox multi-agent simulation |
| **Rust Backend** | `ui/src-tauri/src/engine/` | Embedded Axum HTTP server |
| **Rust Titans** | `ui/src-tauri/src/engine/titans.rs` | MLP memory buffer (Rust) |
| **Rust Agency** | `ui/src-tauri/src/engine/agency.rs` | FEP policy evaluator (Rust) |

## Commands

### Installation

```bash
# Python environment
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Ollama (required for all modes)
ollama pull gemma3:4b
ollama pull llama3.2:latest

# Desktop app dependencies
cd ui && npm install
```

### Running

```bash
# Desktop GUI (Tauri + Next.js) - recommended
cd ui && npm run tauri dev

# Terminal UI (power users)
python -m tui.app
python -m tui.app --backend http://localhost:8085 --debug

# Python core system (research)
python run_core.py --simulation
```

### Testing

```bash
# All unit tests
pytest tests/unit -v

# Single test file
pytest tests/unit/test_agency.py -v

# With coverage
pytest tests/unit --cov=src --cov-report=term-missing

# Integration tests (skip slow)
pytest tests/integration -v -m "not slow"

# Skip tests requiring GPU
pytest -m "not gpu"
```

### Linting & Formatting

```bash
# Python
ruff check src/                    # Lint
ruff check src/ --fix              # Auto-fix
black src/ --check                 # Format check
black src/                         # Format
mypy src/ tui/ --config-file pyproject.toml

# Frontend (from ui/)
npm run lint
npm run type-check
npm run format:check

# Rust (from ui/src-tauri/)
cargo check
cargo clippy -- -D warnings
cargo fmt --check
```

### Building

```bash
# Python package
python -m build

# Desktop app
cd ui && npm run tauri:build

# Full release with sidecar
cd ui && npm run tauri:build:release
```

## Configuration

### Main Config: `config/cortex_medulla.yaml`

Key settings:
- `backend.mode` - Backend selection: `ollama` (default), `native`, or `hybrid`
- `development.simulation_mode` - Enable testing without real models
- `search_first.enabled` - Web search as default for informational queries
- `thermal.max_gpu_power_percent` - GPU power limit (15% for RTX A2000)
- `agency.epistemic_weight` - Curiosity weight (0.6 = high curiosity)
- `medulla.low_surprise_threshold` / `high_surprise_threshold` - Routing thresholds

### Backend Options

| Mode | Description | Requirements |
|------|-------------|--------------|
| `ollama` | Uses Ollama API (recommended) | Ollama installed with models |
| `native` | AirLLM + Mamba + llama-cpp | Optional packages from requirements.txt |
| `hybrid` | Ollama for Cortex, native for Medulla | Both installed |

To use native models, uncomment in `requirements.txt`:
```bash
pip install airllm mamba-ssm causal-conv1d llama-cpp-python
```

### Environment Variables

- `AVA_SIMULATION_MODE=true` - Run tests without models
- `SKIP_ENV_VALIDATION=true` - Skip Next.js env validation during builds

## Development Guidelines

### Component Initialization Order
In `src/core/system.py` (AVACoreSystem):
1. Titans → Medulla → Cortex → Bridge → Agency → Thermal → Episodic

### Adding New Tools
1. Add tool class to `src/ava/tools.py` inheriting from `Tool`
2. Implement `execute()` async method
3. Register via `ToolManager`

### Policy Handlers
Register via `_register_action_callbacks()` in `src/core/system.py`

Available policies: `PRIMARY_SEARCH`, `REFLEX_REPLY`, `DEEP_THOUGHT`, `WEB_BROWSE`, `SELF_MONITOR`, `THERMAL_CHECK`, `SYSTEM_COMMAND`, `VERIFY_LOGIC`, `SIMULATE_OUTCOME`, `FORMAL_CHECK`

### Sentinel Verification
The Sentinel architecture adds formal verification via Z3:
- **Adaptive Trigger:** Verification runs when `entropy > 0.7` OR code/logic detected
- **MATH-VF Critic:** Translates claims to SMT-LIB, runs Z3 `s.check()`
- **Consensus Loop:** Max 5 cycles until Thinker + Verifier + Butler agree

### Rust Backend Development
- HTTP server: `ui/src-tauri/src/engine/`
- Endpoints: `/health`, `/chat`, `/cognitive`, `/memory`, `/belief`, `/stats`
- Uses Axum framework with Tokio runtime

## API Endpoints

**Python server (if running)**: Default port 8080
**Rust embedded server**: `http://127.0.0.1:8085`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/chat` | POST | Send message (auto-routes) |
| `/cognitive` | GET | Entropy, surprise, varentropy |
| `/memory` | GET | Memory statistics |
| `/belief` | GET | Active Inference belief state |

## Key Constraints

**4GB VRAM Limit** (RTX A2000):
- 4-bit quantization mandatory
- Layer-wise inference via AirLLM
- 1-bit models for Medulla
- Total resident: ~2,050 MB, Peak: ~3,650 MB

## TUI Keybindings

| Key | Action |
|-----|--------|
| `Ctrl+K` | Command palette |
| `Ctrl+S` | Force search |
| `Ctrl+D` | Deep think (Cortex) |
| `Ctrl+T` | Toggle metrics |
| `F1` | Help |

## Legacy Systems

Archived code in `legacy/`:
- `legacy/v2_core/` - V2 Cortex/Medulla implementations
- `legacy/python_servers/` - Previous Python server versions
- `legacy/emotional/`, `legacy/memory/` - Superseded systems
