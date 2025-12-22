# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AVA is a **research-grade AI assistant** with a **biomimetic dual-brain architecture** inspired by the human nervous system. It runs locally on constrained hardware (4GB VRAM) and prioritizes accuracy over speed.

**Core Paradigm:** Cortex-Medulla Architecture - fast reflexive responses for simple queries, deep reasoning for complex ones.

## Architecture (v3)

### Dual-Brain System

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
| **Medulla** | `src/core/medulla.py` | Always-on sensory processing, surprise calculation |
| **Cortex** | `src/core/cortex.py` | Deep reasoning via 70B models on 4GB GPU |
| **Bridge** | `src/core/bridge.py` | Projects Medulla state to Cortex embeddings |
| **Agency** | `src/core/agency.py` | Active Inference for autonomous behavior |
| **Titans** | `src/hippocampus/titans.py` | Test-time learning, infinite context |
| **System** | `src/core/system.py` | Orchestrates all components (AVACoreSystem) |
| **Clean API** | `src/ava/` | Simplified interface for external use |

### VRAM Budget (RTX A2000 4GB)

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

## Commands

### Installation

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Ensure Ollama is running
ollama pull gemma3:4b
```

### Running

```bash
# HTTP API Server (primary)
python server.py
python server.py --host 0.0.0.0 --port 8080

# Terminal UI (power users)
python run_tui.py
python run_tui.py --backend http://localhost:8085 --debug

# Core System with full architecture
python run_core.py --simulation

# Desktop GUI (Tauri + Next.js)
cd ui && npm run tauri dev
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/status` | GET | System status |
| `/chat` | POST | Send message (auto-routes Medulla/Cortex) |
| `/think` | POST | Force deep thinking (Cortex) |
| `/tools` | GET | List available tools |
| `/ws` | WebSocket | Streaming chat |

### Testing

```bash
pytest tests/
pytest --cov=src tests/
```

### Python API

```python
from ava import AVA
import asyncio

async def main():
    ava = AVA()
    await ava.start()

    response = await ava.chat("What is Python?")  # Auto-routes
    response = await ava.think("Explain quantum computing")  # Force Cortex

    await ava.stop()

asyncio.run(main())
```

## Configuration

### Main Config: `config/cortex_medulla.yaml`

Key settings:
- `development.simulation_mode` - Enable/disable simulation (testing mode)
- `search_first.enabled` - Web search as default for informational queries
- `thermal.max_gpu_power_percent` - GPU power limit (15% for RTX A2000)
- `agency.epistemic_weight` - Curiosity-driven behavior (0.6 = high curiosity)

### Search-First Paradigm

Web search is the DEFAULT action for informational queries:
- Question words trigger search: "what", "when", "where", "who", "how", "why"
- Minimum 3 sources, 70% agreement threshold for facts
- Falls back to internal generation if search fails

## Key Implementation Details

### Active Inference (Agency)

The system uses Free Energy Principle for autonomous behavior:
- **Belief States**: Probabilistic distributions over user intent
- **Policy Selection**: Minimizes Expected Free Energy G(π)
- **Available Policies**: `PRIMARY_SEARCH`, `REFLEX_REPLY`, `DEEP_THOUGHT`, `WEB_BROWSE`, `SELF_MONITOR`, `THERMAL_CHECK`, `SYSTEM_COMMAND`

### Titans Neural Memory

Test-time learning for infinite context:
- 3-layer MLP that learns during inference
- Fixed 200MB footprint regardless of conversation length
- Surprise-weighted gradient updates

### Thermal Self-Preservation

GPU monitoring with automatic throttling:
- Warning: 75°C, Throttle: 80°C, Pause: 85°C
- Power capped at 15% (10.5W on RTX A2000)

### System Command Safety

All system commands require explicit user confirmation:
```yaml
blocked_system_commands:
  - "rm -rf", "del /f", "format", "shutdown", "reboot"
```

## Development Guidelines

### When modifying the core loop:
1. Changes affect `src/core/system.py` (AVACoreSystem)
2. Component initialization order: Titans → Medulla → Cortex → Bridge → Agency → Thermal → Episodic
3. Policy handlers registered via `_register_action_callbacks()`

### When adding new tools:
1. Add tool class to `src/ava/tools.py` inheriting from `Tool`
2. Implement `execute()` async method
3. Register via `ToolManager`

### When modifying routing logic:
1. Surprise thresholds in `config/cortex_medulla.yaml` → `medulla.low_surprise_threshold` / `high_surprise_threshold`
2. Policy costs in `agency.cortex_effort_cost`, `search_effort_cost`

## Legacy Systems

Legacy code from v1/v2 has been archived to the `legacy/` directory:
- `legacy/developmental/` - Stage tracking (INFANT→MATURE)
- `legacy/emotional/` - Emotion processing
- `legacy/output/` - Articulation filtering
- `legacy/memory/` - Old memory system (replaced by Titans)
- `legacy/v2_core/` - V2 Cortex/Medulla implementations
- `legacy/old_servers/` - Previous API server versions

These are preserved for reference but not active in v3.

## Key Constraints

**4GB VRAM Limit**: All design decisions prioritize minimal VRAM usage:
- 4-bit quantization mandatory
- Layer-wise inference via AirLLM
- 1-bit models for Medulla
- Knowledge distillation for efficiency
