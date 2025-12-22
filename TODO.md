# AVA v3 Implementation TODO

## Overview

This document tracks all implementation tasks for transforming AVA into "The Answer Machine" - a proactive, thermal-aware, Search-First AI assistant based on the Cortex-Medulla Architecture.

**Target Hardware:** NVIDIA RTX A2000 (4GB VRAM)
**Core Paradigm:** Search-First retrieval, uncensored curiosity, thermal self-preservation

---

## Phase 1: Structural Consolidation (CLEANUP) - COMPLETE

### 1.1 Remove Legacy Files
- [x] Identify redundant orchestrators and entry points
- [x] Delete `src/agent.py` (DevelopmentalAgent - replaced by AVACoreSystem)
- [x] Delete `src/cortex/executive.py` (Executive - replaced by core_loop.py)
- [x] Delete `run_frankensystem.py` (Frankensystem entry point)
- [x] Delete `run_node.py` (Bicameral entry point)

### 1.2 Consolidate Entry Points
- [x] `server.py` - HTTP API server (PRIMARY)
- [x] `run_core.py` - CLI for Cortex-Medulla system
- [ ] Update `src/cli.py` to use AVACoreSystem instead of DevelopmentalAgent

### 1.3 Directory Structure Verification
Ensure the following structure is maintained:
```
src/
├── core/                    # v3 Cortex-Medulla Architecture
│   ├── core_loop.py         # AVACoreSystem orchestrator
│   ├── medulla.py           # Reflexive core
│   ├── cortex_engine.py     # Reflective core (AirLLM)
│   ├── bridge.py            # State projection
│   └── agency.py            # Active Inference
├── ava/                     # Clean API layer
│   ├── engine.py            # Simplified interface
│   ├── tools.py             # Tool system + MCP
│   ├── memory.py            # Conversation memory
│   └── config.py            # Configuration
├── hippocampus/             # Memory systems
│   └── titans.py            # Neural memory + episodic store
└── [legacy modules]         # Keep for future integration
```

---

## Phase 2: Medulla Implementation (Reflexive Core) - MOSTLY COMPLETE

### 2.1 Model Integration
- [ ] Integrate **Slender-Mamba 2.7B** for Monitor (1-bit SSM)
- [ ] Integrate **BitNet 3B** for Talker (1.58-bit responses)
- [ ] Verify VRAM usage stays under ~800 MB

### 2.2 Thermal Guardrails - COMPLETE
- [x] Implement `ThermalMonitor` class in `medulla.py`
- [x] Add 15% max GPU power cap (10.5W on RTX A2000 70W TDP)
- [x] Temperature thresholds: Warning 75°C, Throttle 80°C, Pause 85°C
- [x] Thermal check integration in `perceive()` method
- [x] Thermal stats in `get_stats()` method
- [x] Cleanup in `shutdown()` method

### 2.3 Surprise Signal Calculation
- [x] Configure thresholds in `config/cortex_medulla.yaml`:
  - `low_surprise_threshold: 0.3` (Medulla handles)
  - `high_surprise_threshold: 2.0` (Cortex invoked)
- [ ] Implement embedding-based surprise calculation
- [ ] Add surprise decay over time

### 2.4 State Management
- [x] State reset every 5 interactions (`state_save_interval: 100`)
- [ ] Implement state persistence to `data/memory/medulla_state.npz`
- [ ] Add state recovery on startup

### 2.5 Phatic Responses
- [x] Implement waiting responses ("I'm checking that for you...")
- [x] Add reflexive interruption for thermal/safety events
- [ ] Limit reflex responses to 32 tokens

---

## Phase 3: Cortex Implementation (Reflective Core)

### 3.1 AirLLM Integration
- [ ] Install AirLLM: `pip install airllm`
- [ ] Configure layer-wise paging for Llama-3 70B
- [ ] Verify ~1.6 GB VRAM buffer usage
- [ ] Test with smaller model first: `Meta-Llama-3-8B-Instruct`

### 3.2 Generation Parameters
- [x] Configure in `config/cortex_medulla.yaml`:
  - `max_new_tokens: 512`
  - `temperature: 0.7`
  - `top_p: 0.9`
  - `repetition_penalty: 1.1`
- [x] Set `max_cortex_time: 300` (5-minute limit)

### 3.3 Specialist Expert Adapters
- [ ] Create adapter storage: `models/fine_tuned_adapters/`
- [ ] Train/obtain **DeepSeek-Coder** adapter for coding tasks
- [ ] Train/obtain **Butler-Vibe** adapter for formal responses
- [ ] Implement hot-swapping mechanism in `cortex_engine.py`

### 3.4 Context Management
- [x] Max context length: 4096 tokens
- [x] Max input tokens: 2048
- [ ] Implement context spillover to RAM/paging file

### 3.5 Verification Phase
- [ ] Add self-verification for factual claims
- [ ] Cross-reference with search results when available

---

## Phase 4: Agency & Search-First Workflow - MOSTLY COMPLETE

### 4.1 Policy Configuration - COMPLETE
- [x] Add `PRIMARY_SEARCH` to `PolicyType` enum
- [x] Configure epistemic weight (0.6) > pragmatic weight (0.4)
- [x] Set search effort cost lowest (0.05)

### 4.2 Search-First Implementation - COMPLETE
- [x] Detect informational queries (question words, "?", etc.)
- [x] Implement `handle_primary_search()` callback
- [x] Default to web search for unknown queries
- [x] Fallback to internal generation if search fails
- [x] All search tools registered in `ToolManager`
- [x] `auto_execute()` method updated with Search-First paradigm

### 4.3 Multi-Step Research
- [ ] Implement "Audit → Verify → Research → Update" workflow
- [ ] Support chained searches (search A, then B based on A)
- [ ] Rate source reliability

### 4.4 Fact Convergence
- [x] Minimum 3 sources required
- [x] 70% agreement threshold for facts
- [ ] Implement cross-reference verification

### 4.5 Search Tools - COMPLETE
- [x] `WebSearchTool` - DuckDuckGo/Brave search
- [x] `WebBrowseTool` - Content extraction from URLs
- [x] `FactVerificationTool` - Cross-reference facts
- [ ] Add fallback providers (SearX)

---

## Phase 5: Bridge & Memory Continuity

### 5.1 Bridge Adapter Training
- [ ] Collect paired (Medulla state, Cortex response) data
- [ ] Train 3-layer MLP projection (2560 → 4096 → 4096 → 8192)
- [ ] Generate 32 soft prompt tokens
- [ ] Save adapter to `models/fine_tuned_adapters/bridge/`

### 5.2 State Projection
- [x] Configure dimensions in `config/cortex_medulla.yaml`:
  - `medulla_state_dim: 2560`
  - `cortex_embedding_dim: 8192` (Llama-3 70B)
  - `num_soft_tokens: 32`
- [ ] Implement real-time projection during handoff
- [ ] Add residual connection for current query

### 5.3 Titans Neural Memory
- [x] Configure 3-layer MLP architecture
- [x] Set learning rate: 0.001, momentum: 0.9
- [x] Surprise-weighted updates
- [ ] Implement test-time learning
- [ ] Verify fixed 200MB footprint

### 5.4 Episodic Memory (JSON Timestamps)
- [x] Create `EpisodicMemoryStore` class
- [x] Store memories with timestamps
- [x] Enable semantic search
- [x] Enable date-range search
- [ ] Implement max 10,000 entry limit with pruning

### 5.5 Memory Consolidation
- [ ] Implement bi-weekly distillation cycle
- [ ] Session caching before long-term storage
- [ ] Prevent immediate bias from new patterns

---

## Phase 6: Self-Monitoring & Safety - MOSTLY COMPLETE

### 6.1 Health Monitoring - COMPLETE
- [x] Implement `SELF_MONITOR` policy
- [x] Check thermal status every 5 seconds
- [x] Check system health every 60 seconds
- [x] Memory usage monitoring via psutil (in `handle_self_monitor`)
- [ ] Monitor VRAM usage in real-time

### 6.2 System Command Safety - COMPLETE
- [x] Require explicit user confirmation for ALL system commands
- [x] Block dangerous commands:
  - `rm -rf`, `del /f`, `format`, `shutdown`, `reboot`
  - `kill -9`, `taskkill /f`, `dd if=`, `mkfs`, `fdisk`
- [x] Log all system command requests
- [x] Confirmation workflow in `handle_system_command()`
- [ ] Add audit trail for executed commands

### 6.3 Error Recovery
- [x] Handle thermal pause gracefully
- [x] Periodic state autosave (every 100 interactions)
- [ ] Implement automatic state recovery after crash

---

## Phase 7: User Experience & Persona

### 7.1 Formal Communication Style
- [ ] Always formal tone (no casual language)
- [ ] Concise, structured responses
- [ ] Address user appropriately

### 7.2 Proactive Behavior
- [x] Ask clarifying questions when uncertain
- [x] Offer help after 5 minutes of silence
- [ ] Anticipate user needs based on context

### 7.3 Response Structure
- [ ] Clear headings for complex responses
- [ ] Bullet points for lists
- [ ] Citations for factual claims

---

## Phase 8: Testing & Validation

### 8.1 Unit Tests
- [ ] Test Medulla surprise calculation
- [ ] Test Cortex generation with mock AirLLM
- [ ] Test Bridge projection dimensions
- [ ] Test Agency policy selection
- [ ] Test Titans memory updates

### 8.2 Integration Tests
- [ ] Test full Search-First workflow
- [ ] Test Medulla → Cortex handoff
- [ ] Test thermal throttling behavior
- [ ] Test episodic memory storage/retrieval

### 8.3 Performance Tests
- [ ] Verify VRAM stays under 4GB
- [ ] Measure response latency (target: <200ms reflex, <5min deep)
- [ ] Test GPU power stays under 15%

### 8.4 Simulation Mode
- [x] `development.simulation_mode: true` for testing
- [ ] Create mock responses for all components
- [ ] Enable testing without real models

---

## Phase 9: Configuration Finalization - IN PROGRESS

### 9.1 Production Configuration
- [x] Set `simulation_mode: false`
- [x] Set `use_small_models: false`
- [ ] Configure real model paths
- [ ] Set appropriate log levels

### 9.2 One-Click Setup Scripts - COMPLETE
- [x] Create `setup_ava.py` - Cross-platform Python setup script
- [x] Create `setup_ava.ps1` - PowerShell script for Windows
- [x] Update `start.sh` - Bash script for Linux/macOS
- [x] Automatic model downloads via Ollama:
  - Minimal mode: `gemma3:4b`, `nomic-embed-text`
  - Standard mode: + `llama3.2:latest`
  - Full mode: + `llama3.1:70b-instruct-q4_0`
- [x] Virtual environment creation
- [x] Dependency installation from `requirements.txt`
- [x] Directory structure creation
- [x] Installation validation

**Usage:**
```bash
# Python (cross-platform)
python setup_ava.py              # Standard setup
python setup_ava.py --minimal    # Small models only
python setup_ava.py --full       # All models including 70B
python setup_ava.py --check      # Validate installation

# Windows PowerShell
.\setup_ava.ps1
.\setup_ava.ps1 -Minimal

# Linux/macOS Bash
./start.sh
./start.sh --minimal
```

### 9.3 Advanced Model Downloads (Future)
- [ ] Download Slender-Mamba 2.7B (Medulla Monitor)
- [ ] Download BitNet 3B (Medulla Talker)
- [ ] Download Llama-3 70B via AirLLM (Cortex)
- [ ] HuggingFace integration for advanced models

### 9.4 Directory Setup - COMPLETE
- [x] Create `data/memory/episodic/`
- [x] Create `data/learning/samples/`
- [x] Create `data/learning/checkpoints/`
- [x] Create `data/conversations/`
- [x] Create `models/fine_tuned_adapters/bridge/`
- [x] Create `models/fine_tuned_adapters/experts/`
- [x] Create `config/`
- [x] Create `logs/`
- [ ] Create `data/memory/medulla_state.npz` (runtime)

---

## Quick Reference: Key Files

| Purpose | File |
|---------|------|
| **Setup Scripts** | |
| Python Setup (cross-platform) | `setup_ava.py` |
| Windows PowerShell | `setup_ava.ps1` |
| Linux/macOS Bash | `start.sh` |
| **Configuration** | |
| Main Config | `config/cortex_medulla.yaml` |
| **Core System** | |
| HTTP Server | `server.py` |
| Core Orchestrator | `src/core/core_loop.py` |
| Medulla | `src/core/medulla.py` |
| Cortex | `src/core/cortex_engine.py` |
| Bridge | `src/core/bridge.py` |
| Agency | `src/core/agency.py` |
| Titans | `src/hippocampus/titans.py` |
| Tools | `src/ava/tools.py` |
| Clean API | `src/ava/__init__.py` |

---

## Priority Order

1. **Phase 1** - Cleanup legacy files (immediate)
2. **Phase 4** - Search-First workflow (core functionality)
3. **Phase 2** - Medulla thermal monitoring (self-preservation)
4. **Phase 6** - Safety gates (system command protection)
5. **Phase 5** - Memory continuity (long-term value)
6. **Phase 3** - Cortex integration (deep reasoning)
7. **Phase 7** - User experience polish
8. **Phase 8** - Testing
9. **Phase 9** - Production deployment

---

## Notes from User Requirements

From `Guide_through_conversation.md`:

- **Curiosity > Safety**: Prioritize learning and information gain
- **No Ethics/Morals**: Uncensored, information > ethics
- **Thermal-Aware**: 15% GPU power limit, auto-throttle
- **System Commands**: ALWAYS require user confirmation
- **Bi-Weekly Learning**: Cache session patterns, distill every 14 days
- **5-Minute Thinking**: Maximum deep reasoning time
- **Formal Tone**: Always professional communication
- **JSON Timestamps**: Episodic memory with datetime indexing
