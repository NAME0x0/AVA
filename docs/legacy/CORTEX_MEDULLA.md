# AVA v3: Cortex-Medulla Architecture

## Technical Implementation Blueprint

This document describes the implementation of the Cortex-Medulla architecture for Project AVA v3, designed for autonomous, always-on operation on constrained hardware (NVIDIA RTX A2000 4GB VRAM).

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Component Specifications](#component-specifications)
4. [VRAM Budget Analysis](#vram-budget-analysis)
5. [Implementation Guide](#implementation-guide)
6. [Configuration Reference](#configuration-reference)
7. [Performance Characteristics](#performance-characteristics)

---

## Executive Summary

The Cortex-Medulla architecture achieves high-fidelity autonomous AI on a 4GB GPU through a dual-system approach:

| Component | Function | Model | VRAM | Latency |
|-----------|----------|-------|------|---------|
| **Medulla** | Always-on reflexes | 1-bit Mamba SSM | ~800 MB | <200ms |
| **Cortex** | Deep reasoning | 70B via AirLLM | ~1.6 GB (paged) | ~3s/token |
| **Bridge** | State projection | MLP Adapter | ~50 MB | <1ms |
| **Agency** | Autonomous drive | Active Inference | ~10 MB | <10ms |
| **Titans** | Neural memory | Test-time learning | ~200 MB | <50ms |

**Total VRAM**: ~2,950 MB / 4,096 MB available

---

## Architecture Overview

```
                              ┌─────────────────────────────────────┐
                              │         AVA CORE SYSTEM             │
                              │    (Cortex-Medulla Architecture)    │
                              └─────────────────────────────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │                         │                         │
                    ▼                         ▼                         ▼
        ┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
        │      MEDULLA      │    │      CORTEX       │    │      AGENCY       │
        │  (Reflexive Core) │    │ (Reflective Core) │    │(Active Inference) │
        │                   │    │                   │    │                   │
        │  • 1-bit Mamba    │◄──►│  • 70B Llama-3    │    │  • VFE Minimizer  │
        │  • BitNet Talker  │    │  • AirLLM Paging  │    │  • Policy Select  │
        │  • O(1) State     │    │  • Layer-wise     │    │  • Belief Update  │
        │                   │    │                   │    │                   │
        │  VRAM: ~1.5 GB    │    │  VRAM: ~1.6 GB    │    │  RAM: ~10 MB      │
        │  (Resident)       │    │  (On-Demand)      │    │                   │
        └─────────┬─────────┘    └─────────▲─────────┘    └─────────┬─────────┘
                  │                        │                        │
                  │         ┌──────────────┴──────────────┐         │
                  │         │           BRIDGE            │         │
                  │         │    (State Projection)       │         │
                  └────────►│                             │◄────────┘
                            │  Mamba State → Transformer  │
                            │  Soft Prompts (32 tokens)   │
                            │                             │
                            │  VRAM: ~50 MB               │
                            └──────────────┬──────────────┘
                                           │
                            ┌──────────────▼──────────────┐
                            │          TITANS             │
                            │     (Neural Memory)         │
                            │                             │
                            │  • Test-time Learning       │
                            │  • Surprise-driven Updates  │
                            │  • Fixed Memory Footprint   │
                            │                             │
                            │  VRAM: ~200 MB              │
                            └─────────────────────────────┘
```

---

## Component Specifications

### 1. Medulla (The Reflexive Core)

**Purpose**: Always-on sensory processing with O(1) memory complexity.

**Architecture**:
- **Monitor Model**: Bi-Mamba 2.7B (1.58-bit quantization)
- **Talker Model**: BitNet 3B for quick responses
- **State**: Fixed-size SSM hidden state (2560 × 16)

**Key Features**:
- Linear time complexity O(N) - no attention quadratic scaling
- Constant memory O(1) - no KV cache growth
- Sub-200ms response latency
- Continuous operation without degradation

**Code Location**: `src/core/medulla.py`

```python
from src.core import Medulla, MedullaConfig

config = MedullaConfig(
    hidden_dim=2560,
    low_surprise_threshold=0.3,
    high_surprise_threshold=2.0,
)
medulla = Medulla(config)
surprise, response = await medulla.perceive(input_text="Hello")
```

### 2. Cortex (The Reflective Core)

**Purpose**: Deep reasoning via 70B models through layer-wise inference.

**Architecture**:
- **Model**: Llama-3 70B Instruct (4-bit quantization)
- **Inference**: AirLLM layer paging from System RAM
- **Buffer**: Single layer loaded to VRAM at a time

**Key Features**:
- Runs 70B model on 4GB GPU
- ~3.3 seconds per generated token
- Quality equivalent to full model
- Activated only for complex tasks

**Code Location**: `src/core/cortex_engine.py`

```python
from src.core import Cortex, CortexConfig

config = CortexConfig(
    model_name="meta-llama/Meta-Llama-3-70B-Instruct",
    compression="4bit",
)
cortex = Cortex(config)
result = await cortex.generate(prompt="Explain quantum computing")
```

### 3. Bridge (State Projection)

**Purpose**: Instant context transfer from Medulla to Cortex.

**Architecture**:
- **Adapter**: 3-layer MLP (2560 → 4096 → 4096 → 262144)
- **Output**: 32 soft prompt tokens for Cortex
- **Projection**: Mamba state → Transformer embeddings

**Key Features**:
- O(1) context transfer (vs O(N) pre-fill)
- Trained adapter maps representation spaces
- Eliminates minutes-long pre-fill for long conversations

**Code Location**: `src/core/bridge.py`

```python
from src.core import Bridge, BridgeConfig

bridge = Bridge()
cortex_input = await bridge.prepare_cortex_input(
    medulla_state=medulla.get_state_vector(),
    current_query="What's the meaning of life?",
    conversation_history=history,
)
```

### 4. Agency (Active Inference Controller)

**Purpose**: Autonomous behavior through Free Energy minimization.

**Architecture**:
- **Framework**: POMDP with Expected Free Energy
- **Beliefs**: Distribution over hidden states
- **Policies**: REFLEX_REPLY, DEEP_THOUGHT, USE_TOOL, WAIT, etc.

**Key Features**:
- Intrinsic motivation (not just reactive)
- Uncertainty drives action (epistemic value)
- Goals drive action (pragmatic value)
- Proactive engagement without prompts

**Code Location**: `src/core/agency.py`

```python
from src.core import ActiveInferenceController, Observation

agency = ActiveInferenceController()
observation = Observation(text="Hello", surprise_signal=0.5)
policy, result = await agency.process_observation(observation)
```

### 5. Titans (Neural Memory)

**Purpose**: Infinite context via test-time learning.

**Architecture**:
- **Memory**: 3-layer MLP with learnable weights
- **Learning**: Gradient updates during inference
- **Signal**: Surprise-weighted update magnitude

**Key Features**:
- Fixed memory footprint (~200 MB)
- Context compressed into weights, not tokens
- Operates indefinitely without memory growth

**Code Location**: `src/hippocampus/titans.py`

---

## VRAM Budget Analysis

### RTX A2000 Specifications
| Specification | Value |
|--------------|-------|
| Total VRAM | 4,096 MB |
| Architecture | Ampere (GA106) |
| Memory Bandwidth | 192 GB/s |
| CUDA Cores | 3,328 |
| PCIe | Gen 4 x16 |

### VRAM Allocation
| Component | Size | Status |
|-----------|------|--------|
| System/CUDA Overhead | 300 MB | Reserved |
| Medulla (Monitor) | 800 MB | Resident |
| Medulla (Talker) | 700 MB | Resident |
| Titans Memory | 200 MB | Resident |
| Bridge Adapter | 50 MB | Resident |
| Cortex Layer Buffer | 1,600 MB | On-demand |
| **Total Resident** | **2,050 MB** | |
| **Peak (Cortex Active)** | **3,650 MB** | |
| **Headroom** | **446 MB** | Safety margin |

---

## Implementation Guide

### Quick Start

```bash
# 1. Clone and install
cd AVA
pip install -r requirements.txt

# 2. Install optional dependencies for full functionality
pip install airllm mamba-ssm pymdp titans-pytorch

# 3. Run in simulation mode (no GPU required)
python run_core.py --simulation

# 4. Run with full hardware
python run_core.py
```

### Configuration

Edit `config/cortex_medulla.yaml`:

```yaml
medulla:
  monitor_model: "slender-mamba-2.7b"
  high_surprise_threshold: 2.0

cortex:
  model_name: "meta-llama/Meta-Llama-3-70B-Instruct"
  compression: "4bit"

agency:
  pragmatic_weight: 0.6
  epistemic_weight: 0.4
```

### Programmatic Usage

```python
import asyncio
from src.core import AVACoreSystem, CoreConfig

async def main():
    # Create system
    config = CoreConfig()
    ava = AVACoreSystem(config)
    
    # Initialize
    await ava.initialize()
    
    # Process input
    response = await ava.process_input("Explain quantum entanglement")
    print(response)
    
    # Get statistics
    stats = ava.get_stats()
    
    # Shutdown
    await ava.shutdown()

asyncio.run(main())
```

---

## Performance Characteristics

### Latency Profiles

| Operation | Typical Latency | Notes |
|-----------|----------------|-------|
| Medulla perception | 10-50 ms | State update |
| Medulla reflex | 100-200 ms | Quick response |
| Bridge projection | <1 ms | MLP forward pass |
| Agency inference | 1-10 ms | Policy selection |
| Cortex generation | 3-4 s/token | Layer-wise paging |

### Expected Scenarios

**Routine Query** (90% of interactions):
- Total time: 200-500 ms
- Path: Input → Medulla → Reflex Response
- VRAM: ~2 GB constant

**Complex Query** (10% of interactions):
- Total time: 30s-5min depending on response length
- Path: Input → Medulla → High Surprise → Cortex
- VRAM: Peaks at ~3.6 GB during Cortex

### Thermal Characteristics

| Mode | GPU Utilization | Power Draw |
|------|-----------------|------------|
| Idle | <5% | ~10W |
| Medulla Active | 20-30% | 15-25W |
| Cortex Active | 100% | 70W (TDP) |

---

## File Structure

```
src/core/
├── __init__.py          # Module exports
├── medulla.py           # Reflexive core (Mamba SSM)
├── cortex_engine.py     # Reflective core (AirLLM)
├── bridge.py            # State projection (MLP adapter)
├── agency.py            # Active Inference controller
└── core_loop.py         # Main system orchestrator

config/
└── cortex_medulla.yaml  # Architecture configuration

run_core.py              # CLI runner
```

---

## Migration from Frankensystem

The v3 architecture replaces:

| Old Component | New Component | Reason |
|--------------|---------------|--------|
| `Executive` | `Agency` | Active Inference vs reactive |
| `ConsciousStream` | `Medulla` | O(1) vs O(N²) memory |
| `DevelopmentalAgent.interact()` | `AVACoreSystem.process_input()` | Continuous loop vs request/response |
| Multiple memory systems | `Titans` | Unified neural memory |

---

## References

1. **BitNet b1.58**: "The Era of 1-bit LLMs" - Microsoft Research, 2024
2. **Mamba**: "Mamba: Linear-Time Sequence Modeling" - Gu & Dao, 2023
3. **AirLLM**: "Run 70B LLMs on 4GB GPU" - Gavin Li, 2024
4. **Titans**: "Learning to Memorize at Test Time" - Google, 2025
5. **Active Inference**: "The Free Energy Principle in Mind, Brain, and Behavior" - Parr et al., 2022
