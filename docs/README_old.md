<p align="center">
  <img src="docs/assets/ava_logo.png" alt="AVA Logo" width="200" />
</p>

<h1 align="center">AVA: Adaptive Virtual Agent</h1>
<h3 align="center">A Neuro-Symbolic Cognitive Architecture for Continual Learning</h3>

<p align="center">
  <a href="#abstract">Abstract</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#research">Research</a> •
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-3.0.0-blue.svg" alt="Version" />
  <img src="https://img.shields.io/badge/python-3.10+-green.svg" alt="Python" />
  <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="License" />
  <img src="https://img.shields.io/badge/status-research-orange.svg" alt="Status" />
</p>

---

## 🚀 What's New in v3.0 - Cortex-Medulla Architecture

AVA v3 introduces a **biomimetic dual-system architecture** inspired by the human brain:

| Component | Role | Technology |
|-----------|------|------------|
| **Medulla** | Always-on reflexive processing | Mamba SSM / BitNet (1-bit) |
| **Cortex** | Deep reasoning on demand | AirLLM (70B in 4GB VRAM) |
| **Bridge** | Context handoff | Neural State Projection |
| **Agency** | Autonomous behavior | Active Inference / FEP |

### Quick Start (v3)

```bash
# Start the backend
python api_server_v3.py

# In another terminal, start the UI
cd ui
npm install
npm run dev

# Open http://localhost:3000
```

### New UI Features
- **Next.js 14** + **TypeScript** + **Tauri** (Rust) for native performance
- Real-time cognitive state visualization
- Active component indicator (Medulla vs Cortex)
- WebSocket streaming for responses
- ~2ms UI response time

See [docs/CORTEX_MEDULLA.md](docs/CORTEX_MEDULLA.md) for full architecture documentation.

---

## Abstract

**AVA** (Adaptive Virtual Agent) is a research platform implementing a neuro-symbolic cognitive architecture that combines recent advances in neural memory systems, metacognitive awareness, and continual learning. Unlike conventional language model deployments, AVA implements a biologically-inspired dual-loop processing system that enables:

- **Test-time learning** via Titans Neural Memory sidecars
- **Metacognitive awareness** through Entropix entropy analysis
- **Offline consolidation** using QLoRA fine-tuning during idle periods
- **Developmental progression** from novice to expert articulation
- **Emotional modulation** affecting learning rates and response styles

This architecture draws from cutting-edge research including Titans (2025), Nested Learning (Google, 2025), Toolformer (2023), Entropix (2024), and QLoRA (2023), synthesizing these approaches into a cohesive system capable of genuine continual improvement.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Theoretical Foundation](#2-theoretical-foundation)
3. [System Architecture](#3-system-architecture)
4. [Core Components](#4-core-components)
5. [Implementation Details](#5-implementation-details)
6. [Installation & Setup](#6-installation--setup)
7. [Usage Guide](#7-usage-guide)
8. [Configuration Reference](#8-configuration-reference)
9. [Developmental System (Legacy)](#9-developmental-system-legacy)
10. [Roadmap](#10-roadmap)
11. [Contributing](#11-contributing)
12. [Citation](#12-citation)
13. [License](#13-license)

---

## 1. Introduction

### 1.1 Motivation

Current large language models (LLMs) operate as static inference engines—they cannot learn from interactions, adapt to user preferences, or consolidate new knowledge without expensive retraining. AVA addresses this fundamental limitation by implementing a cognitive architecture that:

1. **Learns at test-time** without modifying base model weights
2. **Consolidates knowledge offline** through experience replay
3. **Monitors its own cognitive state** to guide behavior
4. **Develops progressively** like biological intelligence

### 1.2 Design Philosophy

AVA is built on three core principles:

| Principle | Implementation |
|-----------|----------------|
| **Separation of Concerns** | Conscious (online) vs Subconscious (offline) processing |
| **Multi-Timescale Learning** | Millisecond (memory) → Second (inference) → Hour (consolidation) |
| **Metacognitive Awareness** | Entropy-based self-monitoring drives adaptive behavior |

### 1.3 Key Contributions

- **Frankensystem Architecture**: Integration of Titans, Entropix, and QLoRA into a cohesive system
- **Hippocampus Module**: SQLite-backed episodic buffer with priority-based replay
- **Nightmare Engine**: Sleep-cycle consolidation with automatic QLoRA fine-tuning
- **Developmental Framework**: Stage-gated capability progression with emotional modulation

---

## 2. Theoretical Foundation

### 2.1 Titans: Learning to Memorize at Test Time

The Titans paper (arXiv:2501.00663, 2025) introduces **Neural Memory MLPs** that can learn during inference through surprise-driven weight updates:

```
M(x) = W₂ · σ(W₁ · x + b₁) + b₂

Update Rule: W ← W - η · s(x) · ∇L
Where s(x) = surprise signal
```

AVA implements this via the `TitansSidecar` class, which maintains a separate learnable memory module alongside the frozen LLM.

### 2.2 Entropix: Metacognitive Entropy Analysis

Entropix (2024) analyzes the **entropy landscape** of token probability distributions to classify cognitive states:

| State | Entropy (H) | Varentropy (V) | Interpretation |
|-------|-------------|----------------|----------------|
| **FLOW** | Low | Low | Confident, rote knowledge |
| **HESITATION** | Low | High | Knows answer, uncertain phrasing |
| **CONFUSION** | High | Low | Uniformly uncertain, hallucination risk |
| **CREATIVE** | High | High | Exploring diverse possibilities |

This classification drives dynamic behavior: CONFUSION triggers tool retrieval, HESITATION enables chain-of-thought, CREATIVE states are logged for training.

### 2.3 Nested Learning: Multi-Timescale Adaptation

Google's Nested Learning (2025) proposes separating parameters into **fast** and **slow** weights:

- **Fast Weights**: Session-level adaptation (lr=1e-3, momentum=0.9)
- **Slow Weights**: Permanent consolidation (lr=1e-5, momentum=0.999)

This prevents catastrophic forgetting while enabling rapid adaptation.

### 2.4 QLoRA: Efficient Fine-Tuning

QLoRA (arXiv:2305.14314) enables fine-tuning of quantized models through low-rank adaptation:

- **4-bit quantization** reduces memory footprint
- **LoRA adapters** (rank 8-64) enable parameter-efficient updates
- **Gradient checkpointing** allows training on consumer hardware

AVA uses QLoRA during "sleep" cycles to consolidate high-value experiences.

---

## 3. System Architecture

### 3.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AVA COGNITIVE ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     CONSCIOUS LOOP (Online)                         │    │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐       │    │
│  │  │ ENTROPIX │──▶│  TITANS  │───▶│  OLLAMA  │──▶│  TOOLS   │       │    │
│  │  │(Metacog) │    │(Sidecar) │    │  (LLM)   │    │(Registry)│       │    │
│  │  └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘       │    │
│  │       │               │               │               │             │    │
│  │       ▼               ▼               ▼               ▼             │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │                    EPISODIC BUFFER                          │    │    │
│  │  │           (SQLite: High-Surprise Event Storage)             │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    │ Idle → Sleep Trigger                   │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   SUBCONSCIOUS LOOP (Offline)                       │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │                   NIGHTMARE ENGINE                           │   │    │
│  │  │                                                              │   │    │
│  │  │   DROWSY → LIGHT_SLEEP → DEEP_SLEEP → REM → WAKING           │   │    │
│  │  │              │              │           │                    │   │    │
│  │  │         Fast Weights   Slow Weights  QLoRA                   │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow

1. **Input Processing**: User input → Embedding (nomic-embed-text)
2. **Memory Retrieval**: Query Titans Sidecar for context augmentation
3. **Inference**: Ollama LLM generates response with logprobs
4. **Metacognition**: Entropix analyzes entropy/varentropy → Cognitive state
5. **Learning Signal**: Surprise value computed from cognitive state
6. **Memory Update**: High-surprise inputs update Titans weights
7. **Buffering**: Significant episodes stored in SQLite buffer
8. **Consolidation**: Nightmare Engine processes buffer during idle

### 3.3 Module Dependencies

```
src/
├── cortex/                    # Conscious processing
│   ├── entropix.py           # Metacognitive entropy analysis
│   ├── executive.py          # High-level orchestration
│   ├── stream.py             # Real-time conscious stream
│   ├── dreaming.py           # Background consolidation
│   └── ollama_interface.py   # LLM API interface
│
├── hippocampus/              # Memory formation
│   ├── titans.py             # Neural memory sidecar
│   └── episodic_buffer.py    # SQLite replay buffer
│
├── subconscious/             # Offline processing
│   └── nightmare.py          # Sleep-cycle consolidation
│
├── learning/                 # Learning systems
│   ├── nested.py             # Fast/Slow weight manager
│   ├── qlora.py              # QLoRA training wrapper
│   ├── fine_tuning.py        # Training scheduler
│   └── continual.py          # Online learning
│
├── memory/                   # Persistent storage
│   ├── models.py             # Memory data structures
│   ├── semantic.py           # Vector database
│   └── episodic.py           # Event storage
│
├── emotional/                # Affective computing
│   ├── engine.py             # Emotional state machine
│   └── modulation.py         # Learning rate modulation
│
├── developmental/            # Growth progression
│   ├── stages.py             # Stage definitions
│   └── tracker.py            # Progress tracking
│
├── tools/                    # Capability system
│   ├── registry.py           # Tool management
│   └── base_tools.py         # Built-in tools
│
└── output/                   # Response generation
    └── articulation.py       # Speech clarity modulation
```

---

## 4. Core Components

### 4.1 Entropix (Metacognitive Module)

**Location**: `src/cortex/entropix.py`

The Entropix module computes Shannon entropy and varentropy from LLM logprobs to classify cognitive states in real-time.

```python
from src.cortex.entropix import Entropix, EntropixConfig

entropix = Entropix(EntropixConfig(
    high_entropy_threshold=3.0,
    high_varentropy_threshold=2.0,
))

# Diagnose from logprobs
state = entropix.diagnose(logprobs)
print(f"State: {state.label}")  # FLOW, HESITATION, CONFUSION, CREATIVE
print(f"Should use tools: {state.should_use_tools}")
```

**Key Features**:
- Real-time entropy/varentropy computation
- Cognitive state classification with confidence scores
- Action recommendations (tool use, CoT, temperature adjustment)
- Historical statistics tracking

### 4.2 Titans Sidecar (Neural Memory)

**Location**: `src/hippocampus/titans.py`

The Titans Sidecar implements test-time learning through a surprise-weighted neural memory MLP.

```python
from src.hippocampus.titans import TitansSidecar, TitansSidecarConfig

sidecar = TitansSidecar(TitansSidecarConfig(
    input_dim=768,
    learning_rate=1e-3,
    momentum=0.9,
))

# Retrieve memory-augmented context
augmented = sidecar.retrieve(query_embedding)

# Update memory with surprise signal
loss = sidecar.memorize(embedding, surprise=2.5)
```

**Key Features**:
- PyTorch and NumPy backends (auto-selection)
- Surprise-gated weight updates
- Momentum-based gradient accumulation
- Forgetting mechanism (prevents unbounded growth)

### 4.3 Episodic Buffer (Experience Storage)

**Location**: `src/hippocampus/episodic_buffer.py`

SQLite-backed priority queue for storing high-surprise interactions.

```python
from src.hippocampus.episodic_buffer import EpisodicBuffer, Episode

buffer = EpisodicBuffer(BufferConfig(
    db_path="data/memory/episodic/replay.db",
    max_episodes=10000,
))

# Add episode
buffer.add(Episode(
    prompt="What is quantum entanglement?",
    response="...",
    surprise=2.8,
    quality_score=0.9,
))

# Sample for training
batch = buffer.sample(batch_size=32, prioritized=True)
```

**Key Features**:
- Persistent SQLite storage
- Priority-based sampling (surprise × quality × recency)
- Cognitive state tagging
- Export to training formats (JSONL)

### 4.4 Nightmare Engine (Offline Consolidation)

**Location**: `src/subconscious/nightmare.py`

Background system that consolidates experiences during idle periods through QLoRA fine-tuning.

```python
from src.subconscious.nightmare import NightmareEngine, NightmareConfig

nightmare = NightmareEngine(
    episodic_buffer=buffer,
    config=NightmareConfig(
        idle_threshold_minutes=30,
        fast_epochs=2,
        slow_epochs=5,
    ),
)

# Start background monitoring
nightmare.start_background_monitoring()

# Or trigger manual sleep
stats = nightmare.dream()
```

**Sleep Phases**:
| Phase | Duration | Activity |
|-------|----------|----------|
| DROWSY | ~1 min | Prepare, sample episodes |
| LIGHT_SLEEP | ~5 min | Fast weight updates (rank-8 LoRA) |
| DEEP_SLEEP | ~15 min | Slow weight consolidation |
| REM | ~30 min | Full QLoRA fine-tuning |
| WAKING | ~1 min | Merge adapters, cleanup |

---

## 5. Implementation Details

### 5.1 Multi-Timescale Learning

AVA implements learning across multiple timescales:

| Timescale | Component | Learning Rate | Purpose |
|-----------|-----------|---------------|---------|
| Milliseconds | Titans Sidecar | 1e-3 | Immediate context adaptation |
| Seconds | Tool execution | N/A | Capability augmentation |
| Minutes | Fast weights | 1e-3 | Session-level adaptation |
| Hours | Slow weights | 1e-5 | Permanent consolidation |

### 5.2 Surprise Signal Computation

The surprise signal `s(x)` drives learning intensity:

```python
# From Entropix cognitive state
surprise = entropy * surprise_scale

# Modulated by cognitive state
if state == CONFUSION:
    surprise *= 1.5  # Learn more from confusing inputs
elif state == CREATIVE:
    surprise *= 1.2  # Novel patterns worth learning
```

### 5.3 Memory Update Rule

Titans Sidecar weight update:

```python
# Forward pass
output = memory_mlp(embedding)
loss = mse_loss(output, target)

# Backward pass with surprise gating
gradient = compute_gradient(loss)
momentum = momentum_factor * momentum + lr * surprise * gradient
weights = weights - momentum

# Forgetting (weight decay)
weights = weights * (1 - forget_alpha)
```

### 5.4 Episode Priority Scoring

Priority for replay sampling:

```python
priority = (
    0.4 * (surprise / 3.0) +      # Surprise contribution
    0.3 * quality_score +          # Quality contribution
    0.2 * recency_factor +         # Time decay
    0.1 * (1 - replay_count/10)    # Less replayed = higher priority
)
```

---

## 6. Installation & Setup

### 6.1 Prerequisites

- **Python**: 3.10 or higher
- **Ollama**: Latest version with `llama3.2` model
- **Hardware**: 8GB+ RAM, GPU optional but recommended

### 6.2 Installation

```bash
# Clone repository
git clone https://github.com/NAME0x0/AVA.git
cd AVA

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 6.3 Ollama Setup

```bash
# Install Ollama (see https://ollama.ai)
# Start Ollama service
ollama serve

# Pull required models
ollama pull llama3.2:latest
ollama pull nomic-embed-text
```

### 6.4 Directory Structure

```bash
# Create data directories
mkdir -p data/memory/episodic
mkdir -p data/learning/checkpoints
mkdir -p models/fine_tuned_adapters/nightmare
```

---

## 7. Usage Guide

### 7.1 Quick Start (Frankensystem)

The Frankensystem is the recommended entry point:

```bash
python run_frankensystem.py
```

**Commands**:
| Command | Description |
|---------|-------------|
| `/cognitive` | Show Entropix cognitive metrics |
| `/memory` | Show Titans neural memory stats |
| `/buffer` | Show episodic buffer statistics |
| `/sleep` | Force sleep/consolidation cycle |
| `/stats` | Full system statistics |
| `/quit` | Exit gracefully |

### 7.2 Developmental Mode

For the developmental (growth-based) system:

```bash
python run_node.py
```

### 7.3 Programmatic Usage

```python
import asyncio
from run_frankensystem import Frankensystem, FrankenConfig

async def main():
    config = FrankenConfig(
        model="llama3.2:latest",
        enable_sleep=True,
    )
    
    system = Frankensystem(config)
    await system.initialize()
    
    # Process input
    result = await system.process("Explain quantum computing")
    print(f"Response: {result['response']}")
    print(f"Cognitive State: {result['cognitive_state']}")
    print(f"Surprise: {result['surprise']:.2f}")
    
    await system.shutdown()

asyncio.run(main())
```

### 7.4 Configuration Override

```bash
# Custom model
python run_frankensystem.py --model llama3:8b

# Disable sleep consolidation
python run_frankensystem.py --no-sleep

# Debug logging
python run_frankensystem.py --debug
```

---

## 8. Configuration Reference

### 8.1 Main Configuration File

**Location**: `config/frankensystem.yaml`

```yaml
# Ollama settings
ollama:
  host: "http://localhost:11434"
  model: "llama3.2:latest"
  embedding_model: "nomic-embed-text"

# Entropix thresholds
entropix:
  high_entropy_threshold: 3.0
  high_varentropy_threshold: 2.0

# Titans Sidecar
titans:
  input_dim: 768
  learning_rate: 0.001
  momentum: 0.9

# Nightmare Engine
nightmare:
  idle_threshold_minutes: 30
  fast_epochs: 2
  slow_epochs: 5
```

### 8.2 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `AVA_DATA_DIR` | `./data` | Data storage directory |
| `AVA_LOG_LEVEL` | `INFO` | Logging verbosity |

---

## 9. Developmental System (Legacy)

AVA also includes a **Developmental AI** mode where the system matures like a human child, progressing through stages from infant to mature.

### Developmental Stages

| Stage | Age (days) | Clarity | Description |
|-------|------------|---------|-------------|
| 👶 INFANT | 0-30 | 20% | Garbled speech, basic emotions, baby-safe tools only |
| 🧒 TODDLER | 30-90 | 40% | Simple sentences, can ask questions, limited tools |
| 👦 CHILD | 90-365 | 70% | Conversations, enthusiasm, moderate tools |
| 🧑 ADOLESCENT | 365-730 | 85% | Opinions, personality, most tools |
| 👨 YOUNG_ADULT | 730-1825 | 95% | Complex topics, nuanced views, advanced tools |
| 🧓 MATURE | 1825+ | 100% | Full capability, all tools, development complete |

### Emotional System

Five core emotions influence behavior:

- **Hope** 🌈 - Increases learning rate, optimistic responses
- **Fear** 😰 - Decreases learning rate, cautious responses  
- **Joy** 😊 - Boosts learning, enthusiastic responses
- **Surprise** 😲 - Temporary learning boost, curious responses
- **Ambition** 🚀 - Increased learning, goal-oriented responses

### Tool Progression

Tools are "gated" by developmental stage:

- **Level 0** (Infant): echo, time, simple_math
- **Level 1** (Toddler): calculator, word_count
- **Level 2** (Child): web_search, file_read
- **Level 3** (Adolescent): file_write, note_save
- **Level 4** (Young Adult): code_execute, api_call
- **Level 5** (Mature): system_command

To run developmental mode:

```bash
python run_node.py
```

---

## 10. Roadmap

### Phase 1: Foundation ✅
- [x] Titans Neural Memory implementation
- [x] Entropix metacognitive module
- [x] Episodic buffer with SQLite
- [x] Nightmare Engine sleep cycles

### Phase 2: Integration (Current)
- [ ] Full logprobs support in Ollama
- [ ] Real-time cognitive state visualization
- [ ] Multi-modal embedding support
- [ ] Distributed memory sharding

### Phase 3: Advanced Features
- [ ] Hierarchical memory consolidation
- [ ] Tool learning via Toolformer
- [ ] Multi-agent collaboration
- [ ] Long-term knowledge graphs

### Phase 4: Production
- [ ] API server mode
- [ ] Docker containerization
- [ ] Kubernetes deployment
- [ ] Monitoring & observability

---

## 11. Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 src/
black src/ --check
```

### Code Style

- **Python**: Black formatter, 88 character lines
- **Docstrings**: Google style
- **Type hints**: Required for public APIs

---

## 12. Citation

If you use AVA in your research, please cite:

```bibtex
@software{ava2025,
  title = {AVA: Adaptive Virtual Agent},
  author = {NAME0x0},
  year = {2025},
  url = {https://github.com/NAME0x0/AVA},
  note = {A neuro-symbolic cognitive architecture for continual learning}
}
```

### Related Work

```bibtex
@article{titans2025,
  title={Titans: Learning to Memorize at Test Time},
  author={Behrouz, Ali and others},
  journal={arXiv preprint arXiv:2501.00663},
  year={2025}
}

@article{qlora2023,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and others},
  journal={arXiv preprint arXiv:2305.14314},
  year={2023}
}
```

---

## 13. License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>AVA</strong> — Bridging the gap between static models and adaptive intelligence.
</p>

<p align="center">
  <a href="https://github.com/NAME0x0/AVA/issues">Report Bug</a> •
  <a href="https://github.com/NAME0x0/AVA/issues">Request Feature</a> •
  <a href="https://github.com/NAME0x0/AVA/discussions">Discussions</a>
</p> 