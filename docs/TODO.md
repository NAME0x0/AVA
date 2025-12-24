# AVA Project: Task List for Local Agentic AI 🛠️

This document outlines the concrete tasks for developing AVA as a high-capability, local agentic AI on an NVIDIA RTX A2000 (4GB VRAM). Refer to `ROADMAP.md` for phased timelines and `ARCHITECTURE.md` for detailed design.

---

## Current Architecture Status: v3 Cortex-Medulla ✅

The repository has transitioned to the **Cortex-Medulla v3 Architecture**:

### Core Entry Points (Consolidated)
- **`run_core.py`** - v3 Always-On core loop (primary entry point)
- **`server.py`** - HTTP/WebSocket API using `src.ava.AVA`
- **`run_tui.py`** - Terminal UI connecting to server.py
- **`src/cli.py`** - v3 CLI using `src.ava.AVA` (refactored)

### v3 Core Components (`src/core/`)
- [x] **Medulla** (`medulla.py`) - Reflexive O(1) state management
- [x] **Cortex** (`cortex.py`) - Reflective reasoning engine
- [x] **Bridge** (`bridge.py`) - State projection for handoffs
- [x] **Agency** (`agency.py`) - Active Inference VFE minimization
- [x] **System** (`system.py`) - Unified orchestration
- [x] **Adapter Manager** (`adapter_manager.py`) - LoRA hot-swapping

### Search-First Policy ✅
- `web_search_effort_cost: 0.05` (lowest cost)
- `search_first_enabled: True`
- `search_gate_enabled: True`
- Epistemic weight (0.6) > Pragmatic weight (0.4)

### Legacy Code (Archived in `legacy/`)
- Developmental stages system
- Emotional engine
- Old memory stores (replaced by Titans)
- v2 core implementations
- Old API servers

---

## Phase 1: Foundation & Core Optimization

### Project Setup & Model Acquisition
*   [x] **Finalize Project Directory Structure:** Ensure clarity for code, docs, datasets, models.
*   [ ] **Select & Download Base LLM:** Gemma 3n 4B (primary) or 1B (backup/testing).
*   [~] **Setup Python Environment:** Install `bitsandbytes`, `transformers`, `peft`, `trl`, `unsloth`, `torch` (with CUDA for RTX A2000).
*   [x] **Install Ollama:** For local model management and serving (alternative to custom Python server).

### Aggressive Quantization & Initial Testing
*   [~] **Implement 4-bit Quantization:** Use `bitsandbytes` (NF4/FP4) for the chosen base model.
*   [ ] **Benchmark VRAM & Inference Speed:** Post-quantization on RTX A2000.
*   [ ] **Qualitative Coherence Testing:** Basic prompts to check model sanity.
*   [x] **Document Quantization:** In `OPTIMIZATION_STRATEGIES.md`.

### QLoRA Setup & Synthetic Data Pipeline
*   [ ] **Implement QLoRA Fine-tuning:** Using `peft`, `trl`, `unsloth` with the quantized model.
*   [ ] **Develop Synthetic Data Generation Workflow:** Script to use a larger LLM (e.g., via API) to create instruction/evaluation data.
*   [ ] **Create Initial Synthetic Dataset:** For a simple agentic task (e.g., parsing, basic tool selection).
*   [ ] **Test QLoRA Fine-tuning Run:** On the initial synthetic dataset.
*   [x] **Document QLoRA & Synthetic Data:** In `OPTIMIZATION_STRATEGIES.md`.

### Knowledge Distillation & Pruning (Research & Planning)
*   [ ] **Research Practical Knowledge Distillation:** Identify teacher models, data requirements, and pipeline.
*   [ ] **Explore Pruning/Sparsification Methods:** (e.g., SparseGPT) and ONNX export.
*   [x] **Document Initial Findings:** In `OPTIMIZATION_STRATEGIES.md`.

## Phase 2: Agentic Core Development

### Function Calling & Basic Tool Use
*   [ ] **Fine-tune AVA for Function Detection:** Using QLoRA and synthetic data.
*   [ ] **Implement Structured Argument Generation:** (JSON output for function calls).
*   [x] **Develop Basic Tool Interface:** Python classes/functions for initial tools.
*   [ ] **Integrate 1-2 Simple Tools:** (e.g., calculator, date/time lookup).
*   [x] **Document Function Calling:** In `AGENTIC_DESIGN.md`.

### Structured Output Implementation
*   [ ] **Implement Output Parsing/Validation:** Ensure reliable structured outputs (JSON, etc.).
*   [ ] **Integrate LlamaIndex Utilities:** Pydantic Programs or Output Parsers, or custom logic.
*   [ ] **Test Structured Output in Tool Workflows.**
*   [x] **Document Structured Output:** In `AGENTIC_DESIGN.md`.

### Reasoning Mechanisms & MCP Foundation
*   [ ] **Implement Chain-of-Thought (CoT) Prompting:** Develop and test CoT prompt templates.
*   [x] **Plan Model Context Protocol (MCP) Integration:**
    *   [x] Design AVA as MCP Host.
    *   [ ] Plan for simple MCP Server interaction (e.g., local file access).
*   [x] **Document CoT & MCP Concepts:** In `AGENTIC_DESIGN.md`.

### Advanced Tool Integration & MCP Prototyping
*   [ ] **Integrate More Complex Tools:** Web search API, database query via MCP.
*   [ ] **Develop Prototype MCP Server:** For a local data source (e.g., text files, simple DB).
*   [ ] **Test AVA with MCP Server:** Data retrieval and use.
*   [ ] **Refine Agentic Workflows:** Combine reasoning, tools, and structured output.

---

## v3.4 Priority Tasks (Weights-In Phase)

### Model Integration (Simulated → Real)
*   [ ] **Integrate Mamba SSM weights** into Medulla for O(1) state management
*   [ ] **Load BitNet weights** or alternative quantized model for Cortex
*   [ ] **Test Bridge state projection** with actual model tensors
*   [ ] **Validate VRAM budget** stays under 3,000 MB target

### Adapter Library Setup
*   [x] **Create adapter directory structure** (`models/adapters/`)
*   [ ] **Train/Obtain Coding Expert LoRA** for software development tasks
*   [ ] **Train/Obtain Logic Expert LoRA** for reasoning tasks
*   [ ] **Train/Obtain Butler Expert LoRA** for conversational assistance
*   [ ] **Implement hot-swap mechanism** in `adapter_manager.py`

### Thermal Management
*   [ ] **Implement Thermal Sentry** in Medulla for GPU monitoring
*   [ ] **Add power throttling** when approaching 70W TDP
*   [ ] **Test thermal behavior** under sustained load

### Memory System
*   [ ] **Integrate Titans Neural Memory** for infinite context
*   [ ] **Remove legacy episodic/semantic buffers** from active path
*   [ ] **Test memory fragmentation** under sustained operation

---

## Phase 3: User Interface, Connectivity & Refinement

### CLI Development
*   [x] **Develop Robust CLI:** Using `argparse`, `Typer`, or `Click`.
*   [x] **Refactor CLI to v3 AVA class** - Uses `src.ava.AVA` now
*   [ ] **Implement CLI Commands:** For prompts, function calls, settings.
*   [ ] **Ensure CLI Handles Structured/Streamed Responses.**
*   [x] **Document CLI Usage:** In `UI_CONNECTIVITY.md` and `USAGE.md`.

### GUI Implementation (Open WebUI)
*   [ ] **Setup Open WebUI:** With Docker, configure for GPU acceleration.
*   [ ] **Integrate AVA with Open WebUI:** Via Ollama or custom API endpoint.
*   [ ] **Configure Open WebUI for AVA:** System prompts, parameters, knowledge collections (if used).
*   [ ] **(Stretch) Customize Open WebUI:** For agentic features (tool call display, context view).
*   [x] **Document GUI Setup:** In `UI_CONNECTIVITY.md` and `USAGE.md`.

### Remote Access & Token Broadcasting
*   [ ] **Implement Secure Tunneling:** Using `Localtonet` or `ngrok` for AVA's local server.
*   [ ] **Configure Security:** Authentication, IP whitelisting.
*   [ ] **Implement Token Streaming:** Server-Sent Events (SSE) for AVA's output.
*   [ ] **Test Remote Access & Streaming.**
*   [x] **Document Remote Access:** In `UI_CONNECTIVITY.md`.

### Testing, Feedback & Documentation
*   [ ] **Comprehensive End-to-End Testing:** Local and remote scenarios.
*   [ ] **Develop User Feedback Mechanism:** (e.g., simple rating in GUI).
*   [ ] **Iterative Refinement:** Based on testing and feedback.
*   [ ] **Finalize All Documentation:** `INSTALLATION.md`, `USAGE.md`, etc.

## Continuous Improvement (Ongoing)

*   [ ] **Establish Feedback Collection & Analysis Pipeline.**
*   [ ] **Regularly Fine-tune/Update AVA Model:** With new data, techniques.
*   [ ] **Expand Toolset & Agentic Capabilities.**
*   [ ] **Monitor Performance Metrics.**
*   [ ] **Engage with Community for Contributions.**

This task list is a living document and will be updated as development progresses.
