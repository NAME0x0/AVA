# AVA Development Roadmap 🗺️

This document outlines the phased development plan for Project AVA, focusing on creating a compact, powerful, and locally-run agentic AI daily driver for an NVIDIA RTX A2000 (4GB VRAM).

**Core Documentation:**
*   **Overall Architecture:** `[ARCHITECTURE.md](./ARCHITECTURE.md)`
*   **Optimization Details:** `[OPTIMIZATION_STRATEGIES.md](./OPTIMIZATION_STRATEGIES.md)`
*   **Agentic Capabilities:** `[AGENTIC_DESIGN.md](./AGENTIC_DESIGN.md)`
*   **UI and Connectivity:** `[UI_CONNECTIVITY.md](./UI_CONNECTIVITY.md)`

## Phase 1: Foundation & Core Optimization (Target: 8-12 Weeks)

**Goal:** Establish the core technical foundation, select and aggressively optimize the base LLM to fit the hardware constraints, and set up for efficient fine-tuning.

*   **Week 1-2: Project Setup & Model Acquisition**
    *   [ ] Finalize project structure in Git.
    *   [ ] Research and select the primary base model (e.g., Gemma 3n 4B or 1B).
    *   [ ] Set up development environment with necessary libraries (`bitsandbytes`, `transformers`, `peft`, `trl`, `unsloth`, `ollama`).
    *   [ ] Download and verify the chosen base model.
*   **Week 3-5: Aggressive Quantization & Initial Testing**
    *   [ ] Implement 4-bit quantization of the base model using `bitsandbytes`.
    *   [ ] Benchmark VRAM usage and basic inference speed on the RTX A2000.
    *   [ ] Perform initial qualitative tests for model coherence post-quantization.
    *   [ ] Document quantization process and results in `OPTIMIZATION_STRATEGIES.md`.
*   **Week 6-8: QLoRA Setup & Synthetic Data Pipeline**
    *   [ ] Implement QLoRA fine-tuning setup using `peft`, `trl`, `unsloth`.
    *   [ ] Develop an initial pipeline for generating synthetic instruction datasets (e.g., using a larger LLM via API or a simple script).
    *   [ ] Create a small, high-quality synthetic dataset for a basic agentic task (e.g., simple tool use or structured output).
    *   [ ] Conduct a test fine-tuning run with QLoRA on the synthetic data.
    *   [ ] Document PEFT setup and synthetic data strategy in `OPTIMIZATION_STRATEGIES.md`.
*   **Week 9-12: Knowledge Distillation & Pruning Research (Parallel)**
    *   [ ] Research practical knowledge distillation techniques suitable for AVA's scale.
    *   [ ] Identify potential "teacher" models and data sources.
    *   [ ] Explore pruning and sparsification methods (e.g., SparseGPT) and their applicability.
    *   [ ] Document initial findings for distillation and pruning in `OPTIMIZATION_STRATEGIES.md`.

**Outcome of Phase 1:** A highly quantized base LLM capable of running on the RTX A2000 4GB, with a functional QLoRA pipeline for further specialization, and initial strategies for knowledge distillation and synthetic data generation documented.

## Phase 2: Agentic Core Development (Target: 10-14 Weeks)

**Goal:** Build AVA's advanced agentic capabilities, including function calling, structured output, reasoning, and MCP integration.

*   **Week 1-3: Function Calling & Basic Tool Use**
    *   [ ] Fine-tune AVA (QLoRA) to recognize when an external function/tool is needed.
    *   [ ] Implement mechanisms for AVA to generate structured JSON arguments for functions.
    *   [ ] Integrate 1-2 simple tools (e.g., a calculator tool, a date/time tool).
    *   [ ] Document function calling implementation in `AGENTIC_DESIGN.md`.
*   **Week 4-6: Structured Output Implementation**
    *   [ ] Implement methods to ensure AVA consistently produces structured outputs (JSON, specific text formats).
    *   [ ] Explore and integrate libraries like LlamaIndex (Pydantic Programs, Output Parsers) or custom parsing logic.
    *   [ ] Test structured output reliability for tool interactions.
    *   [ ] Document structured output mechanisms in `AGENTIC_DESIGN.md`.
*   **Week 7-9: Reasoning Mechanisms & MCP Foundation**
    *   [ ] Implement Chain-of-Thought (CoT) prompting strategies for AVA.
    *   [ ] Experiment with CoT for multi-step tasks.
    *   [ ] Begin theoretical work and prototyping for Model Context Protocol (MCP) integration: design AVA as an MCP Host and plan for simple MCP Server interaction (e.g., local file access).
    *   [ ] Document CoT and initial MCP concepts in `AGENTIC_DESIGN.md`.
*   **Week 10-14: Advanced Tool Integration & MCP Prototyping**
    *   [ ] Integrate more complex tools (e.g., web search API, local database query via MCP).
    *   [ ] Develop a prototype MCP server for a specific local data source.
    *   [ ] Test AVA's ability to access and use data via the MCP prototype.
    *   [ ] Refine agentic workflows combining reasoning, tool use, and structured output.

**Outcome of Phase 2:** AVA can perform basic agentic tasks involving multiple tools, generate structured outputs, apply simple reasoning, and has a foundational MCP integration for direct data access. Agentic capabilities are well-documented.

## Phase 3: User Interface, Connectivity & Refinement (Target: 8-12 Weeks)

**Goal:** Develop user-friendly interfaces (CLI & GUI), enable remote access, and iteratively refine AVA based on testing and feedback.

*   **Week 1-3: CLI Development & Core Interaction Loop**
    *   [ ] Develop a robust CLI for interacting with AVA's agentic core.
    *   [ ] Implement commands for prompt input, function execution, and managing AVA's settings.
    *   [ ] Ensure the CLI can handle structured and streamed responses.
    *   [ ] Document CLI usage in `UI_CONNECTIVITY.md` and `USAGE.md`.
*   **Week 4-7: GUI Implementation (Open WebUI)**
    *   [ ] Set up Open WebUI and integrate it with AVA's local inference server (e.g., Ollama or custom server).
    *   [ ] Configure Open WebUI for AVA (system prompts, parameters).
    *   [ ] Begin customizing Open WebUI to incorporate agentic UI features (e.g., display of tool calls, context awareness).
    *   [ ] Document GUI setup and features in `UI_CONNECTIVITY.md` and `USAGE.md`.
*   **Week 8-10: Remote Access & Token Broadcasting**
    *   [ ] Implement secure tunneling (e.g., `Localtonet`) to expose AVA's local server.
    *   [ ] Configure security measures (authentication, IP whitelisting).
    *   [ ] Implement token streaming (e.g., using Server-Sent Events) for responsive remote interaction.
    *   [ ] Test remote access from different devices.
    *   [ ] Document remote access setup in `UI_CONNECTIVITY.md`.
*   **Week 11-12: Testing, Feedback Integration & Documentation Finalization**
    *   [ ] Conduct comprehensive testing of all features (local and remote).
    *   [ ] Gather user feedback and identify areas for improvement.
    *   [ ] Make iterative refinements to the model, prompts, and UI.
    *   [ ] Complete all documentation (`INSTALLATION.md`, `USAGE.md`, etc.).

**Outcome of Phase 3:** A functional AVA with CLI and GUI, accessible remotely, capable of performing agentic tasks on the target hardware. Comprehensive documentation is complete.

## Continuous Improvement (Ongoing Post-Phase 3)

*   [ ] **Establish Feedback Loops:** Collect explicit (user ratings) and implicit (interaction analysis) feedback.
*   [ ] **Iterative Model Refinement:** Regularly fine-tune AVA with new synthetic data, distilled knowledge, or user feedback.
*   [ ] **Tool & Capability Expansion:** Incrementally add new tools and enhance agentic reasoning.
*   [ ] **Performance Monitoring:** Continuously monitor VRAM usage, inference speed, and task success rates.
*   [ ] **Community Engagement:** Foster a community for contributions and shared improvements.

This roadmap is ambitious and subject to adjustments based on development progress and emerging challenges. Flexibility and iterative problem-solving will be key.
