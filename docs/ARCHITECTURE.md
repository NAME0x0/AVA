# AVA System Architecture: Local Agentic AI Daily Driver

**Project Vision:** To establish AVA as a compact, powerful, and locally-run agentic AI model capable of serving as a daily driver for advanced tasks on an NVIDIA RTX A2000 GPU with 4GB VRAM. AVA is designed for complex agentic tasks, seamless user experience (CLI and GUI), and remote access via token broadcasting.

This document details the architecture of AVA, focusing on achieving high performance and advanced capabilities within significant hardware constraints.

## I. Core Challenge: 4GB VRAM Limitation

The NVIDIA RTX A2000 (Ampere Architecture) typically features 6GB or 12GB GDDR6 memory. The project's specific 4GB VRAM constraint necessitates an exceptionally aggressive approach to model selection, optimization, and fine-tuning. This limitation drives all architectural decisions.

## II. Foundational Layers

### A. Base Model Selection

*   **Constraint:** Strict 4GB VRAM limit.
*   **Strategy:** Prioritize ultra-small, highly efficient LLM architectures.
*   **Viable Candidates (Post-Quantization):**
    *   **Gemma 3n 4B:** Requires ~2.3-2.6GB VRAM (4-bit quantized). Chosen for its balance of capability and small footprint.
    *   **Gemma 3n 1B:** Requires ~0.5GB VRAM (4-bit quantized). An alternative for even tighter constraints or simpler tasks.
*   **Rationale:** Larger models (e.g., DeepSeek 7B+) are infeasible. Gemma models are engineered for efficiency on consumer hardware, featuring Per-Layer Embeddings (PLE) and MatFormer architecture for dynamic memory footprint reduction.

### B. Core Optimization Strategies

1.  **Aggressive Model Compression:**
    *   **4-bit Quantization (INT4):** Essential for fitting the model into VRAM. Reduces numerical precision of weights/activations. Implemented using libraries like `bitsandbytes` (provides `LLM.int8()` and QLoRA for 4-bit) which is well-suited for Ampere GPUs.
    *   **Knowledge Distillation:** A smaller "student" model (AVA) learns from a larger, more capable "teacher" model. AVA learns from the teacher's soft probabilities and internal representations, enabling it to approximate superior performance with fewer parameters. This is key to AVA being "more advanced" in specific capabilities.
    *   **Pruning & Sparsification:** Systematically removing redundant parameters (weights, connections) using techniques like structured pruning or SparseGPT. Models can be exported to optimized formats like ONNX.

2.  **Parameter-Efficient Fine-Tuning (PEFT):**
    *   **QLoRA (Quantized Low-Rank Adaptation):** Freezes most of the pre-trained LLM's 4-bit quantized weights and introduces a small set of trainable low-rank adaptation (LoRA) weights. This drastically reduces computational/storage costs of fine-tuning and prevents catastrophic forgetting. Implemented using libraries like `trl`, `unsloth`, `peft`, and `bitsandbytes`.
    *   **High-Quality Synthetic Dataset Generation:** Leverage LLMs (e.g., using tools like Gretel Navigator or custom workflows) to create task-specific instruction data (question-answer pairs, step-by-step evaluations). This overcomes data scarcity for specialized agentic tasks.

## III. Agentic Capabilities Architecture

AVA's agentic workflow relies on sophisticated Natural Language Understanding (NLU), Dialogue Management (DM), and Natural Language Generation (NLG), with memory-enhanced architectures for long-term context.

### A. Core Agentic Components

1.  **Function Calling & Tool Use:**
    *   AVA will be fine-tuned to recognize when external tools/APIs are needed and output structured JSON with arguments for execution.
    *   Enables dynamic access to real-time data (weather, databases) and actions (scheduling, code generation).

2.  **Structured Output:**
    *   Ensures AVA produces reliable, parsable outputs (JSON, formatted SQL) for downstream processing and interoperability.
    *   Utilizes libraries like LlamaIndex (Pydantic Programs, Output Parsers) leveraging native function calling or post-processing text completions.

3.  **Reasoning Mechanisms:**
    *   **Chain-of-Thought (CoT) Prompting:** Instructing the LLM to generate explicit step-by-step reasoning.
    *   **Advanced Reasoning (e.g., RL-of-Thoughts - RLoT):** Potential for training a lightweight "navigator" model via reinforcement learning to adaptively select and combine reasoning blocks for complex tasks.

### B. Model Context Protocol (MCP) Integration

*   **Concept:** An open standard allowing AVA (as an MCP Host) to directly access files, APIs, and tools via lightweight MCP Servers, bypassing traditional RAG embeddings/vector searches.
*   **Benefits:**
    *   **Direct, Real-time Data Access:** Reduces computational load, improves data freshness, and minimizes hallucinations.
    *   **Streamlined Tool Orchestration:** Simplifies adding new data sources/tools by standardizing the interface (N+M vs N x M integration complexity).
    *   **Enhanced Security:** Minimizes data movement and storage.

## IV. User Experience (UX) and Connectivity

### A. User Interfaces

1.  **Command Line Interface (CLI):**
    *   For direct control, scripting, and advanced users.
    *   Allows prompt input, function execution, tool interaction, and structured output in the terminal.

2.  **Graphical User Interface (GUI):**
    *   **Recommended Foundation:** Open WebUI – a self-hosted, offline-capable AI platform supporting Ollama and OpenAI-compatible APIs.
    *   **Key Open WebUI Features:** ChatGPT-style interface, LLM selector, parameter playground, knowledge collections (document upload), live web search, GPU acceleration via Docker.
    *   **Advanced Agentic UI Inspirations (from OpenAI Codex, Claude Code Terminal):**
        *   Interactive code snippets.
        *   Project context awareness.
        *   Actionable outputs (e.g., diffs, follow-up tasks).
        *   User configurability (agent behavior, permissions).
        *   Feedback mechanisms.

### B. Remote Access & Token Broadcasting

1.  **Secure Tunneling for Local Inference:**
    *   Expose AVA's local server (e.g., running on a specific port) to the internet via a secure public URL using tools like `Localtonet` or `ngrok`.
    *   Crucial for privacy and cost-saving (data remains local).
    *   Requires security features: basic authentication, IP whitelisting.

2.  **Token Streaming for Responsive UI:**
    *   Incrementally send generated tokens to the client as they become available, eliminating long load times.
    *   **Recommended Method:** Server-Sent Events (SSE) for efficient, uni-directional streaming of text data over HTTP.
    *   Alternative: WebSockets for bi-directional streaming if more interactive communication is needed.

## V. System Overview Diagram

```mermaid
graph TD
    A[User (CLI / GUI / Remote)] --> B{AVA Core};

    subgraph AVA Core
        C[Interface Manager (CLI/GUI Adapters)] --> D{Dialogue Manager};
        D -- User Input --> E[NLU Engine];
        E --> F[Optimized LLM (Gemma 3n 4B + QLoRA)];
        F -- Internal Thought/Reasoning --> F;
        F -- Structured Output/Function Call --> G{Agentic Engine};
        G -- Needs Tool/Data --> H{Tool & MCP Interface};
        H --> I[Local Files (via MCP)];
        H --> J[External APIs (Weather, Calendar etc.)];
        H --> K[Databases (via MCP)];
        I --> H;
        J --> H;
        K --> H;
        H -- Tool/Data Response --> G;
        G -- Action/Response Plan --> L[NLG Engine];
        L -- Formatted Response --> C;
    end

    B -- Token Stream (SSE) --> A;
    M[Synthetic Data Generation Workflow] -.-> F;
    N[Knowledge Distillation Teacher Model] -.-> F;

    subgraph External Services
        O[MCP Servers]
        P[External APIs]
        Q[Databases]
    end

    H <--> O;
    H <--> P;
    H <--> Q;

    R[Secure Tunnel (e.g., Localtonet)] <--> B;
```

## VI. Implementation Roadmap Summary

1.  **Phase 1: Foundation & Optimization:** Base model selection (Gemma 3n 4B), 4-bit quantization, QLoRA setup, synthetic dataset pipeline.
2.  **Phase 2: Agentic Core Development:** Function calling, structured output, reasoning mechanisms (CoT), MCP integration.
3.  **Phase 3: Interface & Connectivity:** CLI development, GUI (Open WebUI base), secure tunneling, token streaming.

**Continuous Improvement:** Establish feedback loops (user ratings, interaction analysis) for ongoing model refinement, prompt adjustments, and tool integration updates. 