# AVA Development Guidelines & Philosophy (Local Agentic AI)

This document outlines the development guidelines and philosophy for Project AVA, focusing on creating a high-capability, local agentic AI on an NVIDIA RTX A2000 (4GB VRAM). The core challenge is balancing advanced functionality with extreme resource constraints.

Refer to `[ARCHITECTURE.md](./ARCHITECTURE.md)` for the system design and `[ROADMAP.md](./ROADMAP.md)` for phased execution.

## I. Guiding Principles

1.  **VRAM-First Optimization:** Every design choice, library selection, and implementation detail must prioritize minimal VRAM usage. The 4GB limit is paramount.
2.  **Aggressive & Smart Compression:** Employ quantization (4-bit mandatory), knowledge distillation, and pruning strategically to maximize capability within the footprint.
3.  **Parameter-Efficient Fine-Tuning (PEFT):** Utilize QLoRA as the primary method for specializing AVA without incurring prohibitive VRAM costs during training/fine-tuning.
4.  **Modular Agentic Design:** Build AVA's agentic capabilities (function calling, tool use, reasoning, MCP) as distinct but interconnected modules for clarity, testability, and iterative improvement.
5.  **Data-Centric Specialization:** Leverage high-quality synthetic data generation to train AVA for specific agentic tasks where real-world data is scarce.
6.  **Iterative Refinement:** Develop AVA in phases, with continuous benchmarking of VRAM usage, inference speed, and task performance. Refine based on empirical results.
7.  **Focus on "Agentic Sharpness" over "General Broadness":** Aim for AVA to be exceptionally good at specific, complex agentic workflows rather than trying to match the general knowledge of much larger models.
8.  **User-Centric Utility:** Design for practical daily use, with intuitive interfaces (CLI, GUI) and reliable remote access.
9.  **Open & Collaborative (where possible):** Utilize open-source tools and libraries extensively. Encourage community contributions for improvements and new capabilities.
10. **Reproducibility & Clear Documentation:** Ensure that development steps, experiments, and configurations are well-documented to allow others (and your future self) to reproduce results.

## II. Key Development Areas & Focus

*(Corresponds to specialized documents: `OPTIMIZATION_STRATEGIES.md`, `AGENTIC_DESIGN.md`, `UI_CONNECTIVITY.md`)*

1.  **Core Model Optimization:**
    *   **Team/Focus:** LLM engineers, optimization specialists.
    *   **Tasks:** Base model selection (Gemma 3n), 4-bit quantization (`bitsandbytes`), QLoRA implementation (`peft`, `trl`, `unsloth`), knowledge distillation pipeline, pruning research.
    *   **Key Metrics:** VRAM footprint, inference latency, perplexity (or other relevant metrics post-quantization/distillation), fine-tuning efficiency.

2.  **Agentic Engine & Workflow Development:**
    *   **Team/Focus:** AI engineers, backend developers.
    *   **Tasks:** Designing and implementing function calling mechanisms, structured output parsing/validation, Chain-of-Thought and advanced reasoning integration, Model Context Protocol (MCP) host/server implementation.
    *   **Key Metrics:** Task success rate for agentic workflows, reliability of tool use, accuracy of structured output, reasoning quality.

3.  **User Interface (UI) & Connectivity:**
    *   **Team/Focus:** Frontend/full-stack developers, UX designers.
    *   **Tasks:** CLI development, Open WebUI setup and customization, secure tunneling implementation for remote access, token streaming for responsive UI.
    *   **Key Metrics:** User satisfaction, ease of use, responsiveness of interfaces (local and remote), stability of connections.

4.  **Data Engineering for Fine-Tuning:**
    *   **Team/Focus:** Data engineers, AI engineers.
    *   **Tasks:** Designing and implementing synthetic data generation pipelines, data cleaning and formatting, managing datasets for QLoRA and knowledge distillation.
    *   **Key Metrics:** Quality and diversity of synthetic data, impact of data on fine-tuning outcomes.

5.  **Testing & Quality Assurance:**
    *   **Team/Focus:** QA engineers, all developers.
    *   **Tasks:** Developing unit tests for individual modules, integration tests for agentic workflows, end-to-end testing for UI and connectivity, performance benchmarking on the target hardware.
    *   **Key Metrics:** Code coverage, bug detection rates, performance stability, VRAM consistency.

## III. Technology Stack (Core Recommendations)

*   **Primary Language:** Python.
*   **LLM & ML Frameworks:** PyTorch, Hugging Face (`transformers`, `peft`, `datasets`), `bitsandbytes`, `trl`, `unsloth`.
*   **Local LLM Management/Serving:** Ollama (recommended for ease of use) or custom Python server (e.g., using FastAPI/Flask).
*   **CLI Development:** `Typer` or `Click` (preferred over `argparse` for richer CLIs).
*   **GUI Foundation:** Open WebUI (with Docker for GPU acceleration).
*   **Remote Access Tunneling:** `Localtonet`, `ngrok`.
*   **Data Handling:** Pandas, NumPy.
*   **Version Control:** Git.
*   **Environment Management:** Conda / venv / Docker.

## IV. Development Workflow & Best Practices

1.  **Version Control (Git):** Use feature branches, regular commits with clear messages, and pull requests for code review (even for solo development, it's good practice).
2.  **Environment Management:** Strictly manage Python environments to avoid dependency conflicts.
3.  **Configuration Management:** Store configurations (model paths, API keys, parameters) in external files (e.g., YAML, `.env`) and use `.env.example` for templates. Do not commit secrets.
4.  **Modular Code:** Write small, well-defined functions and classes with clear responsibilities.
5.  **Testing:** Write tests as you develop. Unit tests for logic, integration tests for workflows.
6.  **Benchmarking:** Regularly benchmark VRAM usage and inference speed, especially after changes to the model or optimization techniques.
7.  **Documentation:** Document code (docstrings), architectural decisions (`docs/` folder), and setup procedures (`INSTALLATION.md`, `README.md`). The detailed blueprint itself is a form of living documentation.
8.  **Issue Tracking:** Use GitHub Issues (or similar) to track tasks, bugs, and feature requests.
9.  **Regular Backups:** Especially for fine-tuned model adapters and generated datasets.

## V. Ethical Considerations & Responsible AI

While AVA is a local model, responsible AI practices are still important:
*   **Bias in Base Models & Data:** Be aware of potential biases in the chosen base LLM and in any data used for fine-tuning (including synthetic data). Strive to mitigate where possible.
*   **Structured Output Safety:** If AVA generates code or commands, ensure appropriate safeguards or user confirmations are in place before execution.
*   **Data Privacy with MCP:** If MCP is used to access sensitive local files, ensure the protocol and server implementations are secure.
*   **Transparency of Capabilities:** Be clear about what AVA can and cannot do, especially regarding the limitations imposed by its size and hardware.

By adhering to these guidelines, Project AVA aims to deliver a uniquely capable and efficient local AI daily driver, pushing the envelope of what's achievable on constrained hardware.
