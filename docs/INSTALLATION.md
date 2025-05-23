# AVA Installation & Setup Guide 🛠️

This guide provides step-by-step instructions for installing and setting up AVA, your local agentic AI daily driver, on a system with an **NVIDIA RTX A2000 GPU (4GB VRAM)**.

**Goal:** To configure your environment to run AVA efficiently, leveraging tools like Ollama for local LLM management, `bitsandbytes` for quantization, and necessary Python libraries for fine-tuning and agentic capabilities.

**Primary Documentation:**
*   `[ARCHITECTURE.md](./ARCHITECTURE.md)`
*   `[OPTIMIZATION_STRATEGIES.md](./OPTIMIZATION_STRATEGIES.md)`

## I. Prerequisites

1.  **Hardware:**
    *   **GPU:** NVIDIA RTX A2000 with 4GB GDDR6 VRAM (ensure this specific model/VRAM amount).
    *   **CPU:** Modern multi-core CPU (e.g., Intel Core i5/i7 8th gen+, AMD Ryzen 5/7 2000 series+).
    *   **RAM:** Minimum 16GB system RAM (32GB recommended for smoother experience, especially if running other applications or for more intensive fine-tuning experiments).
    *   **Storage:** Sufficient SSD storage for OS, development tools, base models, datasets, and AVA codebase (at least 100GB free recommended).
2.  **Software:**
    *   **Operating System:** Linux (recommended for best CUDA support and development ease, e.g., Ubuntu 20.04/22.04 LTS) or Windows 10/11 with WSL2 (Windows Subsystem for Linux 2) for Linux environment.
    *   **NVIDIA Drivers:** Latest stable NVIDIA drivers for your RTX A2000 that support CUDA 11.x or 12.x (check compatibility with PyTorch/bitsandbytes versions).
    *   **CUDA Toolkit:** (Often included with drivers or can be installed separately). Version compatible with PyTorch and `bitsandbytes`.
    *   **Python:** Version 3.9 - 3.11 (Python 3.10 recommended for broad compatibility).
    *   **Git:** For cloning the AVA repository.
    *   **Miniconda/Anaconda (Recommended):** For managing Python environments.
    *   **Docker (Optional but Recommended):** For running Open WebUI and potentially other services in isolated containers.

## II. Environment Setup

1.  **Install NVIDIA Drivers & CUDA:**
    *   Ensure your NVIDIA drivers are up to date. Visit the [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx) page.
    *   Verify CUDA installation by running `nvidia-smi` in your terminal. This command should show your GPU details and CUDA version.

2.  **Setup Python Environment (using Conda - Recommended):**
    ```bash
    # Create a new conda environment for AVA
    conda create -n ava python=3.10 -y

    # Activate the environment
    conda activate ava
    ```

3.  **Install PyTorch with CUDA Support:**
    *   Visit the [PyTorch official website](https://pytorch.org/get-started/locally/) and select the appropriate options for your OS, package manager (pip/conda), CUDA version, etc.
    *   Example (check website for current command for your CUDA version):
        ```bash
        # For example, with pip and CUDA 11.8 (VERIFY ON PYTORCH WEBSITE!)
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```
    *   Verify PyTorch installation and CUDA support in Python:
        ```python
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        ```

4.  **Clone AVA Repository:**
    ```bash
    git clone https://github.com/NAME0x0/AVA.git
    cd AVA
    ```

5.  **Install Core Dependencies:**
    *   Install `bitsandbytes` (ensure compatible version for your OS/CUDA):
        *   For Linux: `pip install bitsandbytes`
        *   For Windows (if not using WSL2, might be more complex, WSL2 recommended): `pip install bitsandbytes-windows` (check official `bitsandbytes` repo for latest Windows instructions).
    *   Install other Python packages from `requirements.txt` (once created for the project):
        ```bash
        pip install -r requirements.txt 
        # Expected packages in requirements.txt will include:
        # transformers, peft, trl, unsloth, datasets, accelerate, sentencepiece, protobuf
        # ollama (Python client), openai (for API access if using teacher models)
        # Flask/FastAPI (if building custom server), typer/click (for CLI)
        # And other necessary utilities
        ```

## III. Install Ollama (for Local LLM Management)

Ollama simplifies running LLMs like Gemma locally. It will be used to serve the quantized and fine-tuned AVA model for interaction with UIs like Open WebUI.

1.  **Download and Install Ollama:**
    *   Visit [https://ollama.com](https://ollama.com) and follow the installation instructions for your OS.
2.  **Verify Ollama Installation:**
    ```bash
    ollama --version
    ```
3.  **Pull a Base Gemma Model (Optional Initial Test):**
    *   You can test Ollama by pulling a base Gemma model (e.g., `gemma:2b` or `gemma:7b` if you have more VRAM for testing, though AVA will use a smaller, custom one).
        ```bash
        ollama pull gemma:2b # AVA will use a ~4B model, this is just for Ollama test
        ollama run gemma:2b "Why is the sky blue?"
        ```

## IV. Acquiring and Setting Up AVA Model

This section will involve steps defined by the project's progress in Phase 1 (Optimization) and Phase 2 (Agentic Core).

1.  **Download Base Model for AVA:**
    *   Follow instructions (once provided in `OPTIMIZATION_STRATEGIES.md` or scripts) to download the chosen Gemma 3n base model (e.g., 4B variant).
    *   Store it in the `models/base_models/` directory (see recommended structure in `docs/README.md`).

2.  **Quantize the Base Model:**
    *   Run the quantization script (e.g., `scripts/quantize_model.py` - to be developed) as per `OPTIMIZATION_STRATEGIES.md`.
    *   This will produce a 4-bit quantized version of the model, significantly reducing its VRAM footprint.

3.  **Create AVA Model in Ollama:**
    *   Once AVA (quantized base or QLoRA fine-tuned version) is ready, you'll create a `Modelfile` for it.
    *   Example `Modelfile` (conceptual - details will depend on final model format and AVA's system prompt):
        ```Modelfile
        FROM ./models/quantized_ava/ava-gemma-4b-q4.gguf # Path to your quantized model file

        TEMPLATE """{{ if .System }}
        {{ .System }} {{ end }}
        USER: {{ .Prompt }}
        ASSISTANT: """

        SYSTEM """You are AVA, a helpful and highly capable local AI assistant optimized for agentic tasks. Be concise and accurate. When asked to perform a task that requires a tool, clearly indicate the tool and parameters you would use in a structured format if necessary."""

        PARAMETER temperature 0.7
        # Add other necessary parameters
        ```
    *   Create the model in Ollama:
        ```bash
        ollama create ava-agent -f ./Modelfile.ava 
        ```

4.  **Run AVA via Ollama:**
    ```bash
    ollama run ava-agent "Your prompt for AVA"
    ```

## V. Setting up User Interfaces

Refer to `[UI_CONNECTIVITY.md](./UI_CONNECTIVITY.md)` for detailed instructions.

1.  **CLI:** Should be usable directly after installing dependencies and setting up the model in Ollama (or via direct Python script if AVA has its own server logic).
2.  **GUI (Open WebUI):**
    *   Install Docker if you haven't already.
    *   Follow Open WebUI instructions to run it via Docker, ensuring it can connect to your local Ollama instance (typically `http://host.docker.internal:11434` from within the Open WebUI container if Ollama runs on the host).
    *   Select the `ava-agent` model in Open WebUI.

## VI. Next Steps

*   **Fine-tuning AVA:** Follow guidelines in `OPTIMIZATION_STRATEGIES.md` and `AGENTIC_DESIGN.md` to fine-tune AVA for specific agentic tasks using QLoRA.
*   **Tool Integration:** Develop and integrate tools as per `AGENTIC_DESIGN.md`.
*   **MCP Setup:** Implement Model Context Protocol components if direct local file/data access is a core requirement beyond basic RAG via Open WebUI.

This installation guide is a living document and will be updated as AVA's development progresses and specific scripts/tools are finalized.
