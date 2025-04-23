# AVA Installation Guide

This guide provides detailed steps to set up the AVA project environment.

## Prerequisites

-   **Hardware:**
    -   Reasonable RAM (32GB+ recommended for larger models)
    -   NVIDIA GPU with CUDA support (Recommended for training/faster inference)
    -   Sufficient free storage (100GB+ recommended)
-   **Operating System:** Windows, macOS, or Linux.
-   **Software:**
    -   Git
    -   Python 3.8+ and Pip
    -   (Optional but Recommended) CUDA Toolkit compatible with PyTorch (e.g., 11.8)

## 1. Clone the Repository

Open your terminal or command prompt and clone the AVA repository:

```bash
git clone https://github.com/NAME0x0/AVA.git
cd AVA
```

## 2. Set Up Python Environment

It's recommended to use a virtual environment:

```bash
# Create a virtual environment (e.g., named .venv)
python -m venv .venv

# Activate the virtual environment
# On Windows (Command Prompt/PowerShell)
.\.venv\Scripts\activate
# On macOS/Linux (Bash/Zsh)
source .venv/bin/activate
```

## 3. Install Python Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

**Note:** If you encounter issues with `PyAudio` on some systems, you might need to install system-level dependencies first (like `portaudio19-dev` on Debian/Ubuntu: `sudo apt-get install portaudio19-dev`). The CI workflow attempts this.

## 4. Install PyTorch with CUDA (Optional, Recommended)

For GPU acceleration, install PyTorch matching your CUDA version. Check the [PyTorch website](https://pytorch.org/get-started/locally/) for the correct command. For CUDA 11.8, the command is often:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

If you don't have a compatible NVIDIA GPU or don't need GPU acceleration, you can install the CPU-only version (usually the default if the CUDA version isn't specified or found).

## 5. Install and Configure Ollama

AVA relies on Ollama to run local language models efficiently.

1.  **Download and Install Ollama:** Go to [ollama.com/download](https://ollama.com/download) and follow the instructions for your operating system.
2.  **Verify Installation:** Open a new terminal and run:

    ```bash
    ollama --version
    ```

3.  **Pull a Model:** Download the model AVA is configured to use (e.g., DeepSeek-R1 8B as mentioned in `DEVELOPMENT.md`).

    ```bash
    ollama pull deepseek-r1:8b
    ```
    
    *(You might need other models depending on the specific configuration or expert modules used).*
4.  **Ensure Ollama is Running:** Ollama usually runs as a background service after installation. You can test it by running `ollama list` in the terminal.

## 6. Install Unsloth (Optional, for Development/Training)

If you plan to fine-tune models using the scripts provided in `DEVELOPMENT.md`, install Unsloth:

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# Or simply:
# pip install unsloth
```

Refer to the [Unsloth GitHub repository](https://github.com/unslothai/unsloth) for the latest installation instructions, especially regarding specific hardware (like different CUDA versions or ROCm).

## 7. Configuration

-   Review configuration files like `src/core/config.py` or `data/settings.json` (if they exist and are used).
-   You may need to add API keys for external services if AVA is configured to use them.
-   Ensure the model names specified in the configuration match the models you have pulled in Ollama.

## Installation Complete!

You should now have the necessary components installed to run AVA. Proceed to the [USAGE.md](USAGE.md) guide for instructions on running the assistant.
