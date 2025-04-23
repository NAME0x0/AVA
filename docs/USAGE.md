# AVA Usage Guide

This guide explains how to run and interact with AVA after completing the installation steps in [INSTALLATION.md](INSTALLATION.md).

## Prerequisites

-   AVA project successfully installed.
-   Ollama installed and running.
-   The required language model(s) (e.g., `deepseek-r1:8b`) pulled via Ollama.

## 1. Ensure Ollama Service is Running

Before starting AVA, make sure the Ollama service is active and the model you intend to use is available.

-   You can check running models with `ollama list` in your terminal.
-   If the model isn't running, Ollama will typically load it on the first request, which might take some time. You can pre-load it by running:

    ```bash
    ollama run deepseek-r1:8b "Hello!"
    ```

    (Replace `deepseek-r1:8b` with the actual model name if different). Exit the Ollama prompt after it loads (usually with `/bye`).

## 2. Running AVA

The primary way to interact with AVA depends on the implemented interfaces. Based on the project structure:

### Option A: Command Line Interface (CLI)

If a dedicated CLI script exists:

```bash
# Ensure your virtual environment is activated
# e.g., source .venv/bin/activate or .\.venv\Scripts\activate

# Run the CLI script
python src/interfaces/cli.py
```

Follow the prompts within the CLI to interact with AVA.

### Option B: Web Interface

If a web interface script exists:

```bash
# Ensure your virtual environment is activated
python src/interfaces/web_interface.py
```

This will likely start a local web server. Open your web browser and navigate to the specified address (e.g., `http://127.0.0.1:5000` - check the script's output for the correct URL).

### Option C: Example Demo Script

If a demonstration script is provided:

```bash
# Ensure your virtual environment is activated
python examples/demo.py
```

This script might showcase specific functionalities.

**Note:** Check the specific implementation details or run the scripts with `-h` or `--help` (if argument parsing is implemented) to confirm the correct way to launch AVA.

## 3. Interacting with AVA

Once running, you can interact with AVA using:

-   **Text Commands:** Type commands or questions into the CLI or web interface.
-   **Voice Commands:** If speech recognition (`src/modules/speech_recognition.py`) and audio processing (`src/modules/audio_processing.py`) are fully integrated and configured, you might be able to use voice input. This typically requires a microphone setup and may need specific activation phrases or buttons.

## 4. Using Custom Trained Models (via Ollama)

If you followed the `DEVELOPMENT.md` guide to train and export a custom model (e.g., `AVA.gguf`) and created an Ollama `Modelfile`:

1.  **Build the Ollama Model:**

    ```bash
    ollama create AVA -f path/to/your/Modelfile
    ```

2.  **Run the Custom Model:**

    ```bash
    ollama run AVA
    ```
    
3.  **Configure AVA (Code):** You might need to update the configuration in `src/core/config.py` or relevant modules (like `src/modules/text_generation.py`) to point to `AVA` (or whatever you named your custom Ollama model) instead of the default `deepseek-r1:8b`.
4.  **Run AVA:** Start the AVA interface as described in Step 2. It should now use your custom model via Ollama.

## 5. Stopping AVA

-   **CLI/Demo:** Usually `Ctrl+C` in the terminal where it's running.
-   **Web Interface:** `Ctrl+C` in the terminal where the server was started.
-   **Ollama Service:** Ollama typically continues running in the background. You usually don't need to stop it unless you want to free up resources. Refer to Ollama documentation for managing the service.
