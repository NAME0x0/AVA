# AVA - Autonomous Virtual Assistant

## Project Overview

AVA (Autonomous Virtual Assistant) is a research-grade AI assistant featuring a biomimetic "Cortex-Medulla" dual-brain architecture. It is designed to run locally on constrained hardware (targeting 4GB VRAM) by intelligently routing queries between a fast, reflexive model ("Medulla") and a deep reasoning model ("Cortex").

**Key Features:**
*   **Dual-Brain Architecture:** 
    *   **Medulla:** Always-on, fast response (Mamba/BitNet based).
    *   **Cortex:** High-latency, deep reasoning (70B+ models via AirLLM paging).
*   **Unified Backend:** A Rust-based single executable (Axum server) handles core logic and orchestration, replacing the legacy Python server for the desktop app.
*   **Frontend Interfaces:**
    *   **Desktop:** Tauri + Next.js (React/TypeScript).
    *   **Terminal (TUI):** Python-based interface using Textual.
*   **Active Inference:** Uses the Free Energy Principle for autonomous decision-making ("Agency").
*   **Memory:** "Titans" architecture for test-time learning and infinite context.

## Technology Stack

*   **Core Backend:** Rust (Axum, Tokio)
*   **Desktop UI:** TypeScript, React, Next.js, Tauri
*   **Terminal UI:** Python (Textual)
*   **AI Inference:** Ollama (external dependency), AirLLM
*   **Legacy/Scripting:** Python 3.10+

## Development Workflow

### Prerequisites
*   **Ollama:** Must be installed and running (`ollama serve`).
*   **Node.js:** v18+
*   **Rust:** Stable toolchain
*   **Python:** 3.10+

### Desktop App (Rust + TypeScript)

The desktop application is located in the `ui/` directory.

*   **Install Dependencies:**
    ```bash
    cd ui
    npm install
    ```
*   **Run in Development Mode:**
    ```bash
    npm run tauri dev
    ```
*   **Build for Production:**
    ```bash
    npm run tauri build
    ```
*   **Rust Tests:**
    ```bash
    cd ui/src-tauri
    cargo test
    ```

### Terminal UI (Python)

The TUI is located in `tui/` and relies on the Python environment.

*   **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    ```
*   **Run TUI (Dev):**
    ```bash
    textual run --dev tui.app:AVATUI
    ```
*   **Python Tests:**
    ```bash
    pytest tests/
    ```

### Python Core & Scripts

*   **Linting:** `flake8 src/`, `black src/ --check`
*   **Type Checking:** `mypy src/`

## Coding Conventions

### General
*   **Commit Messages:** Follow [Conventional Commits](https://www.conventionalcommits.org/) (e.g., `feat(medulla): add thermal throttling`).

### Python
*   Follow **PEP 8**.
*   Use **Type Hints** extensively.
*   Use `black` for formatting.
*   Use `dataclasses` for data containers.

### TypeScript/React
*   Use **Functional Components** and Hooks.
*   Use **Tailwind CSS** for styling.
*   Ensure strict typing with TypeScript interfaces.

## Directory Structure

*   `ui/` - Desktop application (Next.js frontend + Rust backend in `src-tauri`).
*   `tui/` - Terminal User Interface (Textual).
*   `src/` - Python core libraries (Architecture, Agency, Memory).
*   `config/` - Configuration files (`ava.yaml`, `cortex_medulla.yaml`).
*   `docs/` - Comprehensive documentation (`ARCHITECTURE.md`, `DEVELOPMENT.md`).
*   `models/` - Model adapters and base models.
*   `installer/` - Windows installer scripts (NSIS).
