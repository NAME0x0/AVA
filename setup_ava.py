#!/usr/bin/env python3
"""
AVA v3 One-Click Setup
======================

Automated setup script that:
1. Creates virtual environment
2. Installs all dependencies
3. Downloads required models (Ollama + HuggingFace)
4. Creates necessary directories
5. Validates installation

Usage:
    python setup_ava.py              # Full setup
    python setup_ava.py --models     # Download models only
    python setup_ava.py --check      # Validate installation
    python setup_ava.py --minimal    # Minimal setup (small models only)

Requirements:
    - Python 3.9+
    - NVIDIA GPU with CUDA support (recommended)
    - ~50GB disk space for full models
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

# Ollama models to download
OLLAMA_MODELS = {
    "minimal": [
        "gemma3:4b",  # Small model for testing
        "nomic-embed-text",  # Embeddings
    ],
    "standard": [
        "gemma3:4b",  # Quick responses
        "llama3.2:latest",  # Better quality
        "nomic-embed-text",  # Embeddings
    ],
    "full": [
        "gemma3:4b",  # Quick responses
        "llama3.2:latest",  # Standard quality
        "llama3.1:70b-instruct-q4_0",  # Full Cortex (if VRAM allows)
        "nomic-embed-text",  # Embeddings
    ],
}

# HuggingFace models for advanced features
HUGGINGFACE_MODELS = {
    "cortex": {
        "name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "description": "Cortex reasoning model (8B for testing, 70B for production)",
        "size_gb": 16,
        "required": False,
    },
    "mamba": {
        "name": "state-spaces/mamba-2.8b-slimpj",
        "description": "Mamba SSM for Medulla Monitor",
        "size_gb": 6,
        "required": False,
    },
}

# Required directories
DIRECTORIES = [
    "data/memory/episodic",
    "data/memory",
    "data/learning/samples",
    "data/learning/checkpoints",
    "data/conversations",
    "models/fine_tuned_adapters/bridge",
    "models/fine_tuned_adapters/experts",
    "config",
    "logs",
]

# Banner (using raw string to preserve backslashes in ASCII art)
BANNER = r"""
================================================================================
     ___ _    ___   _    ____  _____ _____ _   _ ____
    / _ \ \  / / \ | |  / ___|| ____|_   _| | | |  _ \
   | |_| \ \/ / _ \| | | \___ |  _|   | | | | | | |_) |
   |  _  ||  / ___ \ |  ___) | |___  | | | |_| |  __/
   |_| |_||_/_/   \_\_| |____/|_____| |_|  \___/|_|

    Cortex-Medulla Architecture v3
    One-Click Setup
================================================================================
"""


# =============================================================================
# Progress Tracking System
# =============================================================================


class ProgressTracker:
    """
    Live progress bar with percentage display.

    Shows minimalistic output with a single updating line:
    [Step 3/7] Installing Dependencies  [████████░░░░░░░░░░░░] 40%  httpx
    """

    def __init__(self, total_steps: int = 7, bar_width: int = 20):
        self.total_steps = total_steps
        self.bar_width = bar_width
        self.current_step = 0
        self.step_name = ""
        self.sub_task = ""
        self.step_percent = 0.0
        self.verbose = False
        self.messages: list[str] = []  # Collects [OK], [WARNING], [ERROR] messages

    def start_step(self, step: int, name: str) -> None:
        """Start a new major step."""
        self.current_step = step
        self.step_name = name
        self.step_percent = 0.0
        self.sub_task = ""
        self.messages = []
        # Print step header on new line
        print(f"\n[Step {step}/{self.total_steps}] {name}")
        self._render()

    def update(self, percent: float, sub_task: str = "") -> None:
        """Update progress within current step."""
        self.step_percent = min(100.0, max(0.0, percent))
        if sub_task:
            self.sub_task = sub_task
        self._render()

    def _render(self) -> None:
        """Render the progress bar on the same line."""
        filled = int(self.bar_width * self.step_percent / 100)
        bar = "█" * filled + "░" * (self.bar_width - filled)
        percent_str = f"{self.step_percent:3.0f}%"
        sub = f"  {self.sub_task[:30]}" if self.sub_task else ""
        # Use \r to overwrite the line
        line = f"\r  [{bar}] {percent_str}{sub}"
        # Pad to clear previous longer text
        print(f"{line:<70}", end="", flush=True)

    def success(self, message: str) -> None:
        """Log success message (shown at step completion)."""
        self.messages.append(f"  [OK] {message}")
        if self.verbose:
            print(f"\n  [OK] {message}", end="")
            self._render()

    def warning(self, message: str) -> None:
        """Log warning message (shown at step completion)."""
        self.messages.append(f"  [WARN] {message}")
        if self.verbose:
            print(f"\n  [WARN] {message}", end="")
            self._render()

    def error(self, message: str) -> None:
        """Log error message (always shown immediately)."""
        print(f"\n  [ERROR] {message}")
        self.messages.append(f"  [ERROR] {message}")

    def complete_step(self) -> None:
        """Mark current step as complete."""
        self.step_percent = 100.0
        self._render()
        print()  # New line after progress bar
        # Show summary of messages
        ok_count = sum(1 for m in self.messages if "[OK]" in m)
        warn_count = sum(1 for m in self.messages if "[WARN]" in m)
        err_count = sum(1 for m in self.messages if "[ERROR]" in m)
        if ok_count or warn_count or err_count:
            summary = f"  Done: {ok_count} OK"
            if warn_count:
                summary += f", {warn_count} warnings"
            if err_count:
                summary += f", {err_count} errors"
            print(summary)

    def overall_progress(self) -> float:
        """Calculate overall progress across all steps."""
        base = (self.current_step - 1) / self.total_steps * 100
        step_contrib = self.step_percent / self.total_steps
        return base + step_contrib


# Global progress tracker
progress = ProgressTracker(total_steps=7)


# =============================================================================
# Utility Functions (Legacy wrappers for compatibility)
# =============================================================================


def print_step(step: int, total: int, message: str) -> None:
    """Print a formatted step message (legacy wrapper)."""
    progress.start_step(step, message)


def print_success(message: str) -> None:
    """Print success message (legacy wrapper)."""
    progress.success(message)


def print_warning(message: str) -> None:
    """Print warning message (legacy wrapper)."""
    progress.warning(message)


def print_error(message: str) -> None:
    """Print error message (legacy wrapper)."""
    progress.error(message)


def run_command(
    cmd: list[str],
    check: bool = True,
    capture: bool = False,
    env: dict | None = None,
) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=capture,
            text=True,
            env=env or os.environ.copy(),
        )
        return result.returncode, result.stdout or "", result.stderr or ""
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout or "", e.stderr or ""
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"


def check_command_exists(cmd: str) -> bool:
    """Check if a command exists on the system."""
    return shutil.which(cmd) is not None


def get_python_executable() -> str:
    """Get the Python executable path."""
    if platform.system() == "Windows":
        venv_python = Path("venv/Scripts/python.exe")
    else:
        venv_python = Path("venv/bin/python")

    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def get_pip_executable() -> str:
    """Get the pip executable path."""
    if platform.system() == "Windows":
        venv_pip = Path("venv/Scripts/pip.exe")
    else:
        venv_pip = Path("venv/bin/pip")

    if venv_pip.exists():
        return str(venv_pip)
    return f"{sys.executable} -m pip"


# =============================================================================
# Setup Steps
# =============================================================================


def check_prerequisites() -> bool:
    """Check system prerequisites."""
    print_step(1, 7, "Checking Prerequisites")

    all_ok = True

    # Python version (20%)
    progress.update(10, "Checking Python")
    py_version = sys.version_info
    if py_version >= (3, 9):
        print_success(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        print_error(f"Python 3.9+ required, found {py_version.major}.{py_version.minor}")
        all_ok = False
    progress.update(20)

    # Platform (40%)
    progress.update(30, "Checking platform")
    print_success(f"Platform: {platform.system()} {platform.machine()}")
    progress.update(40)

    # Ollama (60%)
    progress.update(50, "Checking Ollama")
    if check_command_exists("ollama"):
        print_success("Ollama installed")
    else:
        print_warning("Ollama not found - install from https://ollama.ai/")
    progress.update(60)

    # Git (80%)
    progress.update(70, "Checking Git")
    if check_command_exists("git"):
        print_success("Git installed")
    else:
        print_warning("Git not found (optional)")
    progress.update(80)

    # CUDA check (100%)
    progress.update(90, "Checking GPU")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            print_success(f"NVIDIA GPU: {gpu_info}")
        else:
            print_warning("NVIDIA GPU not detected (CPU-only)")
    except FileNotFoundError:
        print_warning("nvidia-smi not found (CPU-only)")

    progress.complete_step()
    return all_ok


def create_virtual_environment() -> bool:
    """Create Python virtual environment."""
    print_step(2, 7, "Creating Virtual Environment")

    venv_path = Path("venv")

    progress.update(20, "Checking existing venv")
    if venv_path.exists():
        print_success("Virtual environment already exists")
        progress.complete_step()
        return True

    progress.update(50, "Creating venv")
    try:
        import venv

        venv.create("venv", with_pip=True)
        print_success("Virtual environment created")
        progress.complete_step()
        return True
    except Exception as e:
        print_error(f"Failed to create venv: {e}")
        progress.complete_step()
        return False


def install_dependencies() -> bool:
    """Install Python dependencies."""
    print_step(3, 7, "Installing Dependencies")

    pip = get_pip_executable()

    # Upgrade pip (10%)
    progress.update(5, "Upgrading pip")
    run_command([pip, "install", "--upgrade", "pip"], check=False)
    progress.update(10)

    # Install requirements (70%)
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        progress.update(15, "Installing requirements.txt")
        code, _, err = run_command([pip, "install", "-r", "requirements.txt"])
        if code != 0:
            print_warning("Some packages may have failed")
        else:
            print_success("requirements.txt installed")
        progress.update(70)
    else:
        print_warning("requirements.txt not found")
        progress.update(70)

    # Install optional v3 dependencies (100%)
    v3_packages = [
        "pynvml",  # GPU monitoring
        "psutil",  # System monitoring
        "aiofiles",  # Async file operations
    ]

    for i, pkg in enumerate(v3_packages):
        pct = 70 + (i + 1) * 10
        progress.update(pct, f"Installing {pkg}")
        code, _, _ = run_command([pip, "install", pkg], check=False, capture=True)
        if code == 0:
            print_success(f"Installed {pkg}")
        else:
            print_warning(f"Could not install {pkg} (optional)")

    print_success("Core dependencies installed")
    progress.complete_step()
    return True


def create_directories() -> bool:
    """Create required directories."""
    print_step(4, 7, "Creating Directories")

    total_dirs = len(DIRECTORIES)
    for i, dir_path in enumerate(DIRECTORIES):
        pct = int((i + 1) / total_dirs * 100)
        progress.update(pct, dir_path)
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        print_success(f"Created {dir_path}")

    progress.complete_step()
    return True


def start_ollama() -> bool:
    """Ensure Ollama is running."""
    print_step(5, 7, "Starting Ollama Service")

    progress.update(10, "Checking Ollama")
    if not check_command_exists("ollama"):
        print_warning("Ollama not installed - skipping")
        progress.complete_step()
        return False

    # Check if already running
    progress.update(30, "Checking if running")
    import urllib.request

    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        print_success("Ollama is already running")
        progress.complete_step()
        return True
    except Exception:
        pass

    # Try to start Ollama
    progress.update(50, "Starting service")

    if platform.system() == "Windows":
        # On Windows, Ollama runs as a service or needs manual start
        print_warning("Please start Ollama manually: ollama serve")
        progress.update(70, "Waiting...")
        time.sleep(5)
    else:
        # On Linux/Mac, start in background
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(3)

    # Verify it's running
    progress.update(90, "Verifying connection")
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
        print_success("Ollama is running")
        progress.complete_step()
        return True
    except Exception:
        print_warning("Could not connect to Ollama")
        progress.complete_step()
        return False


def download_ollama_models(mode: str = "standard") -> bool:
    """Download Ollama models."""
    print_step(6, 7, f"Downloading Ollama Models ({mode} mode)")

    progress.update(5, "Checking Ollama")
    if not check_command_exists("ollama"):
        print_warning("Ollama not installed - skipping")
        progress.complete_step()
        return False

    models = OLLAMA_MODELS.get(mode, OLLAMA_MODELS["standard"])
    total_models = len(models)

    for i, model in enumerate(models):
        pct = int((i / total_models) * 90) + 10
        progress.update(pct, f"Pulling {model}")
        code, stdout, stderr = run_command(
            ["ollama", "pull", model],
            check=False,
        )

        if code == 0:
            print_success(f"Downloaded {model}")
        else:
            print_warning(f"Failed to download {model}")

    progress.complete_step()
    return True


def validate_installation() -> bool:
    """Validate the installation."""
    print_step(7, 7, "Validating Installation")

    all_ok = True

    # Check venv (20%)
    progress.update(10, "Checking venv")
    python = get_python_executable()
    if Path(python).exists():
        print_success("Virtual environment: OK")
    else:
        print_error("Virtual environment: MISSING")
        all_ok = False
    progress.update(20)

    # Check key imports (60%)
    check_imports = [
        ("httpx", "HTTP client"),
        ("yaml", "YAML parser"),
        ("numpy", "NumPy"),
        ("pydantic", "Pydantic"),
    ]

    for i, (module, name) in enumerate(check_imports):
        pct = 20 + int((i + 1) / len(check_imports) * 40)
        progress.update(pct, f"Checking {name}")
        code, _, _ = run_command(
            [python, "-c", f"import {module}"],
            check=False,
            capture=True,
        )
        if code == 0:
            print_success(f"{name}: OK")
        else:
            print_warning(f"{name}: Not installed")

    # Check directories (75%)
    progress.update(65, "Checking directories")
    for dir_path in DIRECTORIES[:3]:  # Check first few
        if Path(dir_path).exists():
            print_success(f"Directory {dir_path}: OK")
        else:
            print_warning(f"Directory {dir_path}: Missing")
    progress.update(75)

    # Check config (85%)
    progress.update(80, "Checking config")
    config_file = Path("config/cortex_medulla.yaml")
    if config_file.exists():
        print_success("Configuration: OK")
    else:
        print_warning("Configuration file missing")
    progress.update(85)

    # Check Ollama models (100%)
    progress.update(90, "Checking Ollama models")
    if check_command_exists("ollama"):
        code, stdout, _ = run_command(
            ["ollama", "list"],
            check=False,
            capture=True,
        )
        if code == 0 and stdout.strip():
            model_count = len(stdout.strip().split("\n")) - 1
            print_success(f"Ollama models: {model_count} installed")
        else:
            print_warning("No Ollama models found")

    progress.complete_step()
    return all_ok


def print_next_steps() -> None:
    """Print next steps after setup."""
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print(
        """
Next Steps:
-----------

1. Start the AVA server:
   python server.py

2. Or run the core system:
   python run_core.py

3. Test with curl:
   curl -X POST http://localhost:8085/chat \\
     -H "Content-Type: application/json" \\
     -d '{"message": "Hello AVA!"}'

4. View status:
   curl http://localhost:8085/status

Configuration:
--------------
- Main config: config/cortex_medulla.yaml
- Set simulation_mode: false for production (requires models)
- Set simulation_mode: true for testing without models

Documentation:
--------------
- README.md - Overview
- docs/CORTEX_MEDULLA.md - Architecture details
- TODO.md - Implementation status
"""
    )


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main setup entry point."""
    parser = argparse.ArgumentParser(
        description="AVA v3 One-Click Setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_ava.py              # Full setup with standard models
  python setup_ava.py --minimal    # Minimal setup with small models
  python setup_ava.py --full       # Full setup with all models (70B)
  python setup_ava.py --models     # Download models only
  python setup_ava.py --check      # Validate installation
        """,
    )

    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Install minimal models only (gemma3:4b)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Install all models including 70B (requires 50GB+ disk)",
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="Download models only (skip environment setup)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate installation only",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip model downloads",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output (all [OK]/[WARN] messages)",
    )

    args = parser.parse_args()

    # Configure progress tracker
    progress.verbose = args.verbose

    print(BANNER)

    # Detect beginner mode (first-time user)
    is_first_time = not Path("venv").exists() and not Path("config").exists()
    if is_first_time and not any([args.minimal, args.full, args.models, args.check]):
        print("Welcome! Detected first-time setup.")
        print("Using minimal mode (smaller models for faster setup).")
        print("Run with --full later for complete experience.\n")

    # Determine model mode
    if args.minimal or is_first_time:
        model_mode = "minimal"
    elif args.full:
        model_mode = "full"
    else:
        model_mode = "standard"

    # Check only mode
    if args.check:
        success = validate_installation()
        sys.exit(0 if success else 1)

    # Models only mode
    if args.models:
        start_ollama()
        download_ollama_models(model_mode)
        sys.exit(0)

    # Full setup
    success = True

    success = check_prerequisites() and success
    success = create_virtual_environment() and success
    success = install_dependencies() and success
    success = create_directories() and success

    if not args.skip_models:
        start_ollama()
        download_ollama_models(model_mode)

    validate_installation()
    print_next_steps()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
