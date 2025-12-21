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
from typing import Dict, List, Optional, Tuple

# =============================================================================
# Configuration
# =============================================================================

# Ollama models to download
OLLAMA_MODELS = {
    "minimal": [
        "gemma3:4b",           # Small model for testing
        "nomic-embed-text",    # Embeddings
    ],
    "standard": [
        "gemma3:4b",           # Quick responses
        "llama3.2:latest",     # Better quality
        "nomic-embed-text",    # Embeddings
    ],
    "full": [
        "gemma3:4b",           # Quick responses
        "llama3.2:latest",     # Standard quality
        "llama3.1:70b-instruct-q4_0",  # Full Cortex (if VRAM allows)
        "nomic-embed-text",    # Embeddings
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

# Banner
BANNER = """
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
# Utility Functions
# =============================================================================

def print_step(step: int, total: int, message: str) -> None:
    """Print a formatted step message."""
    print(f"\n[{step}/{total}] {message}")
    print("-" * 60)


def print_success(message: str) -> None:
    """Print success message."""
    print(f"  [OK] {message}")


def print_warning(message: str) -> None:
    """Print warning message."""
    print(f"  [WARNING] {message}")


def print_error(message: str) -> None:
    """Print error message."""
    print(f"  [ERROR] {message}")


def run_command(
    cmd: List[str],
    check: bool = True,
    capture: bool = False,
    env: Optional[Dict] = None,
) -> Tuple[int, str, str]:
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

    # Python version
    py_version = sys.version_info
    if py_version >= (3, 9):
        print_success(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        print_error(f"Python 3.9+ required, found {py_version.major}.{py_version.minor}")
        all_ok = False

    # Platform
    print_success(f"Platform: {platform.system()} {platform.machine()}")

    # Ollama
    if check_command_exists("ollama"):
        print_success("Ollama installed")
    else:
        print_warning("Ollama not found - install from https://ollama.ai/")
        print_warning("  After installing, run: ollama serve")

    # Git (optional)
    if check_command_exists("git"):
        print_success("Git installed")
    else:
        print_warning("Git not found (optional)")

    # CUDA check
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            print_success(f"NVIDIA GPU: {gpu_info}")
        else:
            print_warning("NVIDIA GPU not detected (CPU-only mode)")
    except FileNotFoundError:
        print_warning("nvidia-smi not found (CPU-only mode)")

    return all_ok


def create_virtual_environment() -> bool:
    """Create Python virtual environment."""
    print_step(2, 7, "Creating Virtual Environment")

    venv_path = Path("venv")

    if venv_path.exists():
        print_success("Virtual environment already exists")
        return True

    try:
        import venv
        venv.create("venv", with_pip=True)
        print_success("Virtual environment created")
        return True
    except Exception as e:
        print_error(f"Failed to create venv: {e}")
        return False


def install_dependencies() -> bool:
    """Install Python dependencies."""
    print_step(3, 7, "Installing Dependencies")

    pip = get_pip_executable()

    # Upgrade pip
    print("  Upgrading pip...")
    run_command([pip, "install", "--upgrade", "pip"], check=False)

    # Install requirements
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        print("  Installing from requirements.txt...")
        code, _, err = run_command([pip, "install", "-r", "requirements.txt"])
        if code != 0:
            print_warning(f"Some packages may have failed: {err[:200]}")
    else:
        print_warning("requirements.txt not found")

    # Install optional v3 dependencies
    print("  Installing v3 architecture dependencies...")

    v3_packages = [
        "pynvml",           # GPU monitoring
        "psutil",           # System monitoring
        "aiofiles",         # Async file operations
    ]

    for pkg in v3_packages:
        code, _, _ = run_command([pip, "install", pkg], check=False, capture=True)
        if code == 0:
            print_success(f"Installed {pkg}")
        else:
            print_warning(f"Could not install {pkg} (optional)")

    print_success("Core dependencies installed")
    return True


def create_directories() -> bool:
    """Create required directories."""
    print_step(4, 7, "Creating Directories")

    for dir_path in DIRECTORIES:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        print_success(f"Created {dir_path}")

    return True


def start_ollama() -> bool:
    """Ensure Ollama is running."""
    print_step(5, 7, "Starting Ollama Service")

    if not check_command_exists("ollama"):
        print_warning("Ollama not installed - skipping")
        return False

    # Check if already running
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        print_success("Ollama is already running")
        return True
    except Exception:
        pass

    # Try to start Ollama
    print("  Starting Ollama service...")

    if platform.system() == "Windows":
        # On Windows, Ollama runs as a service or needs manual start
        print_warning("Please start Ollama manually: ollama serve")
        print("  Waiting 5 seconds...")
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
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
        print_success("Ollama is running")
        return True
    except Exception:
        print_warning("Could not connect to Ollama - models may need manual download")
        return False


def download_ollama_models(mode: str = "standard") -> bool:
    """Download Ollama models."""
    print_step(6, 7, f"Downloading Ollama Models ({mode} mode)")

    if not check_command_exists("ollama"):
        print_warning("Ollama not installed - skipping model downloads")
        return False

    models = OLLAMA_MODELS.get(mode, OLLAMA_MODELS["standard"])

    for model in models:
        print(f"\n  Downloading {model}...")
        code, stdout, stderr = run_command(
            ["ollama", "pull", model],
            check=False,
        )

        if code == 0:
            print_success(f"Downloaded {model}")
        else:
            print_warning(f"Failed to download {model}: {stderr[:100]}")

    return True


def validate_installation() -> bool:
    """Validate the installation."""
    print_step(7, 7, "Validating Installation")

    all_ok = True

    # Check venv
    python = get_python_executable()
    if Path(python).exists():
        print_success("Virtual environment: OK")
    else:
        print_error("Virtual environment: MISSING")
        all_ok = False

    # Check key imports
    check_imports = [
        ("httpx", "HTTP client"),
        ("yaml", "YAML parser"),
        ("numpy", "NumPy"),
        ("pydantic", "Pydantic"),
    ]

    for module, name in check_imports:
        code, _, _ = run_command(
            [python, "-c", f"import {module}"],
            check=False,
            capture=True,
        )
        if code == 0:
            print_success(f"{name}: OK")
        else:
            print_warning(f"{name}: Not installed")

    # Check directories
    for dir_path in DIRECTORIES[:3]:  # Check first few
        if Path(dir_path).exists():
            print_success(f"Directory {dir_path}: OK")
        else:
            print_warning(f"Directory {dir_path}: Missing")

    # Check config
    config_file = Path("config/cortex_medulla.yaml")
    if config_file.exists():
        print_success("Configuration: OK")
    else:
        print_warning("Configuration file missing")

    # Check Ollama models
    if check_command_exists("ollama"):
        code, stdout, _ = run_command(
            ["ollama", "list"],
            check=False,
            capture=True,
        )
        if code == 0 and stdout.strip():
            model_count = len(stdout.strip().split("\n")) - 1  # Exclude header
            print_success(f"Ollama models: {model_count} installed")
        else:
            print_warning("No Ollama models found")

    return all_ok


def print_next_steps() -> None:
    """Print next steps after setup."""
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("""
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
""")


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

    args = parser.parse_args()

    print(BANNER)

    # Determine model mode
    if args.minimal:
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
