#!/usr/bin/env python3
"""
AVA Installation Verification Script
=====================================

Verifies that AVA is correctly installed and all dependencies work.

Usage:
    python scripts/verify_install.py

Exit codes:
    0 - All checks passed
    1 - Some checks failed
"""

import subprocess
import sys
from pathlib import Path

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_header(text: str) -> None:
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}{text}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}\n")


def print_check(name: str, passed: bool, details: str = "") -> None:
    status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"  [{status}] {name}")
    if details:
        print(f"         {details}")


def check_python_version() -> bool:
    """Check Python version is 3.10+."""
    version = sys.version_info
    passed = version >= (3, 10)
    print_check("Python version", passed, f"Python {version.major}.{version.minor}.{version.micro}")
    return passed


def check_package(package: str, import_name: str = None) -> bool:
    """Check if a package is importable."""
    import_name = import_name or package
    try:
        __import__(import_name)
        print_check(f"Package: {package}", True)
        return True
    except ImportError:
        print_check(f"Package: {package}", False, f"pip install {package}")
        return False


def check_ollama() -> bool:
    """Check if Ollama is reachable."""
    try:
        import httpx

        response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_count = len(models)
            print_check("Ollama connection", True, f"{model_count} model(s) available")
            return True
        else:
            print_check("Ollama connection", False, f"Status: {response.status_code}")
            return False
    except ImportError:
        print_check("Ollama connection", False, "httpx not installed")
        return False
    except Exception:
        print_check("Ollama connection", False, "Start with: ollama serve")
        return False


def check_gpu() -> bool:
    """Check if NVIDIA GPU is available."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_info = result.stdout.strip().split("\n")[0]
            print_check("NVIDIA GPU", True, gpu_info)
            return True
        else:
            print_check("NVIDIA GPU", False, "nvidia-smi returned no data")
            return False
    except FileNotFoundError:
        print_check("NVIDIA GPU", False, "nvidia-smi not found (CPU mode available)")
        return False
    except Exception as e:
        print_check("NVIDIA GPU", False, str(e))
        return False


def check_project_structure() -> bool:
    """Check project directory structure."""
    root = Path(__file__).parent.parent
    required_paths = [
        "src/ava/__init__.py",
        "src/core/system.py",
        "src/core/medulla.py",
        "config/cortex_medulla.yaml",
        "requirements.txt",
    ]

    all_exist = True
    for path_str in required_paths:
        path = root / path_str
        if not path.exists():
            print_check(f"File: {path_str}", False, "Missing")
            all_exist = False

    if all_exist:
        print_check("Project structure", True, "All required files present")

    return all_exist


def check_ava_import() -> bool:
    """Check if AVA can be imported."""
    try:
        # Add src to path
        root = Path(__file__).parent.parent
        sys.path.insert(0, str(root / "src"))

        from ava import __version__

        print_check("AVA import", True, f"Version {__version__}")
        return True
    except ImportError as e:
        print_check("AVA import", False, str(e))
        return False
    except Exception as e:
        print_check("AVA import", False, str(e))
        return False


def main() -> int:
    print_header("AVA Installation Verification")

    results = []

    # Core checks
    print(f"{BOLD}Core Requirements:{RESET}")
    results.append(check_python_version())
    results.append(check_project_structure())
    results.append(check_ava_import())

    # Package checks
    print(f"\n{BOLD}Required Packages:{RESET}")
    packages = [
        ("aiohttp", "aiohttp"),
        ("httpx", "httpx"),
        ("typer", "typer"),
        ("rich", "rich"),
        ("textual", "textual"),
        ("pydantic", "pydantic"),
        ("numpy", "numpy"),
        ("pyyaml", "yaml"),
    ]
    for package, import_name in packages:
        results.append(check_package(package, import_name))

    # External services
    print(f"\n{BOLD}External Services:{RESET}")
    results.append(check_ollama())
    results.append(check_gpu())

    # Summary
    passed = sum(results)
    total = len(results)

    print_header("Summary")

    if all(results):
        print(f"{GREEN}{BOLD}All {total} checks passed!{RESET}")
        print("\nAVA is ready to use. Start with:")
        print(f"  {CYAN}python server.py{RESET}     # HTTP API server")
        print(f"  {CYAN}python run_tui.py{RESET}   # Terminal UI")
        print(f"  {CYAN}ava doctor{RESET}          # Quick diagnostics")
        return 0
    else:
        failed = total - passed
        print(f"{YELLOW}{BOLD}{passed}/{total} checks passed ({failed} failed){RESET}")
        print("\nTo fix missing packages:")
        print(f"  {CYAN}pip install -r requirements.txt{RESET}")
        print("\nTo start Ollama:")
        print(f"  {CYAN}ollama serve{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
