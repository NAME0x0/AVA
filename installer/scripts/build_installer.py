#!/usr/bin/env python3
"""
AVA Installer Build Script
==========================

Builds the Windows installer using NSIS or WiX.

Usage:
    python build_installer.py              # Build installer
    python build_installer.py --portable   # Build portable ZIP
    python build_installer.py --all        # Build all variants
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

# Paths
SCRIPT_DIR = Path(__file__).parent
INSTALLER_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = INSTALLER_DIR.parent
CONFIG_FILE = INSTALLER_DIR / "config" / "installer.yaml"
VERSION_FILE = PROJECT_ROOT / "VERSION"
BUILD_DIR = INSTALLER_DIR / "build"
DIST_DIR = INSTALLER_DIR / "dist"


def get_version() -> str:
    """Read version from VERSION file."""
    if VERSION_FILE.exists():
        return VERSION_FILE.read_text().strip()
    return "0.0.0"


def load_config() -> dict:
    """Load installer configuration."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return yaml.safe_load(f)
    return {}


def check_prerequisites() -> bool:
    """Check that all build prerequisites are installed."""
    print("Checking prerequisites...")

    # Check for NSIS
    nsis_path = shutil.which("makensis")
    if nsis_path:
        print(f"  [OK] NSIS found: {nsis_path}")
    else:
        print("  [WARN] NSIS not found - install from https://nsis.sourceforge.io/")
        return False

    # Check for Rust/Cargo
    cargo_path = shutil.which("cargo")
    if cargo_path:
        print(f"  [OK] Cargo found: {cargo_path}")
    else:
        print("  [WARN] Cargo not found - install from https://rustup.rs/")
        return False

    # Check for Node.js
    node_path = shutil.which("node")
    if node_path:
        print(f"  [OK] Node.js found: {node_path}")
    else:
        print("  [WARN] Node.js not found - install from https://nodejs.org/")
        return False

    return True


def build_tauri_app() -> bool:
    """Build the Tauri desktop application."""
    print("\nBuilding Tauri application...")

    ui_dir = PROJECT_ROOT / "ui"
    if not ui_dir.exists():
        print("  [ERROR] ui/ directory not found")
        return False

    # Install npm dependencies
    print("  Installing npm dependencies...")
    result = subprocess.run(
        ["npm", "install"],
        cwd=ui_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  [ERROR] npm install failed: {result.stderr}")
        return False

    # Build Tauri app
    print("  Building Tauri app (this may take a few minutes)...")
    result = subprocess.run(
        ["npm", "run", "tauri", "build"],
        cwd=ui_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  [ERROR] Tauri build failed: {result.stderr}")
        return False

    print("  [OK] Tauri app built successfully")
    return True


def build_nsis_installer() -> bool:
    """Build the NSIS installer."""
    print("\nBuilding NSIS installer...")

    nsis_script = INSTALLER_DIR / "nsis" / "ava-installer.nsi"
    if not nsis_script.exists():
        print("  [ERROR] NSIS script not found")
        return False

    version = get_version()
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["makensis", f"-DVERSION={version}", str(nsis_script)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  [ERROR] NSIS build failed: {result.stderr}")
        return False

    print(f"  [OK] Installer built: dist/AVA-{version}-Setup.exe")
    return True


def build_portable() -> bool:
    """Build portable ZIP distribution."""
    print("\nBuilding portable distribution...")

    version = get_version()
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    # Create portable directory
    portable_dir = BUILD_DIR / f"AVA-{version}-portable"
    if portable_dir.exists():
        shutil.rmtree(portable_dir)
    portable_dir.mkdir(parents=True)

    # Copy Tauri executable
    tauri_exe = PROJECT_ROOT / "ui" / "src-tauri" / "target" / "release" / "ava-ui.exe"
    if tauri_exe.exists():
        print("  Copying Tauri executable...")
        shutil.copy2(tauri_exe, portable_dir / "AVA.exe")
    else:
        print("  [WARN] Tauri executable not found, skipping")

    # Copy Python backend
    print("  Copying Python backend...")
    src_dir = PROJECT_ROOT / "src"
    if src_dir.exists():
        shutil.copytree(src_dir, portable_dir / "src", dirs_exist_ok=True)

    # Copy essential files
    essential_files = [
        "server.py",
        "run_tui.py",
        "run_core.py",
        "requirements.txt",
        "VERSION",
        "README.md",
        "LICENSE",
    ]
    for filename in essential_files:
        src_file = PROJECT_ROOT / filename
        if src_file.exists():
            shutil.copy2(src_file, portable_dir / filename)

    # Copy config directory
    config_dir = PROJECT_ROOT / "config"
    if config_dir.exists():
        shutil.copytree(config_dir, portable_dir / "config", dirs_exist_ok=True)

    # Copy TUI directory
    tui_dir = PROJECT_ROOT / "tui"
    if tui_dir.exists():
        shutil.copytree(tui_dir, portable_dir / "tui", dirs_exist_ok=True)

    # Create launcher batch file
    launcher_content = """@echo off
echo Starting AVA...
echo.

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found! Please install Python 3.10 or later.
    echo Download from: https://python.org/downloads/
    pause
    exit /b 1
)

REM Check for dependencies
if not exist "%~dp0\\venv" (
    echo Creating virtual environment...
    python -m venv "%~dp0\\venv"
    call "%~dp0\\venv\\Scripts\\activate.bat"
    pip install -r "%~dp0\\requirements.txt" --quiet
) else (
    call "%~dp0\\venv\\Scripts\\activate.bat"
)

REM Start the backend server in background
start /b python "%~dp0\\server.py"

REM Start the GUI
if exist "%~dp0\\AVA.exe" (
    start "" "%~dp0\\AVA.exe"
) else (
    echo GUI executable not found. Starting in TUI mode...
    python "%~dp0\\run_tui.py"
)
"""
    launcher_path = portable_dir / "Start AVA.bat"
    launcher_path.write_text(launcher_content)

    # Create ZIP
    print("  Creating ZIP archive...")
    zip_path = DIST_DIR / f"AVA-{version}-portable.zip"
    shutil.make_archive(
        str(zip_path).replace(".zip", ""),
        "zip",
        BUILD_DIR,
        f"AVA-{version}-portable",
    )

    print(f"  [OK] Portable build: dist/AVA-{version}-portable.zip")
    return True


def main():
    parser = argparse.ArgumentParser(description="Build AVA installer")
    parser.add_argument("--portable", action="store_true", help="Build portable ZIP")
    parser.add_argument("--all", action="store_true", help="Build all variants")
    parser.add_argument("--skip-tauri", action="store_true", help="Skip Tauri build")
    args = parser.parse_args()

    print("=" * 60)
    print("AVA Installer Build Script")
    print("=" * 60)

    version = get_version()
    print(f"Version: {version}")
    print()

    if not check_prerequisites():
        print("\nPrerequisites not met. Please install missing tools.")
        sys.exit(1)

    if not args.skip_tauri:
        if not build_tauri_app():
            sys.exit(1)

    if args.portable or args.all:
        if not build_portable():
            sys.exit(1)

    if not args.portable or args.all:
        if not build_nsis_installer():
            sys.exit(1)

    print("\n" + "=" * 60)
    print("Build completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
