#!/usr/bin/env python3
"""
Build AVA Server Sidecar

This script:
1. Builds the server.py as a PyInstaller executable
2. Copies the executable to the Tauri binaries directory
3. Names it according to Tauri sidecar conventions

Usage:
    python scripts/build_sidecar.py
    python scripts/build_sidecar.py --clean

Requirements:
    pip install pyinstaller
"""

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
TAURI_BIN_DIR = PROJECT_ROOT / "ui" / "src-tauri" / "binaries"
SPEC_FILE = PROJECT_ROOT / "server.spec"
DIST_DIR = PROJECT_ROOT / "dist"


def get_target_triple() -> str:
    """Get the Rust target triple for the current platform."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows":
        if machine in ("amd64", "x86_64"):
            return "x86_64-pc-windows-msvc"
        elif machine in ("arm64", "aarch64"):
            return "aarch64-pc-windows-msvc"
        else:
            return "i686-pc-windows-msvc"
    elif system == "darwin":
        if machine in ("arm64", "aarch64"):
            return "aarch64-apple-darwin"
        else:
            return "x86_64-apple-darwin"
    elif system == "linux":
        if machine in ("amd64", "x86_64"):
            return "x86_64-unknown-linux-gnu"
        elif machine in ("arm64", "aarch64"):
            return "aarch64-unknown-linux-gnu"
        else:
            return "i686-unknown-linux-gnu"

    raise RuntimeError(f"Unsupported platform: {system} {machine}")


def get_exe_extension() -> str:
    """Get the executable extension for the current platform."""
    return ".exe" if platform.system() == "Windows" else ""


def clean_build() -> None:
    """Clean previous build artifacts."""
    print("Cleaning previous build artifacts...")

    dirs_to_clean = [
        PROJECT_ROOT / "build",
        PROJECT_ROOT / "dist",
        PROJECT_ROOT / "__pycache__",
    ]

    for d in dirs_to_clean:
        if d.exists():
            shutil.rmtree(d)
            print(f"  Removed: {d}")


def build_sidecar(clean: bool = False) -> Path:
    """Build the PyInstaller sidecar executable."""

    if clean:
        clean_build()

    print(f"\n{'='*60}")
    print("Building AVA Server Sidecar")
    print(f"{'='*60}\n")

    # Check PyInstaller is available
    try:
        result = subprocess.run(
            [sys.executable, "-m", "PyInstaller", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"PyInstaller version: {result.stdout.strip()}")
    except subprocess.CalledProcessError:
        print("ERROR: PyInstaller not found. Install with: pip install pyinstaller")
        sys.exit(1)

    # Check spec file exists
    if not SPEC_FILE.exists():
        print(f"ERROR: Spec file not found: {SPEC_FILE}")
        sys.exit(1)

    print(f"\nSpec file: {SPEC_FILE}")
    print(f"Output dir: {DIST_DIR}")

    # Run PyInstaller
    print("\nRunning PyInstaller...\n")
    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--clean", "--noconfirm", str(SPEC_FILE)],
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        print("\nERROR: PyInstaller build failed!")
        sys.exit(1)

    # Get the built executable
    ext = get_exe_extension()
    src_exe = DIST_DIR / f"ava-server{ext}"

    if not src_exe.exists():
        print(f"\nERROR: Built executable not found: {src_exe}")
        sys.exit(1)

    print(f"\nBuild successful: {src_exe}")
    print(f"Size: {src_exe.stat().st_size / (1024*1024):.1f} MB")

    return src_exe


def install_sidecar(src_exe: Path) -> Path:
    """Copy the sidecar to Tauri binaries directory."""

    print(f"\n{'='*60}")
    print("Installing Sidecar to Tauri")
    print(f"{'='*60}\n")

    # Create binaries directory
    TAURI_BIN_DIR.mkdir(parents=True, exist_ok=True)

    # Get target triple and extension
    target_triple = get_target_triple()
    ext = get_exe_extension()

    # Tauri expects: binaries/ava-server-{target-triple}[.exe]
    dst_exe = TAURI_BIN_DIR / f"ava-server-{target_triple}{ext}"

    print(f"Target triple: {target_triple}")
    print(f"Source: {src_exe}")
    print(f"Destination: {dst_exe}")

    # Copy executable
    shutil.copy2(src_exe, dst_exe)

    # Make executable on Unix
    if platform.system() != "Windows":
        dst_exe.chmod(0o755)

    print("\nSidecar installed successfully!")
    print(f"Size: {dst_exe.stat().st_size / (1024*1024):.1f} MB")

    # Update tauri.conf.json to include the sidecar
    update_tauri_config()

    return dst_exe


def update_tauri_config() -> None:
    """Update tauri.conf.json to include the sidecar in externalBin."""
    import json

    config_path = PROJECT_ROOT / "ui" / "src-tauri" / "tauri.conf.json"

    print(f"\nUpdating {config_path}...")

    with open(config_path) as f:
        config = json.load(f)

    # Update externalBin to include the sidecar
    if "tauri" in config and "bundle" in config["tauri"]:
        external_bin = config["tauri"]["bundle"].get("externalBin", [])
        sidecar_entry = "binaries/ava-server"

        if sidecar_entry not in external_bin:
            external_bin.append(sidecar_entry)
            config["tauri"]["bundle"]["externalBin"] = external_bin

            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            print(f"  Added '{sidecar_entry}' to externalBin")
        else:
            print(f"  '{sidecar_entry}' already in externalBin")
    else:
        print("  Warning: Could not find tauri.bundle in config")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Build AVA server sidecar for Tauri")
    parser.add_argument(
        "--clean", action="store_true", help="Clean build artifacts before building"
    )
    parser.add_argument(
        "--build-only", action="store_true", help="Only build, don't install to Tauri"
    )
    args = parser.parse_args()

    # Build
    exe_path = build_sidecar(clean=args.clean)

    # Install
    if not args.build_only:
        install_sidecar(exe_path)
        print(f"\n{'='*60}")
        print("Done! Sidecar is ready for Tauri bundling.")
        print(f"{'='*60}\n")
        print("Next steps:")
        print("  1. Run: cd ui && npm run tauri:build")
        print("  2. Find installer in: ui/src-tauri/target/release/bundle/")
    else:
        print(f"\nBuild complete: {exe_path}")


if __name__ == "__main__":
    main()
