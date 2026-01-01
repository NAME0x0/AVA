#!/usr/bin/env python3
"""
Build Portable AVA Application
==============================

This script builds the unified AVA application (Rust backend + Tauri frontend)
into a single portable executable.

Usage:
    python scripts/build_portable.py              # Build for current platform
    python scripts/build_portable.py --release    # Build optimized release
    python scripts/build_portable.py --debug      # Build debug version

Output:
    dist/AVA-portable-{version}-{platform}.zip

Requirements:
    - Rust toolchain (rustup)
    - Node.js and npm
    - Tauri CLI: npm install -g @tauri-apps/cli
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
UI_DIR = PROJECT_ROOT / "ui"
TAURI_DIR = UI_DIR / "src-tauri"
DIST_DIR = PROJECT_ROOT / "dist"

# Version from Cargo.toml
VERSION = "3.3.3"


def run_command(cmd: list[str], cwd: Path = None, env: dict = None) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=merged_env,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"Command not found: {cmd[0]}")
        return False


def check_prerequisites() -> bool:
    """Check that all build tools are available."""
    print("Checking prerequisites...")
    
    checks = [
        (["rustc", "--version"], "Rust compiler"),
        (["cargo", "--version"], "Cargo"),
        (["node", "--version"], "Node.js"),
        (["npm", "--version"], "npm"),
    ]
    
    all_ok = True
    for cmd, name in checks:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.strip().split('\n')[0]
                print(f"  ✓ {name}: {version}")
            else:
                print(f"  ✗ {name}: not working")
                all_ok = False
        except FileNotFoundError:
            print(f"  ✗ {name}: not found")
            all_ok = False
    
    # Check for Tauri CLI
    try:
        result = subprocess.run(
            ["npm", "list", "@tauri-apps/cli"],
            cwd=UI_DIR,
            capture_output=True,
            text=True,
        )
        if "@tauri-apps/cli" in result.stdout:
            print("  ✓ Tauri CLI: installed")
        else:
            print("  ⚠ Tauri CLI: not installed (will install)")
    except Exception:
        print("  ⚠ Tauri CLI: unknown status")
    
    return all_ok


def install_dependencies() -> bool:
    """Install npm dependencies."""
    print("\nInstalling npm dependencies...")
    return run_command(["npm", "install"], cwd=UI_DIR)


def build_frontend() -> bool:
    """Build the Next.js frontend."""
    print("\nBuilding frontend...")
    return run_command(["npm", "run", "build"], cwd=UI_DIR)


def build_tauri(release: bool = True) -> bool:
    """Build the Tauri application."""
    print(f"\nBuilding Tauri application ({'release' if release else 'debug'})...")
    
    cmd = ["npm", "run", "tauri", "build"]
    if not release:
        cmd.append("--debug")
    
    # Set environment for optimized build
    env = {
        "CARGO_PROFILE_RELEASE_LTO": "true",
        "CARGO_PROFILE_RELEASE_CODEGEN_UNITS": "1",
        "CARGO_PROFILE_RELEASE_OPT_LEVEL": "3",
    }
    
    return run_command(cmd, cwd=UI_DIR, env=env)


def get_platform_info() -> tuple[str, str, str]:
    """Get platform-specific information."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "windows":
        ext = ".exe"
        archive_ext = ".zip"
        platform_name = f"windows-{machine}"
    elif system == "darwin":
        ext = ""
        archive_ext = ".tar.gz"
        platform_name = f"macos-{machine}"
    else:
        ext = ""
        archive_ext = ".tar.gz"
        platform_name = f"linux-{machine}"
    
    return platform_name, ext, archive_ext


def create_portable_package(release: bool = True) -> Path:
    """Create the portable package with all necessary files."""
    print("\nCreating portable package...")
    
    platform_name, exe_ext, archive_ext = get_platform_info()
    
    # Find the built executable
    if release:
        build_dir = TAURI_DIR / "target" / "release"
    else:
        build_dir = TAURI_DIR / "target" / "debug"
    
    # Platform-specific executable location
    system = platform.system().lower()
    if system == "windows":
        exe_name = f"AVA{exe_ext}"
        bundle_dir = build_dir / "bundle" / "nsis"
    elif system == "darwin":
        exe_name = "AVA.app"
        bundle_dir = build_dir / "bundle" / "macos"
    else:
        exe_name = "ava"
        bundle_dir = build_dir / "bundle" / "appimage"
    
    exe_path = build_dir / exe_name
    
    if not exe_path.exists():
        print(f"  ✗ Executable not found at {exe_path}")
        # Try bundle directory
        print(f"  Looking in bundle directory: {bundle_dir}")
        return None
    
    # Create distribution directory
    dist_name = f"AVA-{VERSION}-{platform_name}"
    dist_path = DIST_DIR / dist_name
    
    if dist_path.exists():
        shutil.rmtree(dist_path)
    dist_path.mkdir(parents=True)
    
    # Copy executable
    if exe_path.is_dir():  # macOS .app bundle
        shutil.copytree(exe_path, dist_path / exe_name)
    else:
        shutil.copy2(exe_path, dist_path / exe_name)
    print(f"  ✓ Copied executable")
    
    # Copy configuration
    config_src = TAURI_DIR / "config"
    if config_src.exists():
        shutil.copytree(config_src, dist_path / "config")
        print(f"  ✓ Copied configuration")
    
    # Create data directory placeholder
    (dist_path / "data").mkdir(exist_ok=True)
    (dist_path / "data" / ".gitkeep").touch()
    print(f"  ✓ Created data directory")
    
    # Create README
    readme_content = f"""# AVA Neural Interface v{VERSION}

## Quick Start

1. **Install Ollama** (if not already installed):
   - Download from https://ollama.ai/
   - Run `ollama serve` in a terminal

2. **Pull a model**:
   ```
   ollama pull gemma3:4b
   ```

3. **Run AVA**:
   - Windows: Double-click `AVA.exe`
   - macOS: Open `AVA.app`
   - Linux: Run `./ava`

## Configuration

Edit `config/ava.toml` to customize:
- Model settings
- Server port
- Ollama connection

## Portable Mode

This is a portable installation. All data is stored in the `data` folder
next to the executable. You can move this entire folder to another location.

## Troubleshooting

- **"Cannot connect to Ollama"**: Make sure Ollama is running (`ollama serve`)
- **"No models found"**: Pull a model with `ollama pull gemma3:4b`
- **Port in use**: Edit `config/ava.toml` to change the port

## Links

- GitHub: https://github.com/NAME0x0/AVA
- Ollama: https://ollama.ai/
"""
    (dist_path / "README.md").write_text(readme_content)
    print(f"  ✓ Created README")
    
    # Create archive
    archive_path = DIST_DIR / f"{dist_name}{archive_ext}"
    if archive_ext == ".zip":
        shutil.make_archive(str(DIST_DIR / dist_name), "zip", DIST_DIR, dist_name)
    else:
        shutil.make_archive(str(DIST_DIR / dist_name), "gztar", DIST_DIR, dist_name)
    
    print(f"\n✓ Created portable package: {archive_path}")
    
    return archive_path


def main():
    parser = argparse.ArgumentParser(description="Build portable AVA application")
    parser.add_argument("--release", action="store_true", default=True, help="Build release version (default)")
    parser.add_argument("--debug", action="store_true", help="Build debug version")
    parser.add_argument("--skip-frontend", action="store_true", help="Skip frontend build")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    args = parser.parse_args()
    
    release = not args.debug
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           AVA Portable Build Script v{VERSION}                  ║
╠══════════════════════════════════════════════════════════════╣
║  Building: {'Release' if release else 'Debug':<10}                                      ║
║  Platform: {platform.system():<10}                                      ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n✗ Prerequisites check failed. Please install missing tools.")
        sys.exit(1)
    
    # Create dist directory
    DIST_DIR.mkdir(exist_ok=True)
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies():
            print("\n✗ Failed to install dependencies")
            sys.exit(1)
    
    # Build frontend
    if not args.skip_frontend:
        if not build_frontend():
            print("\n✗ Failed to build frontend")
            sys.exit(1)
    
    # Build Tauri
    if not build_tauri(release):
        print("\n✗ Failed to build Tauri application")
        sys.exit(1)
    
    # Create portable package
    archive = create_portable_package(release)
    
    if archive and archive.exists():
        size_mb = archive.stat().st_size / (1024 * 1024)
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                     BUILD SUCCESSFUL                          ║
╠══════════════════════════════════════════════════════════════╣
║  Output: {archive.name:<52} ║
║  Size:   {size_mb:.1f} MB{' '*(48-len(f'{size_mb:.1f} MB'))} ║
╚══════════════════════════════════════════════════════════════╝
""")
    else:
        print("\n✗ Failed to create portable package")
        sys.exit(1)


if __name__ == "__main__":
    main()
