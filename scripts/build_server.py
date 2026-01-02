#!/usr/bin/env python3
"""
Build AVA Server - Standalone Executable

This script builds the AVA server as a standalone executable using PyInstaller.
Unlike build_sidecar.py, this creates an independent application that:
- Shows a console window with verbose logging
- Can be distributed separately from the UI
- Includes version information in the executable

Usage:
    python scripts/build_server.py
    python scripts/build_server.py --clean
    python scripts/build_server.py --one-dir    # Directory mode (faster build)
    python scripts/build_server.py --version    # Show version info

Requirements:
    pip install pyinstaller

Output:
    dist/ava-server.exe (Windows)
    dist/ava-server (Linux/macOS)
"""

import argparse
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"
SPEC_FILE = PROJECT_ROOT / "ava_server.spec"

# Version from VERSION file (single source of truth)
def get_version() -> str:
    """Read version from VERSION file."""
    version_file = PROJECT_ROOT / "VERSION"
    if version_file.exists():
        return version_file.read_text().strip()
    return "4.0.0"

VERSION = get_version()


def get_exe_extension() -> str:
    """Get the executable extension for the current platform."""
    return ".exe" if platform.system() == "Windows" else ""


def get_platform_name() -> str:
    """Get a human-readable platform name."""
    system = platform.system()
    machine = platform.machine()
    return f"{system}-{machine}"


def clean_build() -> None:
    """Clean previous build artifacts."""
    print("Cleaning previous build artifacts...")

    dirs_to_clean = [BUILD_DIR, DIST_DIR]

    for d in dirs_to_clean:
        if d.exists():
            shutil.rmtree(d)
            print(f"  Removed: {d}")

    # Clean .spec file if it exists
    if SPEC_FILE.exists():
        SPEC_FILE.unlink()
        print(f"  Removed: {SPEC_FILE}")


def check_dependencies() -> bool:
    """Check that required build dependencies are available."""
    print("Checking build dependencies...")

    # Check PyInstaller
    try:
        result = subprocess.run(
            [sys.executable, "-m", "PyInstaller", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"  PyInstaller: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ERROR: PyInstaller not found")
        print("  Install with: pip install pyinstaller")
        return False

    # Check entry point exists
    entry_point = PROJECT_ROOT / "legacy" / "python_servers" / "ava_server.py"
    if not entry_point.exists():
        print(f"  ERROR: Entry point not found: {entry_point}")
        return False
    print(f"  Entry point: {entry_point}")

    # Check config exists
    config_dir = PROJECT_ROOT / "config"
    if not config_dir.exists():
        print(f"  WARNING: Config directory not found: {config_dir}")
    else:
        print(f"  Config dir: {config_dir}")

    return True


def generate_spec_file(one_dir: bool = False) -> Path:
    """Generate a PyInstaller .spec file for the build."""

    print("\nGenerating PyInstaller spec file...")

    # Determine icon path
    icon_path = PROJECT_ROOT / "ui" / "src-tauri" / "icons" / "icon.ico"
    icon_line = f"    icon=r'{icon_path}'," if icon_path.exists() else ""

    # Build EXE args based on mode
    if one_dir:
        exe_data_args = ""
        collect_section = """
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ava-server',
)
"""
    else:
        exe_data_args = """    a.binaries,
    a.zipfiles,
    a.datas,"""
        collect_section = ""

    spec_content = f'''# -*- mode: python ; coding: utf-8 -*-
"""
AVA Server - PyInstaller Spec
Generated: {datetime.now().isoformat()}
Version: {VERSION}
Platform: {get_platform_name()}
Mode: {"one-dir" if one_dir else "one-file"}
"""

import sys
from pathlib import Path

block_cipher = None
PROJECT_ROOT = Path(r'{PROJECT_ROOT}')

# Hidden imports for dynamic loading
hidden_imports = [
    # Core AVA modules
    'ava', 'ava.engine', 'ava.config', 'ava.memory', 'ava.tools', 'ava.cli',
    # Core architecture
    'core', 'core.system', 'core.medulla', 'core.cortex', 'core.bridge', 'core.agency',
    # Cortex modules
    'cortex', 'cortex.ollama_interface', 'cortex.stream',
    # Hippocampus (memory)
    'hippocampus', 'hippocampus.buffer', 'hippocampus.titans',
    # Tools
    'tools', 'tools.base_tools', 'tools.registry',
    # Dependencies
    'aiohttp', 'aiohttp.web', 'httpx', 'yaml', 'pydantic', 'numpy', 'rich',
    'asyncio', 'multidict', 'yarl', 'aiosignal', 'frozenlist', 'certifi',
    'anyio', 'sniffio', 'h11', 'httpcore', 'aiofiles',
]

# Data files to include
datas = [
    (str(PROJECT_ROOT / 'config'), 'config'),
    (str(PROJECT_ROOT / 'src'), 'src'),
    (str(PROJECT_ROOT / 'server.py'), '.'),  # Include server.py for imports
]

# Exclude large unused packages
excludes = [
    'tkinter', 'matplotlib', 'jupyter', 'notebook', 'IPython',
    'PIL', 'cv2', 'tensorflow', 'keras', 'torch', 'transformers',
    'scipy', 'sklearn', 'test', 'tests', 'pytest', 'textual',
]

a = Analysis(
    [str(PROJECT_ROOT / 'legacy' / 'python_servers' / 'ava_server.py')],
    pathex=[str(PROJECT_ROOT), str(PROJECT_ROOT / 'src'), str(PROJECT_ROOT / 'legacy' / 'python_servers')],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filter out test files and __pycache__
a.datas = [
    (name, path, type_)
    for name, path, type_ in a.datas
    if '__pycache__' not in path and 'test' not in name.lower()
]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
{exe_data_args}
    [],
    name='ava-server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
{icon_line}
)
{collect_section}
'''

    SPEC_FILE.write_text(spec_content)
    print(f"  Generated: {SPEC_FILE}")

    return SPEC_FILE


def build_server(clean: bool = False, one_dir: bool = False) -> Path:
    """Build the PyInstaller executable."""

    if clean:
        clean_build()

    print(f"\n{'=' * 60}")
    print(f"Building AVA Server v{VERSION}")
    print(f"Platform: {get_platform_name()}")
    print(f"Mode: {'one-dir' if one_dir else 'one-file'}")
    print(f"{'=' * 60}\n")

    if not check_dependencies():
        sys.exit(1)

    # Generate spec file
    spec_file = generate_spec_file(one_dir=one_dir)

    # Run PyInstaller
    print("\nRunning PyInstaller...\n")
    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--clean", "--noconfirm", str(spec_file)],
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        print("\nERROR: PyInstaller build failed!")
        sys.exit(1)

    # Get the built executable
    ext = get_exe_extension()
    if one_dir:
        src_exe = DIST_DIR / "ava-server" / f"ava-server{ext}"
    else:
        src_exe = DIST_DIR / f"ava-server{ext}"

    if not src_exe.exists():
        print(f"\nERROR: Built executable not found: {src_exe}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("Build Successful!")
    print(f"{'=' * 60}")
    print(f"\nOutput: {src_exe}")
    print(f"Size: {src_exe.stat().st_size / (1024 * 1024):.1f} MB")

    return src_exe


def show_version_info() -> None:
    """Show version and build info."""
    print(f"AVA Server Build Script")
    print(f"  Version: {VERSION}")
    print(f"  Platform: {get_platform_name()}")
    print(f"  Python: {sys.version}")
    print(f"  Project: {PROJECT_ROOT}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build AVA Server as a standalone executable",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/build_server.py              # Build one-file executable
    python scripts/build_server.py --one-dir    # Build directory (faster)
    python scripts/build_server.py --clean      # Clean build
        """,
    )
    parser.add_argument(
        "--clean", action="store_true", help="Clean build artifacts before building"
    )
    parser.add_argument(
        "--one-dir",
        action="store_true",
        help="Build as directory instead of single file (faster build, easier debugging)",
    )
    parser.add_argument(
        "--version", action="store_true", help="Show version info and exit"
    )
    args = parser.parse_args()

    if args.version:
        show_version_info()
        return

    exe_path = build_server(clean=args.clean, one_dir=args.one_dir)

    print(f"\nNext steps:")
    print(f"  1. Test: {exe_path} --check")
    print(f"  2. Run: {exe_path} --port 8085")
    print(f"  3. Or run: {exe_path} --verbose")


if __name__ == "__main__":
    main()
