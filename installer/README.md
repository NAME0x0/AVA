# AVA Installer

This directory contains the build scripts and configuration for creating AVA installers.

## Directory Structure

```
installer/
├── config/
│   └── installer.yaml     # Installer configuration
├── nsis/
│   ├── ava-installer.nsi  # NSIS installer script (to be created)
│   ├── license.txt        # License for installer
│   └── assets/            # Installer graphics
├── scripts/
│   └── build_installer.py # Build automation script
├── build/                 # Temporary build files (gitignored)
└── dist/                  # Output installers (gitignored)
```

## Prerequisites

Before building the installer, ensure you have:

1. **NSIS** - Nullsoft Scriptable Install System
   - Download: https://nsis.sourceforge.io/
   - Add to PATH after installation

2. **Rust + Cargo** - For building Tauri app
   - Download: https://rustup.rs/

3. **Node.js 18+** - For building the frontend
   - Download: https://nodejs.org/

4. **Python 3.10+** - For build scripts
   - Download: https://python.org/

## Building the Installer

### Quick Build

```bash
cd installer/scripts
python build_installer.py
```

### Build Options

```bash
# Build NSIS installer only
python build_installer.py

# Build portable ZIP
python build_installer.py --portable

# Build all variants
python build_installer.py --all

# Skip Tauri build (use existing)
python build_installer.py --skip-tauri
```

### Output

- `dist/AVA-{version}-Setup.exe` - Windows installer
- `dist/AVA-{version}-portable.zip` - Portable version

## Configuration

Edit `config/installer.yaml` to customize:

- Installation directory
- Component selection (GUI, TUI, CLI)
- Startup behavior
- Shortcuts and registry entries

## Development

To test the installer locally:

1. Build the Tauri app: `cd ui && npm run tauri build`
2. Build installer: `python installer/scripts/build_installer.py`
3. Run the installer from `dist/`

## Notes

- The installer downloads Python if not present on the system
- Ollama is not bundled; users are prompted to install it
- All user data is stored in `%APPDATA%\AVA`
