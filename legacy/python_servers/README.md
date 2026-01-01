# Legacy Python Servers

This folder contains the original Python-based server implementations that have been replaced by the Rust-native backend embedded in the Tauri application.

## Files

- `server.py` - Original HTTP API server
- `ava_server.py` - Standalone backend server  
- `run_core.py` - Cortex-Medulla architecture runner
- `run_tui.py` - Terminal UI launcher
- `*.spec` - PyInstaller specification files
- `start.bat`, `start.sh` - Startup scripts
- `setup_ava.py`, `setup_ava.ps1` - Setup scripts

## Migration

As of v3.4.0, AVA uses a fully Rust-native backend embedded in the Tauri desktop application. This provides:

- **Single portable executable** - No Python runtime required
- **Better performance** - Native Rust HTTP server (Axum)
- **Simpler deployment** - No separate server process to manage
- **Cross-platform** - Builds for Windows, macOS, and Linux

## If You Need Python Backend

If you need to use the Python server for development or compatibility:

1. Ensure Python 3.10+ is installed
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python ava_server.py --port 8085`

The Tauri app can connect to an external Python server by setting the backend URL.
