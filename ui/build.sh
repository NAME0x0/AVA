#!/bin/bash
set -e

echo "============================================"
echo "AVA UI Build Script (Unix)"
echo "============================================"
echo

cd "$(dirname "$0")"

echo "[1/4] Checking prerequisites..."

if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js is not installed!"
    echo "Please install from https://nodejs.org/"
    exit 1
fi

SKIP_TAURI=0
if ! command -v cargo &> /dev/null; then
    echo "WARNING: Rust is not installed. Tauri build will be skipped."
    echo "Install from https://rustup.rs/ for desktop builds."
    SKIP_TAURI=1
fi

echo "Node.js: OK"
if [ $SKIP_TAURI -eq 0 ]; then
    echo "Rust: OK"
fi
echo

echo "[2/4] Installing dependencies..."
npm install
echo

echo "[3/4] Building Next.js..."
npm run build
echo

if [ $SKIP_TAURI -eq 1 ]; then
    echo "[4/4] Skipping Tauri build (Rust not installed)"
    echo
    echo "============================================"
    echo "Web build complete!"
    echo "Output: ui/out/"
    echo
    echo "To run: Open out/index.html in a browser"
    echo "============================================"
else
    echo "[4/4] Building Tauri desktop app..."
    npm run tauri:build
    echo
    echo "============================================"
    echo "Build complete!"
    echo
    echo "Web output: ui/out/"
    echo "Desktop app: ui/src-tauri/target/release/"
    echo "Installer: ui/src-tauri/target/release/bundle/"
    echo "============================================"
fi
