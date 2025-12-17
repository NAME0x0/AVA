#!/bin/bash
echo "Building AVA UI..."
echo

cd "$(dirname "$0")"

# Check for Rust
if ! command -v cargo &> /dev/null; then
    echo "[ERROR] Rust is not installed!"
    echo "Install from: https://rustup.rs"
    exit 1
fi

# Build release
echo "[1/2] Compiling release build..."
cargo build --release

if [ $? -ne 0 ]; then
    echo "[ERROR] Build failed!"
    exit 1
fi

echo "[2/2] Build complete!"
echo
echo "Binary location: target/release/ava-ui"
echo "File size: $(du -h target/release/ava-ui | cut -f1)"

echo
echo "To run: ./target/release/ava-ui"
