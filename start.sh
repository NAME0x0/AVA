#!/bin/bash
# ==========================================================
# AVA - One-Click Setup and Launch
# ==========================================================
# This script sets up and runs AVA automatically.
# Run with: chmod +x start.sh && ./start.sh
# ==========================================================

set -e

echo ""
echo " _____ _    _____  "
echo "|  _  | |  |  _  | "
echo "| || | |  | || | "
echo "|  _  | |  |  _  | "
echo "|_| |_|____|_| |_| "
echo ""
echo "Adaptive Virtual Agent - Setup"
echo "=========================================================="
echo ""

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed!"
    echo ""
    echo "Please install Python 3.10+ from:"
    echo "https://www.python.org/downloads/"
    exit 1
fi

echo "[OK] Python found: $(python3 --version)"

# Check for Ollama
if ! command -v ollama &> /dev/null; then
    echo "[WARNING] Ollama is not installed"
    echo ""
    echo "Please install Ollama from:"
    echo "https://ollama.ai/"
    echo ""
    echo "After installing, run: ollama pull gemma3:4b"
else
    echo "[OK] Ollama found"
fi

# Create venv if needed
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "[OK] Virtual environment created"
fi

# Activate venv
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "[OK] Dependencies installed"

# Create directories
mkdir -p data/conversations config
echo "[OK] Directories ready"

# Check Ollama
echo ""
echo "Checking Ollama..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "[WARNING] Ollama is not running!"
    echo ""
    echo "Starting Ollama in background..."
    ollama serve &
    sleep 3
fi

# Check for models
if ! curl -s http://localhost:11434/api/tags | grep -qE "gemma3|llama3|phi3"; then
    echo "[WARNING] No compatible models found!"
    echo ""
    echo "Downloading gemma3:4b..."
    ollama pull gemma3:4b
fi

echo "[OK] Models ready"

# Start server
echo ""
echo "=========================================================="
echo "Starting AVA..."
echo "=========================================================="
echo ""
echo "Server will be available at: http://localhost:8085"
echo ""
echo "Test with:"
echo "  curl -X POST http://localhost:8085/chat -H 'Content-Type: application/json' -d '{\"message\": \"Hello!\"}'"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python server.py
