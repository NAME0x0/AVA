#!/bin/bash
# ==========================================================
# AVA v3 One-Click Setup for Linux/macOS
# ==========================================================
# Run with: chmod +x start.sh && ./start.sh
# Or: bash start.sh [--minimal|--full|--models|--check]
# ==========================================================

set -e

# Parse arguments
MODEL_MODE="standard"
MODELS_ONLY=false
CHECK_ONLY=false
SKIP_MODELS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --minimal) MODEL_MODE="minimal"; shift ;;
        --full) MODEL_MODE="full"; shift ;;
        --models) MODELS_ONLY=true; shift ;;
        --check) CHECK_ONLY=true; shift ;;
        --skip-models) SKIP_MODELS=true; shift ;;
        *) shift ;;
    esac
done

# Model configurations
declare -A MODELS
MODELS[minimal]="gemma3:4b nomic-embed-text"
MODELS[standard]="gemma3:4b llama3.2:latest nomic-embed-text"
MODELS[full]="gemma3:4b llama3.2:latest llama3.1:70b-instruct-q4_0 nomic-embed-text"

# Banner
echo ""
echo "================================================================================"
echo "     ___ _    ___   _    ____  _____ _____ _   _ ____"
echo "    / _ \ \  / / \ | |  / ___|| ____|_   _| | | |  _ \\"
echo "   | |_| \ \/ / _ \| | | \___ |  _|   | | | | | | |_) |"
echo "   |  _  ||  / ___ \ |  ___) | |___  | | | |_| |  __/"
echo "   |_| |_||_/_/   \_\_| |____/|_____| |_|  \___/|_|"
echo ""
echo "    Cortex-Medulla Architecture v3"
echo "    One-Click Setup for Linux/macOS"
echo "================================================================================"
echo ""

# ==========================================================
# Step 1: Check Prerequisites
# ==========================================================
echo "[1/7] Checking Prerequisites"
echo "------------------------------------------------------------"

# Check Python
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 --version)
    echo "  [OK] $PY_VERSION"
else
    echo "  [ERROR] Python 3 is not installed!"
    echo "  Please install Python 3.9+ from: https://www.python.org/downloads/"
    exit 1
fi

# Check Ollama
OLLAMA_INSTALLED=false
if command -v ollama &> /dev/null; then
    echo "  [OK] Ollama installed"
    OLLAMA_INSTALLED=true
else
    echo "  [WARNING] Ollama not found"
    echo "  Install from: https://ollama.ai/"
fi

# Check NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "")
    if [ -n "$GPU_INFO" ]; then
        echo "  [OK] NVIDIA GPU: $GPU_INFO"
    else
        echo "  [WARNING] No NVIDIA GPU detected (CPU-only mode)"
    fi
else
    echo "  [WARNING] nvidia-smi not found (CPU-only mode)"
fi

if [ "$CHECK_ONLY" = true ]; then
    echo ""
    echo "Validation complete."
    exit 0
fi

if [ "$MODELS_ONLY" = true ]; then
    # Skip to model download section
    SKIP_ENV=true
else
    SKIP_ENV=false
fi

# ==========================================================
# Step 2: Create Virtual Environment
# ==========================================================
if [ "$SKIP_ENV" = false ]; then
    echo ""
    echo "[2/7] Creating Virtual Environment"
    echo "------------------------------------------------------------"

    if [ -d "venv" ]; then
        echo "  [OK] Virtual environment already exists"
    else
        python3 -m venv venv
        echo "  [OK] Virtual environment created"
    fi

    # ==========================================================
    # Step 3: Install Dependencies
    # ==========================================================
    echo ""
    echo "[3/7] Installing Dependencies"
    echo "------------------------------------------------------------"

    source venv/bin/activate

    echo "  Upgrading pip..."
    pip install --upgrade pip -q

    if [ -f "requirements.txt" ]; then
        echo "  Installing from requirements.txt..."
        pip install -r requirements.txt -q
        echo "  [OK] Core dependencies installed"
    fi

    # Optional packages
    for pkg in pynvml psutil aiofiles; do
        pip install $pkg -q 2>/dev/null && echo "  [OK] Installed $pkg" || echo "  [WARNING] Could not install $pkg"
    done

    # ==========================================================
    # Step 4: Create Directories
    # ==========================================================
    echo ""
    echo "[4/7] Creating Directories"
    echo "------------------------------------------------------------"

    DIRECTORIES=(
        "data/memory/episodic"
        "data/memory"
        "data/learning/samples"
        "data/learning/checkpoints"
        "data/conversations"
        "models/fine_tuned_adapters/bridge"
        "models/fine_tuned_adapters/experts"
        "config"
        "logs"
    )

    for dir in "${DIRECTORIES[@]}"; do
        mkdir -p "$dir"
        echo "  [OK] $dir"
    done
fi

# ==========================================================
# Step 5: Start Ollama
# ==========================================================
echo ""
echo "[5/7] Starting Ollama Service"
echo "------------------------------------------------------------"

if [ "$OLLAMA_INSTALLED" = false ]; then
    echo "  [SKIP] Ollama not installed"
else
    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "  [OK] Ollama is already running"
    else
        echo "  Starting Ollama in background..."
        ollama serve &>/dev/null &
        sleep 3
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "  [OK] Ollama started"
        else
            echo "  [WARNING] Could not start Ollama - please start manually: ollama serve"
        fi
    fi
fi

# ==========================================================
# Step 6: Download Ollama Models
# ==========================================================
if [ "$SKIP_MODELS" = false ] && [ "$OLLAMA_INSTALLED" = true ]; then
    echo ""
    echo "[6/7] Downloading Ollama Models ($MODEL_MODE mode)"
    echo "------------------------------------------------------------"

    for model in ${MODELS[$MODEL_MODE]}; do
        echo "  Downloading $model..."
        if ollama pull "$model" 2>/dev/null; then
            echo "  [OK] Downloaded $model"
        else
            echo "  [WARNING] Failed to download $model"
        fi
    done
else
    echo ""
    echo "[6/7] Skipping Model Downloads"
fi

# ==========================================================
# Step 7: Validate Installation
# ==========================================================
echo ""
echo "[7/7] Validating Installation"
echo "------------------------------------------------------------"

# Check venv
if [ -f "venv/bin/python" ]; then
    echo "  [OK] Virtual environment"
fi

# Check config
if [ -f "config/cortex_medulla.yaml" ]; then
    echo "  [OK] Configuration file"
else
    echo "  [WARNING] Configuration file missing"
fi

# Check models
if [ "$OLLAMA_INSTALLED" = true ]; then
    MODEL_COUNT=$(ollama list 2>/dev/null | tail -n +2 | wc -l)
    echo "  [OK] Ollama models: $MODEL_COUNT installed"
fi

# ==========================================================
# Complete!
# ==========================================================
echo ""
echo "================================================================================"
echo "SETUP COMPLETE!"
echo "================================================================================"
echo ""
echo "Next Steps:"
echo "-----------"
echo ""
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Start the AVA server:"
echo "   python server.py"
echo ""
echo "3. Or run the core system:"
echo "   python run_core.py"
echo ""
echo "4. Test with curl or browser:"
echo "   curl http://localhost:8085/status"
echo ""
echo "Configuration:"
echo "  config/cortex_medulla.yaml"
echo ""
