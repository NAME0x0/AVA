# ==========================================================
# AVA v3 One-Click Setup for Windows
# ==========================================================
# Run with: .\setup_ava.ps1
# Or: powershell -ExecutionPolicy Bypass -File setup_ava.ps1
# ==========================================================

param(
    [switch]$Minimal,
    [switch]$Full,
    [switch]$ModelsOnly,
    [switch]$SkipModels,
    [switch]$Check
)

$ErrorActionPreference = "Continue"

# Banner
Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "     ___ _    ___   _    ____  _____ _____ _   _ ____" -ForegroundColor Cyan
Write-Host "    / _ \ \  / / \ | |  / ___|| ____|_   _| | | |  _ \" -ForegroundColor Cyan
Write-Host "   | |_| \ \/ / _ \| | | \___ |  _|   | | | | | | |_) |" -ForegroundColor Cyan
Write-Host "   |  _  ||  / ___ \ |  ___) | |___  | | | |_| |  __/" -ForegroundColor Cyan
Write-Host "   |_| |_||_/_/   \_\_| |____/|_____| |_|  \___/|_|" -ForegroundColor Cyan
Write-Host "" -ForegroundColor Cyan
Write-Host "    Cortex-Medulla Architecture v3" -ForegroundColor White
Write-Host "    One-Click Setup for Windows" -ForegroundColor White
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Determine model mode
if ($Minimal) {
    $ModelMode = "minimal"
    $Models = @("gemma3:4b", "nomic-embed-text")
} elseif ($Full) {
    $ModelMode = "full"
    $Models = @("gemma3:4b", "llama3.2:latest", "llama3.1:70b-instruct-q4_0", "nomic-embed-text")
} else {
    $ModelMode = "standard"
    $Models = @("gemma3:4b", "llama3.2:latest", "nomic-embed-text")
}

# ==========================================================
# Step 1: Check Prerequisites
# ==========================================================
Write-Host "[1/7] Checking Prerequisites" -ForegroundColor Yellow
Write-Host ("-" * 60)

# Check Python
$PythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] $PythonVersion" -ForegroundColor Green
} else {
    Write-Host "  [ERROR] Python not found!" -ForegroundColor Red
    Write-Host "  Please install Python 3.9+ from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

# Check Ollama
$OllamaExists = Get-Command ollama -ErrorAction SilentlyContinue
if ($OllamaExists) {
    Write-Host "  [OK] Ollama installed" -ForegroundColor Green
} else {
    Write-Host "  [WARNING] Ollama not found" -ForegroundColor Yellow
    Write-Host "  Install from: https://ollama.ai/" -ForegroundColor Yellow
}

# Check NVIDIA GPU
try {
    $GPU = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] NVIDIA GPU: $GPU" -ForegroundColor Green
    } else {
        Write-Host "  [WARNING] No NVIDIA GPU detected (CPU-only mode)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  [WARNING] nvidia-smi not found (CPU-only mode)" -ForegroundColor Yellow
}

if ($Check) {
    Write-Host "`nValidation complete." -ForegroundColor Green
    exit 0
}

if ($ModelsOnly) {
    # Skip to model download
    $SkipEnv = $true
} else {
    $SkipEnv = $false
}

# ==========================================================
# Step 2: Create Virtual Environment
# ==========================================================
if (-not $SkipEnv) {
    Write-Host "`n[2/7] Creating Virtual Environment" -ForegroundColor Yellow
    Write-Host ("-" * 60)

    if (Test-Path "venv") {
        Write-Host "  [OK] Virtual environment already exists" -ForegroundColor Green
    } else {
        python -m venv venv
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  [OK] Virtual environment created" -ForegroundColor Green
        } else {
            Write-Host "  [ERROR] Failed to create venv" -ForegroundColor Red
            exit 1
        }
    }

    # ==========================================================
    # Step 3: Install Dependencies
    # ==========================================================
    Write-Host "`n[3/7] Installing Dependencies" -ForegroundColor Yellow
    Write-Host ("-" * 60)

    # Activate venv and install
    & .\venv\Scripts\pip.exe install --upgrade pip 2>&1 | Out-Null
    Write-Host "  Upgrading pip..." -ForegroundColor Gray

    if (Test-Path "requirements.txt") {
        Write-Host "  Installing from requirements.txt..." -ForegroundColor Gray
        & .\venv\Scripts\pip.exe install -r requirements.txt 2>&1 | Out-Null
        Write-Host "  [OK] Core dependencies installed" -ForegroundColor Green
    }

    # Install optional packages
    $OptionalPackages = @("pynvml", "psutil", "aiofiles")
    foreach ($pkg in $OptionalPackages) {
        & .\venv\Scripts\pip.exe install $pkg 2>&1 | Out-Null
        Write-Host "  [OK] Installed $pkg" -ForegroundColor Green
    }

    # ==========================================================
    # Step 4: Create Directories
    # ==========================================================
    Write-Host "`n[4/7] Creating Directories" -ForegroundColor Yellow
    Write-Host ("-" * 60)

    $Directories = @(
        "data\memory\episodic",
        "data\memory",
        "data\learning\samples",
        "data\learning\checkpoints",
        "data\conversations",
        "models\fine_tuned_adapters\bridge",
        "models\fine_tuned_adapters\experts",
        "config",
        "logs"
    )

    foreach ($dir in $Directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
        Write-Host "  [OK] $dir" -ForegroundColor Green
    }
}

# ==========================================================
# Step 5: Start Ollama
# ==========================================================
Write-Host "`n[5/7] Starting Ollama Service" -ForegroundColor Yellow
Write-Host ("-" * 60)

if (-not $OllamaExists) {
    Write-Host "  [SKIP] Ollama not installed" -ForegroundColor Yellow
} else {
    # Check if Ollama is running
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -ErrorAction Stop
        Write-Host "  [OK] Ollama is already running" -ForegroundColor Green
    } catch {
        Write-Host "  Please start Ollama manually:" -ForegroundColor Yellow
        Write-Host "    1. Open a new terminal" -ForegroundColor White
        Write-Host "    2. Run: ollama serve" -ForegroundColor White
        Write-Host "  Waiting 5 seconds..." -ForegroundColor Gray
        Start-Sleep -Seconds 5
    }
}

# ==========================================================
# Step 6: Download Ollama Models
# ==========================================================
if (-not $SkipModels -and $OllamaExists) {
    Write-Host "`n[6/7] Downloading Ollama Models ($ModelMode mode)" -ForegroundColor Yellow
    Write-Host ("-" * 60)

    foreach ($model in $Models) {
        Write-Host "  Downloading $model..." -ForegroundColor Gray
        ollama pull $model 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  [OK] Downloaded $model" -ForegroundColor Green
        } else {
            Write-Host "  [WARNING] Failed to download $model" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "`n[6/7] Skipping Model Downloads" -ForegroundColor Yellow
}

# ==========================================================
# Step 7: Validate Installation
# ==========================================================
Write-Host "`n[7/7] Validating Installation" -ForegroundColor Yellow
Write-Host ("-" * 60)

# Check venv
if (Test-Path "venv\Scripts\python.exe") {
    Write-Host "  [OK] Virtual environment" -ForegroundColor Green
}

# Check config
if (Test-Path "config\cortex_medulla.yaml") {
    Write-Host "  [OK] Configuration file" -ForegroundColor Green
} else {
    Write-Host "  [WARNING] Configuration file missing" -ForegroundColor Yellow
}

# Check models
if ($OllamaExists) {
    $modelList = ollama list 2>&1
    if ($LASTEXITCODE -eq 0) {
        $modelCount = ($modelList | Measure-Object -Line).Lines - 1
        Write-Host "  [OK] Ollama models: $modelCount installed" -ForegroundColor Green
    }
}

# ==========================================================
# Complete!
# ==========================================================
Write-Host ""
Write-Host "================================================================================" -ForegroundColor Green
Write-Host "SETUP COMPLETE!" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "-----------" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Activate the virtual environment:" -ForegroundColor White
Write-Host "   .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Start the AVA server:" -ForegroundColor White
Write-Host "   python server.py" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Or run the core system:" -ForegroundColor White
Write-Host "   python run_core.py" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Test with curl or browser:" -ForegroundColor White
Write-Host "   curl http://localhost:8085/status" -ForegroundColor Gray
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  config\cortex_medulla.yaml" -ForegroundColor Gray
Write-Host ""
