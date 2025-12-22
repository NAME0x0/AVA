@echo off
REM ==========================================================
REM AVA - One-Click Setup and Launch
REM ==========================================================
REM This script sets up and runs AVA automatically.
REM Just double-click this file!
REM ==========================================================

REM Set UTF-8 code page for proper character display
chcp 65001 >nul 2>&1

echo.
echo    █████████   █████   █████   █████████
echo   ███░░░░░███ ░░███   ░░███   ███░░░░░███
echo  ░███    ░███  ░███    ░███  ░███    ░███
echo  ░███████████  ░███    ░███  ░███████████
echo  ░███░░░░░███  ░░███   ███   ░███░░░░░███
echo  ░███    ░███   ░░░█████░    ░███    ░███
echo  █████   █████    ░░███      █████   █████
echo ░░░░░   ░░░░░      ░░░      ░░░░░   ░░░░░
echo.
echo ══════════════════════════════════════════════════════
echo        Adaptive Virtual Agent
echo ══════════════════════════════════════════════════════
echo.

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo %ESC%[31m[ERROR]%ESC%[0m Python is not installed!
    echo.
    echo Please install Python 3.10+ from:
    echo %ESC%[36mhttps://www.python.org/downloads/%ESC%[0m
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo %ESC%[32m[OK]%ESC%[0m Python found

REM Check for Ollama
ollama --version >nul 2>&1
if errorlevel 1 (
    echo %ESC%[33m[WARNING]%ESC%[0m Ollama is not installed or not in PATH
    echo.
    echo Please install Ollama from:
    echo %ESC%[36mhttps://ollama.ai/%ESC%[0m
    echo.
    echo After installing, run: %ESC%[36mollama pull gemma3:4b%ESC%[0m
    echo.
) else (
    echo %ESC%[32m[OK]%ESC%[0m Ollama found
)

REM Check if venv exists
if not exist "venv" (
    echo.
    echo %ESC%[36mCreating virtual environment...%ESC%[0m
    python -m venv venv
    echo %ESC%[32m[OK]%ESC%[0m Virtual environment created
)

REM Activate venv
call venv\Scripts\activate.bat

REM Install/upgrade dependencies
echo.
echo %ESC%[36mInstalling dependencies...%ESC%[0m
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo %ESC%[32m[OK]%ESC%[0m Dependencies installed

REM Create data directories
if not exist "data" mkdir data
if not exist "data\conversations" mkdir data\conversations
if not exist "config" mkdir config
echo %ESC%[32m[OK]%ESC%[0m Directories ready

REM Check if Ollama is running
echo.
echo %ESC%[36mChecking Ollama...%ESC%[0m
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo %ESC%[33m[WARNING]%ESC%[0m Ollama is not running!
    echo.
    echo %ESC%[36mStarting Ollama in background...%ESC%[0m
    start /B ollama serve
    timeout /t 3 /nobreak >nul
)

REM Check for models
echo %ESC%[36mChecking for models...%ESC%[0m
curl -s http://localhost:11434/api/tags | findstr "gemma3\|llama3\|phi3" >nul 2>&1
if errorlevel 1 (
    echo %ESC%[33m[WARNING]%ESC%[0m No compatible models found!
    echo.
    echo %ESC%[36mDownloading gemma3:4b (this may take a few minutes)...%ESC%[0m
    ollama pull gemma3:4b
)

echo %ESC%[32m[OK]%ESC%[0m Models ready

REM Start the server
echo.
echo %ESC%[38;5;245m══════════════════════════════════════════════════════%ESC%[0m
echo %ESC%[1;38;5;51m  Starting AVA...%ESC%[0m
echo %ESC%[38;5;245m══════════════════════════════════════════════════════%ESC%[0m
echo.
echo Server will be available at: %ESC%[36mhttp://localhost:8085%ESC%[0m
echo.
echo Test with:
echo   %ESC%[90mcurl -X POST http://localhost:8085/chat -H "Content-Type: application/json" -d "{\"message\": \"Hello!\"}"%ESC%[0m
echo.
echo %ESC%[90mPress Ctrl+C to stop%ESC%[0m
echo.

python server.py

REM Deactivate venv on exit
deactivate
