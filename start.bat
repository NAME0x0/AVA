@echo off
REM ==========================================================
REM AVA - One-Click Setup and Launch
REM ==========================================================
REM This script sets up and runs AVA automatically.
REM Just double-click this file!
REM ==========================================================

echo.
echo  _____ _    _____  
echo ^|  _  ^| ^|  ^|  _  ^| 
echo ^| ^|^| ^| ^|  ^| ^|^| ^| 
echo ^|  _  ^| ^|  ^|  _  ^| 
echo ^|_^| ^|_^|____^|_^| ^|_^| 
echo.
echo Adaptive Virtual Agent - Setup
echo ==========================================================
echo.

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed!
    echo.
    echo Please install Python 3.10+ from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo [OK] Python found

REM Check for Ollama
ollama --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Ollama is not installed or not in PATH
    echo.
    echo Please install Ollama from:
    echo https://ollama.ai/
    echo.
    echo After installing, run: ollama pull gemma3:4b
    echo.
) else (
    echo [OK] Ollama found
)

REM Check if venv exists
if not exist "venv" (
    echo.
    echo Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created
)

REM Activate venv
call venv\Scripts\activate.bat

REM Install/upgrade dependencies
echo.
echo Installing dependencies...
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo [OK] Dependencies installed

REM Create data directories
if not exist "data" mkdir data
if not exist "data\conversations" mkdir data\conversations
if not exist "config" mkdir config
echo [OK] Directories ready

REM Check if Ollama is running
echo.
echo Checking Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Ollama is not running!
    echo.
    echo Starting Ollama in background...
    start /B ollama serve
    timeout /t 3 /nobreak >nul
)

REM Check for models
echo Checking for models...
curl -s http://localhost:11434/api/tags | findstr "gemma3\|llama3\|phi3" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] No compatible models found!
    echo.
    echo Downloading gemma3:4b (this may take a few minutes)...
    ollama pull gemma3:4b
)

echo [OK] Models ready

REM Start the server
echo.
echo ==========================================================
echo Starting AVA...
echo ==========================================================
echo.
echo Server will be available at: http://localhost:8085
echo.
echo Test with:
echo   curl -X POST http://localhost:8085/chat -H "Content-Type: application/json" -d "{\"message\": \"Hello!\"}"
echo.
echo Press Ctrl+C to stop
echo.

python server.py

REM Deactivate venv on exit
deactivate
