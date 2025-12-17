@echo off
title AVA - Neural Interface

echo.
echo  ╔═══════════════════════════════════════════════════════════════════╗
echo  ║                                                                   ║
echo  ║     █████╗ ██╗   ██╗ █████╗                                       ║
echo  ║    ██╔══██╗██║   ██║██╔══██╗                                      ║
echo  ║    ███████║██║   ██║███████║   Adaptive Virtual Agent             ║
echo  ║    ██╔══██║╚██╗ ██╔╝██╔══██║   Neural Interface Launcher          ║
echo  ║    ██║  ██║ ╚████╔╝ ██║  ██║                                      ║
echo  ║    ╚═╝  ╚═╝  ╚═══╝  ╚═╝  ╚═╝                                      ║
echo  ║                                                                   ║
echo  ╚═══════════════════════════════════════════════════════════════════╝
echo.

:: Check for Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed!
    pause
    exit /b 1
)

:: Check for Ollama
echo [1/4] Checking Ollama...
curl -s http://localhost:11434/api/tags >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] Ollama not running. Starting...
    start /min "Ollama" ollama serve
    timeout /t 3 >nul
)

:: Start API server in background
echo [2/4] Starting API server...
start /min "AVA-API" python api_server.py --port 8080
timeout /t 2 >nul

:: Check if UI is built
if not exist "ui\target\release\ava-ui.exe" (
    echo [3/4] UI not built. Building now...
    cd ui
    cargo build --release
    cd ..
    if %ERRORLEVEL% NEQ 0 (
        echo [WARN] UI build failed. Running in terminal mode.
        goto :terminal_mode
    )
)

:: Start UI
echo [4/4] Launching Neural Interface...
start "" "ui\target\release\ava-ui.exe"
echo.
echo UI launched! API running on http://localhost:8080
echo.
echo Press any key to stop the API server...
pause >nul

:: Cleanup
taskkill /fi "WindowTitle eq AVA-API" >nul 2>nul
exit /b 0

:terminal_mode
echo.
echo Running in terminal mode...
python run_frankensystem.py
