@echo off
echo ============================================
echo AVA Cortex-Medulla System Launcher
echo ============================================
echo.

cd /d "%~dp0"

echo Starting AVA API Server...
start "AVA Backend" cmd /c "python api_server_v3.py"
echo Backend starting on http://localhost:8085
echo.

timeout /t 3 /nobreak >nul

echo Starting AVA UI...
cd ui
if exist "node_modules" (
    echo Running Next.js development server...
    start "AVA UI" cmd /c "npm run dev"
    echo.
    echo ============================================
    echo AVA is starting!
    echo.
    echo Backend: http://localhost:8085
    echo UI:      http://localhost:3000
    echo.
    echo Press any key to open the UI in your browser...
    echo ============================================
    pause >nul
    start http://localhost:3000
) else (
    echo.
    echo ERROR: UI dependencies not installed!
    echo.
    echo Please run the following first:
    echo   cd ui
    echo   npm install
    echo.
    pause
)
