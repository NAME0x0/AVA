@echo off
echo ============================================
echo AVA UI Build Script (Windows)
echo ============================================
echo.

cd /d "%~dp0"

echo [1/4] Checking prerequisites...
where node >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Node.js is not installed!
    echo Please install from https://nodejs.org/
    exit /b 1
)

where cargo >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo WARNING: Rust is not installed. Tauri build will be skipped.
    echo Install from https://rustup.rs/ for desktop builds.
    set SKIP_TAURI=1
)

echo Node.js: OK
if not defined SKIP_TAURI echo Rust: OK
echo.

echo [2/4] Installing dependencies...
call npm install
if %ERRORLEVEL% neq 0 (
    echo ERROR: npm install failed!
    exit /b 1
)
echo.

echo [3/4] Building Next.js...
call npm run build
if %ERRORLEVEL% neq 0 (
    echo ERROR: Next.js build failed!
    exit /b 1
)
echo.

if defined SKIP_TAURI (
    echo [4/4] Skipping Tauri build (Rust not installed)
    echo.
    echo ============================================
    echo Web build complete!
    echo Output: ui/out/
    echo.
    echo To run: Open out/index.html in a browser
    echo ============================================
) else (
    echo [4/4] Building Tauri desktop app...
    call npm run tauri:build
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Tauri build failed!
        exit /b 1
    )
    echo.
    echo ============================================
    echo Build complete!
    echo.
    echo Web output: ui/out/
    echo Desktop app: ui/src-tauri/target/release/
    echo Installer: ui/src-tauri/target/release/bundle/
    echo ============================================
)

pause
