@echo off
echo Building AVA UI...
echo.

cd /d "%~dp0"

:: Check for Rust
where cargo >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Rust is not installed!
    echo Install from: https://rustup.rs
    pause
    exit /b 1
)

:: Build release
echo [1/2] Compiling release build...
cargo build --release

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Build failed!
    pause
    exit /b 1
)

echo [2/2] Build complete!
echo.
echo Binary location: target\release\ava-ui.exe
echo File size:
for %%I in (target\release\ava-ui.exe) do echo   %%~zI bytes

echo.
echo To run: .\target\release\ava-ui.exe
pause
