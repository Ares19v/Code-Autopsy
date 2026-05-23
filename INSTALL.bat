@echo off
:: ════════════════════════════════════════════════════════════════════════════
:: Code Autopsy — INSTALL.bat
:: Sets up the Python virtual environment and installs all dependencies.
:: Run this ONCE before using Run_Project.bat
:: ════════════════════════════════════════════════════════════════════════════

title Code Autopsy — Install
color 0B
echo.
echo  ╔══════════════════════════════════════╗
echo  ║        Code Autopsy — Install        ║
echo  ╚══════════════════════════════════════╝
echo.

:: ── Check Python ─────────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

:: ── Create virtual environment ────────────────────────────────────────────────
if exist ".venv" (
    echo  [SKIP]  .venv already exists. Delete it manually to reinstall.
) else (
    echo  [INFO]  Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo  [ERROR] Failed to create .venv
        pause
        exit /b 1
    )
    echo  [OK]    Virtual environment created.
)

:: ── Install PyTorch (CUDA 12.x) ───────────────────────────────────────────────
echo.
echo  [INFO]  Installing PyTorch with CUDA 12.4 support...
echo  [INFO]  (This may take a few minutes on first install)
.venv\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/cu124 --quiet
if errorlevel 1 (
    echo  [WARN]  PyTorch CUDA install failed. Trying CPU-only fallback...
    .venv\Scripts\pip.exe install torch --quiet
)
echo  [OK]    PyTorch installed.

:: ── Install requirements ──────────────────────────────────────────────────────
echo.
echo  [INFO]  Installing requirements...
.venv\Scripts\pip.exe install -r requirements.txt --quiet
if errorlevel 1 (
    echo  [ERROR] Failed to install requirements.txt
    pause
    exit /b 1
)
echo  [OK]    Requirements installed.

:: ── Copy .env.example ────────────────────────────────────────────────────────
if not exist ".env" (
    echo.
    echo  [INFO]  Creating .env from .env.example...
    copy ".env.example" ".env" >nul
    echo  [WARN]  Please fill in your API keys in .env before running the project.
) else (
    echo  [SKIP]  .env already exists.
)

:: ── Done ──────────────────────────────────────────────────────────────────────
echo.
echo  ════════════════════════════════════════
echo  [DONE]  Installation complete!
echo  Run "Run_Project.bat" to start the app.
echo  ════════════════════════════════════════
echo.
pause
