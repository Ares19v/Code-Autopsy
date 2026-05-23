@echo off
:: ════════════════════════════════════════════════════════════════════════════
:: Code Autopsy — Run_Project.bat
:: One-click launcher: starts the API server + Gradio demo, opens browser.
:: Prerequisites: Run INSTALL.bat first.
:: ════════════════════════════════════════════════════════════════════════════

title Code Autopsy — Launcher
color 0A
echo.
echo  ╔══════════════════════════════════════════╗
echo  ║        Code Autopsy — Launcher           ║
echo  ╠══════════════════════════════════════════╣
echo  ║  API   → http://localhost:8000           ║
echo  ║  UI    → http://localhost:7860           ║
echo  ║  Docs  → http://localhost:8000/docs      ║
echo  ╚══════════════════════════════════════════╝
echo.

:: ── Check .venv ───────────────────────────────────────────────────────────────
if not exist ".venv\Scripts\python.exe" (
    echo  [ERROR] Virtual environment not found.
    echo  [INFO]  Please run INSTALL.bat first.
    echo.
    pause
    exit /b 1
)

:: ── Check .env ────────────────────────────────────────────────────────────────
if not exist ".env" (
    echo  [WARN]  .env not found — copying from .env.example
    copy ".env.example" ".env" >nul
    echo  [WARN]  Fill in your API keys in .env if needed.
    echo.
)

:: ── Check adapter ─────────────────────────────────────────────────────────────
if not exist "adapter" (
    echo  [WARN]  No adapter found at .\adapter — running base model only.
    echo  [INFO]  Train first with: .venv\Scripts\python.exe training\train.py
    echo.
)

:: ── Start FastAPI server in a new window ─────────────────────────────────────
echo  [INFO]  Starting FastAPI server on port 8000...
start "Code Autopsy — API" cmd /k ".venv\Scripts\python.exe -m uvicorn serve.api:app --host 0.0.0.0 --port 8000"

:: ── Wait for API to be ready (model loading takes ~60s) ──────────────────────
echo  [INFO]  Waiting for API to be ready (model loading takes ~60s)...
echo  [INFO]  Watch the API window for "Model ready."
echo.
timeout /t 10 /nobreak >nul

:: ── Start Gradio demo in a new window ────────────────────────────────────────
echo  [INFO]  Starting Gradio demo on port 7860...
start "Code Autopsy — Demo" cmd /k ".venv\Scripts\python.exe demo\app.py"

:: ── Wait a moment then open browser ─────────────────────────────────────────
timeout /t 5 /nobreak >nul
echo  [INFO]  Opening browser...
start "" "http://localhost:7860"

:: ── Done ─────────────────────────────────────────────────────────────────────
echo.
echo  ══════════════════════════════════════════
echo  [DONE]  Both services are starting up.
echo.
echo  API will be ready when you see:
echo    "Model ready." in the API window.
echo.
echo  Close both terminal windows to stop.
echo  ══════════════════════════════════════════
echo.
pause
