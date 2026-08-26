@echo off
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
    echo.
)

set "PYTHON_CMD=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_CMD=.venv\Scripts\python.exe"

:: ── Start FastAPI server in a new window ─────────────────────────────────────
echo  [INFO]  Starting FastAPI server on port 8000...
start "Code Autopsy — API" cmd /k "%PYTHON_CMD% -m uvicorn serve.api:app --host 0.0.0.0 --port 8000"

:: ── Wait for API to be ready ─────────────────────────────────────────────────
echo  [INFO]  Waiting for API to be ready...
timeout /t 6 /nobreak >nul

:: ── Start React frontend in a new window ────────────────────────────────────────
echo  [INFO]  Starting React frontend on port 5173...
start "Code Autopsy — Frontend" cmd /k "cd frontend && npm run dev"

:: ── Wait a moment then open browser ─────────────────────────────────────────
timeout /t 5 /nobreak >nul
echo  [INFO]  Opening browser...
start "" "http://localhost:5173"

echo.
echo  ══════════════════════════════════════════
echo  [DONE]  Both services are starting up.
echo  Close both terminal windows to stop.
echo  ══════════════════════════════════════════
echo.
pause
