@echo off
:: ════════════════════════════════════════════════════════════════════════════
:: Code Autopsy — UNINSTALL.bat
:: Removes the virtual environment and optionally cleans data/weights.
:: ════════════════════════════════════════════════════════════════════════════

title Code Autopsy — Uninstall
color 0C
echo.
echo  ╔══════════════════════════════════════╗
echo  ║       Code Autopsy — Uninstall       ║
echo  ╚══════════════════════════════════════╝
echo.
echo  This will remove the .venv directory.
echo.

:: ── Confirm ───────────────────────────────────────────────────────────────────
set /p CONFIRM="  Remove .venv? (y/N): "
if /i not "%CONFIRM%"=="y" (
    echo  [ABORT] Uninstall cancelled.
    pause
    exit /b 0
)

:: ── Remove venv ───────────────────────────────────────────────────────────────
if exist ".venv" (
    echo  [INFO]  Removing .venv...
    rmdir /s /q ".venv"
    echo  [OK]    .venv removed.
) else (
    echo  [SKIP]  .venv not found.
)

:: ── Optional: remove processed data ──────────────────────────────────────────
echo.
set /p CLEAN_DATA="  Also remove data/processed and checkpoints? (y/N): "
if /i "%CLEAN_DATA%"=="y" (
    if exist "data\processed" (
        rmdir /s /q "data\processed"
        echo  [OK]    data\processed removed.
    )
    if exist "checkpoints" (
        rmdir /s /q "checkpoints"
        echo  [OK]    checkpoints removed.
    )
)

:: ── Done ──────────────────────────────────────────────────────────────────────
echo.
echo  ════════════════════════════════════════
echo  [DONE]  Uninstall complete.
echo  Run "INSTALL.bat" to set up again.
echo  ════════════════════════════════════════
echo.
pause
