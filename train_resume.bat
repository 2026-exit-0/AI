@echo off
setlocal EnableDelayedExpansion
REM Resume interrupted training. Tries venv activation but falls back to current shell's python.

cd /d C:\damda\AI

REM Try to activate a venv if available
REM   Lab PC: C:\damda\.venv  (parent of AI)
REM   Laptop: C:\Users\YSB\...\2026-damda\AI\myvenv
set VENV_OK=0
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    set VENV_OK=1
) else if exist ..\.venv\Scripts\activate.bat (
    call ..\.venv\Scripts\activate.bat
    set VENV_OK=1
) else if exist myvenv\Scripts\activate.bat (
    call myvenv\Scripts\activate.bat
    set VENV_OK=1
) else if exist ..\myvenv\Scripts\activate.bat (
    call ..\myvenv\Scripts\activate.bat
    set VENV_OK=1
) else if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    set VENV_OK=1
)

if "!VENV_OK!"=="0" (
    echo [INFO] No venv activate.bat found, using current shell's python.
)

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] python not found on PATH.
    exit /b 1
)

start "damda-train" /B python -m src.train --config configs\baseline.yaml --resume >> train_console.log 2>&1

echo.
echo Training resumed in background.
echo   Progress: powershell -Command "Get-Content runs\main\train.log -Tail 5 -Encoding UTF8"
echo   GPU     : nvidia-smi
echo.
endlocal
