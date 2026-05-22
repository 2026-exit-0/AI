@echo off
setlocal EnableDelayedExpansion
REM Start new main training. Tries venv activation but falls back to current shell's python.

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
    echo [INFO] No venv activate.bat found in .venv / myvenv / venv.
    echo        Using current shell's python instead.
)

REM Verify python is callable
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] python not found on PATH. Activate your venv manually first.
    exit /b 1
)

echo Using python:
where python
python --version
echo.

REM Safety guard: warn if existing checkpoints would be overwritten
if exist checkpoints\epoch001.pt goto :guard
goto :run

:guard
echo [WARN] Existing checkpoints found in checkpoints\
echo        Starting new run may overwrite them.
echo        To preserve: ren runs\main main_vX  AND  rmdir /s /q checkpoints
echo.
set /p ans=Continue anyway [y/N]:
if /i not "!ans!"=="y" exit /b 1

:run
start "damda-train" /B python -m src.train --config configs/baseline.yaml > train_console.log 2>&1

echo.
echo Training started in background.
echo   Progress: powershell -Command "Get-Content runs\main\train.log -Tail 5 -Encoding UTF8"
echo   GPU     : nvidia-smi
echo.
endlocal
