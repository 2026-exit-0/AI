@echo off
setlocal EnableDelayedExpansion
REM Start new main training (after checkpoint cleanup). All-ASCII, goto-based for cmd parser safety.

cd /d C:\damda\AI

REM Auto-detect venv: .venv (lab PC) or myvenv (laptop) or venv
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else if exist myvenv\Scripts\activate.bat (
    call myvenv\Scripts\activate.bat
) else if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo [ERROR] No venv found ^(.venv / myvenv / venv^).
    exit /b 1
)

REM Safety guard: warn if existing checkpoints would be overwritten
if exist checkpoints\epoch001.pt goto :guard
goto :run

:guard
echo.
echo [WARN] Existing checkpoints found in checkpoints\
echo        Starting new run may overwrite them.
echo        To preserve, first run:
echo            ren runs\main main_vX
echo            rmdir /s /q checkpoints
echo.
set /p ans=Continue anyway [y/N]:
if /i not "!ans!"=="y" exit /b 1

:run
REM Background detached run -- survives SSH disconnect best-effort
start "damda-train" /B python -m src.train --config configs/baseline.yaml > train_console.log 2>&1

echo.
echo Training started in background.
echo   Progress: powershell -Command "Get-Content runs\main\train.log -Tail 5 -Encoding UTF8"
echo   GPU     : nvidia-smi
echo.
endlocal
