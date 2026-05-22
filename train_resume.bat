@echo off
setlocal EnableDelayedExpansion
REM Resume interrupted training. All-ASCII.

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

REM Background detached run with --resume
start "damda-train" /B python -m src.train --config configs\baseline.yaml --resume >> train_console.log 2>&1

echo.
echo Training resumed in background.
echo   Progress: powershell -Command "Get-Content runs\main\train.log -Tail 5 -Encoding UTF8"
echo   GPU     : nvidia-smi
echo.
endlocal
