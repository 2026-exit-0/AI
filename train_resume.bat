@echo off
REM Resume interrupted training. All-ASCII for cmd safety.

cd /d C:\damda\AI
call myvenv\Scripts\activate.bat

REM Background detached run with --resume
start "damda-train" /B python -m src.train --config configs\baseline.yaml --resume >> train_console.log 2>&1

echo.
echo Training resumed in background.
echo   Check progress: powershell -Command "Get-Content runs\main\train.log -Tail 5 -Encoding UTF8"
echo   Check GPU    : nvidia-smi
echo.
