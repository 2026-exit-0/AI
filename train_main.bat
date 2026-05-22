@echo off
REM Start new main training (after checkpoint cleanup). All-ASCII for cmd safety.

cd /d C:\damda\AI
call myvenv\Scripts\activate.bat

REM Safety guard: warn if existing checkpoints would be overwritten
if exist checkpoints\epoch001.pt (
    echo.
    echo [WARN] Existing checkpoints found in checkpoints\
    echo        Starting new run may overwrite them.
    echo        To preserve, first run:
    echo            ren runs\main main_vX   ^(X = version^)
    echo            rmdir /s /q checkpoints
    echo.
    set /p ans=Continue anyway? (y/N):
    if /i not "%ans%"=="y" exit /b 1
)

REM Background detached run -- survives SSH disconnect best-effort
start "damda-train" /B python -m src.train --config configs/baseline.yaml > train_console.log 2>&1

echo.
echo Training started in background.
echo   Check progress: powershell -Command "Get-Content runs\main\train.log -Tail 5 -Encoding UTF8"
echo   Check GPU    : nvidia-smi
echo.
