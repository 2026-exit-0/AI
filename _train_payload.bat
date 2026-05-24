@echo off
REM Actual training entry point called by schtasks (via train_detach.bat).
REM Do not call directly. Use train_detach.bat instead.

cd /d C:\damda\AI

REM Auto-activate venv (lab PC .venv / laptop myvenv)
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else if exist ..\.venv\Scripts\activate.bat (
    call ..\.venv\Scripts\activate.bat
) else if exist myvenv\Scripts\activate.bat (
    call myvenv\Scripts\activate.bat
) else if exist ..\myvenv\Scripts\activate.bat (
    call ..\myvenv\Scripts\activate.bat
)

REM Do NOT use "start /B" here. schtasks context will exit when the .bat
REM finishes; the child must run in foreground so schtasks waits.
python -m src.train --config configs/baseline.yaml > train_console.log 2>&1
