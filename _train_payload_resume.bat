@echo off
REM Resume training entry point called by schtasks (via train_detach_resume.bat).
REM Do not call directly. Use train_detach_resume.bat instead.

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

REM --resume picks up the latest checkpoint automatically.
REM Output appended to existing train_console.log.
python -m src.train --config configs/baseline.yaml --resume >> train_console.log 2>&1
