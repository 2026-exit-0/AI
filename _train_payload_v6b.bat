@echo off
REM v6-macro 학습 진입점 (schtasks 가 호출). 직접 실행 금지 — 아래 schtasks 로 detach 실행.
REM   schtasks /create /tn "damda-train-v6b" /tr "C:\damda\AI\_train_payload_v6b.bat" /sc once /sd 2099/01/01 /st 00:00 /f
REM   schtasks /run /tn "damda-train-v6b"

cd /d C:\damda\AI

REM venv 자동 활성화 (lab PC .venv / laptop myvenv)
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else if exist ..\.venv\Scripts\activate.bat (
    call ..\.venv\Scripts\activate.bat
) else if exist myvenv\Scripts\activate.bat (
    call myvenv\Scripts\activate.bat
) else if exist ..\myvenv\Scripts\activate.bat (
    call ..\myvenv\Scripts\activate.bat
)

REM schtasks 컨텍스트에서는 foreground 로 (start /B 금지)
python -m src.train --config configs/macro_v6b.yaml > train_console_v6b.log 2>&1
