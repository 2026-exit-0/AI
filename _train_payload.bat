@echo off
REM schtasks 가 직접 호출하는 실제 학습 진입점.
REM train_detach.bat 가 이 파일을 task 로 등록 후 실행함.
REM 일반 사용자는 이 파일을 직접 호출할 일 없음 (train_detach.bat 사용).

cd /d C:\damda\AI

REM venv 자동 활성화 (졸프실 .venv / 노트북 myvenv 모두 지원)
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else if exist ..\.venv\Scripts\activate.bat (
    call ..\.venv\Scripts\activate.bat
) else if exist myvenv\Scripts\activate.bat (
    call myvenv\Scripts\activate.bat
) else if exist ..\myvenv\Scripts\activate.bat (
    call ..\myvenv\Scripts\activate.bat
)

REM start /B 사용 금지 — schtasks 컨텍스트가 종료되면 자식도 죽음.
REM 그냥 python 을 foreground 로 실행. schtasks 가 이 .bat 의 종료를 기다림.
python -m src.train --config configs/baseline.yaml > train_console.log 2>&1
