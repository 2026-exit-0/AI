@echo off
REM 끊긴 학습을 이어 시작. SSH 들어와서 train_resume 만 치면 됨.
REM 새 학습 시작은 train_main.bat 또는 직접 명령 사용.

cd /d C:\damda\AI
call myvenv\Scripts\activate.bat

REM 백그라운드 분리 실행 — SSH 끊김 영향 최소화
start "damda-train" /B python -m src.train --config configs\baseline.yaml --resume >> train_console.log 2>&1

echo.
echo 학습을 백그라운드에서 재개했습니다.
echo   - 진행 확인: powershell -Command "Get-Content runs\main\train.log -Tail 5 -Encoding UTF8"
echo   - GPU 확인 : nvidia-smi
echo.
