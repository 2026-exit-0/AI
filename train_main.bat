@echo off
REM 새 본 학습 시작 (체크포인트 초기화 후). 이미 학습이 있으면 보존부터 하세요.

cd /d C:\damda\AI
call myvenv\Scripts\activate.bat

REM 안전 가드: 이미 ckpt 있으면 사용자 확인
if exist checkpoints\epoch001.pt (
    echo.
    echo [WARN] checkpoints\ 에 기존 학습 결과가 있습니다.
    echo        새 학습을 시작하면 덮어쓸 수 있습니다.
    echo        보존하려면 먼저 다음을 실행하세요:
    echo            ren runs\main main_vX  (X = 버전번호)
    echo            rmdir /s /q checkpoints
    echo.
    set /p ans="그대로 시작할까요? (y/N): "
    if /i not "%ans%"=="y" exit /b 1
)

REM 백그라운드 분리 실행
start "damda-train" /B python -m src.train --config configs\baseline.yaml > train_console.log 2>&1

echo.
echo 새 학습을 백그라운드에서 시작했습니다.
echo   - 진행 확인: powershell -Command "Get-Content runs\main\train.log -Tail 5 -Encoding UTF8"
echo   - GPU 확인 : nvidia-smi
echo.
