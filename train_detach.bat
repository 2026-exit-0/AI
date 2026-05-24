@echo off
setlocal
REM SSH 세션 / 노트북 종료와 무관하게 학습 시작.
REM Windows 작업 스케줄러(schtasks) 로 등록 후 즉시 실행 → 부모 cmd 와 완전 분리.
REM   - SSH 끊겨도 OK
REM   - 노트북 꺼도 OK (학습은 졸프실 PC 에서 계속)
REM   - PC 자체가 절전/종료되면 당연히 죽음 (powercfg 절전 비활성 미리 해둘 것)

set TASKNAME=damda-train

REM 안전 가드: 기존 체크포인트 존재 시 경고
if exist C:\damda\AI\checkpoints\epoch001.pt (
    echo [WARN] checkpoints\epoch001.pt 가 있습니다 ^(이전 학습 결과^).
    echo        새 학습은 같은 폴더에 덮어쓰기 시작합니다.
    echo        보존하려면 먼저:
    echo            ren runs\main main_v?
    echo            ren checkpoints checkpoints_v?
    echo.
    set /p ans=Continue anyway [y/N]:
    if /i not "%ans%"=="y" exit /b 1
)

REM 동일 이름 task 가 있으면 제거 (이전 실행 잔재)
schtasks /query /tn "%TASKNAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] 기존 "%TASKNAME%" task 삭제
    schtasks /delete /tn "%TASKNAME%" /f >nul
)

REM task 등록 (먼 미래 1회성으로 등록 후 수동 trigger)
schtasks /create /tn "%TASKNAME%" /tr "C:\damda\AI\_train_payload.bat" /sc once /sd 01/01/2099 /st 00:00 /f >nul
if errorlevel 1 (
    echo [ERROR] schtasks 등록 실패. 권한 확인.
    exit /b 1
)

REM 즉시 실행
schtasks /run /tn "%TASKNAME%"
if errorlevel 1 (
    echo [ERROR] schtasks 실행 실패.
    exit /b 1
)

echo.
echo =========================================================
echo  학습 task "%TASKNAME%" 시작됨 ^(완전 detach^)
echo =========================================================
echo  - SSH 끊김 영향 없음
echo  - 노트북 종료 영향 없음
echo  - 졸프실 PC 절전/종료되면 죽음 ^(미리 powercfg 설정 필수^)
echo.
echo  진행 확인:
echo    powershell -Command "Get-Content C:\damda\AI\runs\main\train.log -Tail 5 -Encoding UTF8"
echo    nvidia-smi
echo.
echo  task 상태:
echo    schtasks /query /tn "%TASKNAME%" /v /fo LIST
echo.
echo  학습 강제 중단:
echo    schtasks /end /tn "%TASKNAME%"
echo.
echo  task 완전 삭제 (학습 끝난 뒤):
echo    schtasks /delete /tn "%TASKNAME%" /f
echo.
endlocal
