@echo off
setlocal
REM Resume interrupted training detached from SSH/cmd.
REM Uses schtasks like train_detach.bat but runs _train_payload_resume.bat
REM (which adds --resume so the latest checkpoint is picked up automatically).

set TASKNAME=damda-train

REM Remove existing task with the same name (leftover from previous run)
schtasks /query /tn "%TASKNAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Deleting existing "%TASKNAME%" task
    schtasks /delete /tn "%TASKNAME%" /f >nul
)

REM Register resume task
schtasks /create /tn "%TASKNAME%" /tr "C:\damda\AI\_train_payload_resume.bat" /sc once /sd 2099/01/01 /st 00:00 /f >nul
if errorlevel 1 (
    echo [ERROR] schtasks /create failed. Check permissions or date locale.
    exit /b 1
)

REM Trigger immediately
schtasks /run /tn "%TASKNAME%"
if errorlevel 1 (
    echo [ERROR] schtasks /run failed.
    exit /b 1
)

echo.
echo =========================================================
echo  Resume task "%TASKNAME%" started ^(fully detached^)
echo =========================================================
echo  Resumes from the latest checkpoint in C:\damda\AI\checkpoints
echo  Same detach guarantees as train_detach.bat ^(SSH/laptop safe^).
echo.
echo  Check progress:
echo    powershell -Command "Get-Content C:\damda\AI\runs\main\train.log -Tail 5 -Encoding UTF8"
echo    nvidia-smi
echo.
endlocal
