@echo off
setlocal
REM Start training detached from SSH/cmd. Uses Windows Task Scheduler (schtasks)
REM so the child python is not killed when SSH disconnects or laptop shuts down.
REM Caveat: lab PC sleep/shutdown still kills it. Run powercfg first.

set TASKNAME=damda-train

REM Safety guard: warn if existing checkpoints would be overwritten
if exist C:\damda\AI\checkpoints\epoch001.pt (
    echo [WARN] checkpoints\epoch001.pt found ^(previous run^).
    echo        New run will overwrite the same folder.
    echo        To preserve, first run:
    echo            ren runs\main main_vX
    echo            ren checkpoints checkpoints_vX
    echo.
    set /p ans=Continue anyway [y/N]:
    if /i not "%ans%"=="y" exit /b 1
)

REM Remove existing task with the same name (leftover from previous run)
schtasks /query /tn "%TASKNAME%" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Deleting existing "%TASKNAME%" task
    schtasks /delete /tn "%TASKNAME%" /f >nul
)

REM Register task ^(one-time, scheduled in far future, then manually triggered^)
REM Date format follows system locale. Korean Windows uses yyyy/mm/dd.
schtasks /create /tn "%TASKNAME%" /tr "C:\damda\AI\_train_payload.bat" /sc once /sd 2099/01/01 /st 00:00 /f >nul
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
echo  Training task "%TASKNAME%" started ^(fully detached^)
echo =========================================================
echo  - SSH disconnect: safe
echo  - Laptop shutdown: safe
echo  - Lab PC sleep/shutdown: still kills it ^(set powercfg first^)
echo.
echo  Check progress:
echo    powershell -Command "Get-Content C:\damda\AI\runs\main\train.log -Tail 5 -Encoding UTF8"
echo    nvidia-smi
echo.
echo  Task status:
echo    schtasks /query /tn "%TASKNAME%" /v /fo LIST
echo.
echo  Force stop training:
echo    schtasks /end /tn "%TASKNAME%"
echo.
echo  Delete task after training done:
echo    schtasks /delete /tn "%TASKNAME%" /f
echo.
endlocal
