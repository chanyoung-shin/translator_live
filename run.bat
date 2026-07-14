@echo off
title LiveBridge - 실시간 회의 번역
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo 먼저 setup.bat 을 실행해 주세요.
    pause
    exit /b 1
)

echo LiveBridge 서버를 시작합니다... 브라우저가 곧 열립니다.
echo 이 창을 닫으면 앱이 종료됩니다.
start "" /min cmd /c "timeout /t 3 /nobreak >nul & start "" http://127.0.0.1:8765"
.venv\Scripts\python.exe -m server.main
pause
