@echo off
title LiveBridge 설치
echo ============================================
echo  LiveBridge 설치를 시작합니다
echo  (인터넷 연결 필요, 약 3~5GB 다운로드)
echo ============================================
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo [오류] Python 런처가 없습니다. https://python.org 에서 Python 3.11을 설치해 주세요.
    pause
    exit /b 1
)

if not exist .venv (
    echo [1/4] 가상환경 생성 중...
    py -3.11 -m venv .venv
    if errorlevel 1 (
        echo Python 3.11이 필요합니다
        pause
        exit /b 1
    )
) else (
    echo [1/4] 가상환경 확인 완료
)

where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo [안내] NVIDIA GPU 미감지 - CPU 버전으로 설치합니다 ^(다운로드가 훨씬 작아요^)
    echo        CPU에서는 음성인식이 자동으로 작은 모델로 전환됩니다.
    set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
    set "LLAMA_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cpu"
) else (
    set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
    set "LLAMA_INDEX=https://abetlen.github.io/llama-cpp-python/whl/cu124"
)

echo [2/4] PyTorch 설치 중... (GPU 버전은 약 2.5GB, 오래 걸려요)
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
.venv\Scripts\python.exe -m pip install "torch>=2.4" --index-url %TORCH_INDEX%

echo [3/4] 앱 의존성 설치 중...
.venv\Scripts\python.exe -m pip install -r requirements.txt

echo [4/4] llama.cpp 로컬 번역 LLM 런타임 설치 중...
.venv\Scripts\python.exe -m pip install llama-cpp-python --only-binary=llama-cpp-python --extra-index-url %LLAMA_INDEX%
if errorlevel 1 (
    echo [안내] llama-cpp-python 휠 설치 실패 - 앱은 transformers/NLLB/Google 번역으로 자동 대체됩니다.
)

echo.
echo [완료] 설치가 끝났습니다! run.bat 을 실행하세요.
echo        AI 모델 자체는 첫 [시작] 버튼을 누를 때 자동 다운로드됩니다. 약 4~5GB
pause
