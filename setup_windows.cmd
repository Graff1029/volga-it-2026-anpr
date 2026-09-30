@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
echo Setting up Volga IT for Windows and RTX 5070...
py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.12 x64 is required. Read README.md, then run this file again.
    pause
    exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 goto fail
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail
".venv\Scripts\python.exe" check_env.py --require-cuda
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip freeze > results\environment-requirements.txt
echo Setup finished. Send results\environment.json to the assistant.
pause
exit /b 0
:fail
echo SETUP FAILED. Send the last error lines to the assistant.
pause
exit /b 1
