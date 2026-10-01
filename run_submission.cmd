@echo off
setlocal
if "%~2"=="" (
  echo Usage: run_submission.cmd INPUT_FOLDER OUTPUT_CSV
  exit /b 2
)
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" if not "%VOLGA_PYTHON%"=="" set "PY=%VOLGA_PYTHON%"
if not exist "%PY%" (
  echo ERROR: .venv is absent. Install Python 3.12 and run setup_windows.cmd first.
  exit /b 2
)
"%PY%" "%~dp0predict.py" --input "%~1" --output "%~2" --models "%~dp0models" --settings "%~dp0settings_submission.json" --ocr-checkpoint "%~dp0models\fine_tuned\easyocr_autoria_mixed_20260930_from_step1000_step1000.pt" --format-mode strict
exit /b %ERRORLEVEL%
