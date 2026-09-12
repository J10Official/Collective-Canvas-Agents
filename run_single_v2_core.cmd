@echo off
setlocal
cd /d "%~dp0"
python -m harness.single_batch --config configs\single_v2_core.json
set "exit_code=%ERRORLEVEL%"
if not "%exit_code%"=="0" pause
exit /b %exit_code%
