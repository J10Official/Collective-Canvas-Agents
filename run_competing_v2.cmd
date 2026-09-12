@echo off
setlocal
cd /d "%~dp0"
python -m harness.competing_batch --config configs\competing_v2.json
set "exit_code=%ERRORLEVEL%"
if not "%exit_code%"=="0" pause
exit /b %exit_code%
