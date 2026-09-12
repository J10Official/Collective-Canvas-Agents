@echo off
setlocal
cd /d "%~dp0"
python -m harness.server --size 32 --port 8767 --state harness\data\canvas.json --open
set "exit_code=%ERRORLEVEL%"
if not "%exit_code%"=="0" pause
exit /b %exit_code%
