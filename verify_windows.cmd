@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run start_windows.cmd first to install the project.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" check_installation.py > windows_check.txt 2>&1
type windows_check.txt
pause
