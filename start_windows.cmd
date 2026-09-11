@echo off
setlocal
cd /d "%~dp0"
if not exist "requirements.txt" (
    echo Extract the complete ZIP first, then run start_windows.cmd inside the NextBeat folder.
    pause
    exit /b 1
)
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv
if not exist ".venv\Scripts\python.exe" (
    echo Install Python 3.12, then try again. See README.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo Dependency installation failed. Read the error above.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py
pause
