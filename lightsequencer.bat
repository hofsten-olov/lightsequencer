@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found in "%~dp0.venv".
    echo Run setup_dev_environment.bat first to create it.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" "lightSequencerDMX_full_v2.7.py"

if errorlevel 1 (
    echo.
    echo The app exited with an error - see above.
    pause
)
