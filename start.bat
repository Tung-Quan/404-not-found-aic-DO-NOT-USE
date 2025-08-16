@echo off
REM Frame Search System - Windows Batch Launcher
REM ============================================

echo.
echo FRAME SEARCH SYSTEM - Windows Launcher
echo ============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.8+
    echo Download from: https://python.org/downloads/
    pause
    exit /b 1
)

echo OK: Python detected
echo.

REM Check if launcher exists
if not exist "launcher.py" (
    echo ERROR: launcher.py not found!
    echo Make sure you're in the correct directory
    pause
    exit /b 1
)

echo Starting Frame Search Launcher...
echo Press Ctrl+C to stop
echo.

REM Run the launcher
python launcher.py

echo.
echo Frame Search System stopped
pause
