@echo off
echo =====================================================
echo 🔍 Simplified AI Search System - Quick Start
echo =====================================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

echo ✅ Python detected

REM Check if virtual environment exists
if exist ".venv" (
    echo 🐍 Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo ⚠️ No virtual environment found, using system Python
)

REM Parse command line arguments
if "%1"=="--status" (
    echo 📊 Checking system status...
    python quick_start.py --status
    goto end
)

if "%1"=="--manual" (
    echo 📋 Showing manual commands...
    python quick_start.py --manual
    goto end
)

if "%1"=="--start" (
    echo 🚀 Starting web interface...
    python quick_start.py --start
    goto end
)

if "%1"=="--setup" (
    echo 🔧 Setup only...
    python quick_start.py --setup-only
    goto end
)

if "%1"=="--build" (
    if "%2"=="" (
        echo ❌ Please specify frames directory: quick_start.bat --build frames
        goto end
    )
    echo 🔨 Building index from %2...
    python quick_start.py --build-index %2
    goto end
)

REM Default: Run complete workflow
echo 🚀 Running complete workflow...
python quick_start.py

:end
echo.
echo 💡 Available commands:
echo    quick_start.bat           - Complete workflow
echo    quick_start.bat --status  - Check system status
echo    quick_start.bat --manual  - Show manual commands
echo    quick_start.bat --start   - Start web interface only
echo    quick_start.bat --setup   - Setup dependencies only
echo    quick_start.bat --build ^<dir^> - Build index from directory
echo.
if not "%1"=="--start" pause
