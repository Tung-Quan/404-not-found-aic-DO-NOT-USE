@echo off
title Enhanced Video Search System - Smart Auto-Install

REM Set encoding and environment for better compatibility
set PYTHONIOENCODING=utf-8
set PYTHONPATH=%CD%
chcp 65001 >nul 2>&1

echo ================================================================
echo     Enhanced Video Search System - Smart Auto-Install
echo ================================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "start.py" (
    echo [ERROR] start.py not found. Please run from project root directory.
    pause
    exit /b 1
)

REM Auto-activate virtual environment if exists
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call .venv\Scripts\activate.bat
)

REM Run smart launcher
echo [INFO] Starting Smart Auto-Install Launcher...
echo.
python start.py

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start the system
    echo Check the error messages above for details.
    pause
    exit /b 1
)

echo.
echo [INFO] System exited normally.
pause
