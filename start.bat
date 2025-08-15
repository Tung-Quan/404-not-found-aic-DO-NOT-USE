@echo off
title Enhanced Video Search System

REM Set encoding and environment
set PYTHONIOENCODING=utf-8
chcp 65001 >nul 2>&1

echo ================================================================
echo     Enhanced Video Search System - Quick Start
echo ================================================================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Check directory
if not exist "scripts" (
    echo [ERROR] Please run from project root directory
    pause
    exit /b 1
)

REM Run unified launcher
echo [INFO] Starting unified launcher...
echo.
python start.py

pause
