@echo off
title Enhanced Video Search System - Simple Launcher

echo ========================================================
echo        Enhanced Video Search System v2.0
echo ========================================================
echo.
echo Simple Launch Options:
echo.
echo [1]  Complete Setup (First Time Users)
echo [2] Enhanced API Server (will open separate menu)
echo [3] Advanced Web Interface  
echo [4] Interactive Demo
echo [5]  System Status (one-time check)
echo [6]  Install Missing Dependencies
echo [0] Exit
echo.
set /p choice="Choose option [0-6]: "

if "%choice%"=="1" (
    echo.
    echo Running Complete Setup...
    call .venv\Scripts\activate.bat
    python scripts\setup_complete.py
    echo.
    echo Setup completed.
    pause
) else if "%choice%"=="2" (
    echo.
    echo Opening Enhanced API Server Menu...
    call scripts\start_server.bat
) else if "%choice%"=="3" (
    echo.
    echo Starting Advanced Web Interface...
    echo Opening browser at: http://localhost:8501
    echo Press Ctrl+C to stop the server
    echo.
    call .venv\Scripts\activate.bat
    streamlit run src\ui\enhanced_web_interface.py --server.port 8501
) else if "%choice%"=="4" (
    echo.
    echo Starting Interactive Demo...
    call .venv\Scripts\activate.bat
    python demos\enhanced_video_demo.py
    echo.
    echo Demo completed.
    pause
) else if "%choice%"=="5" (
    echo.
    echo Checking System Status...
    call .venv\Scripts\activate.bat
    python scripts\check_status.py
    echo.
    echo Status check completed.
    pause
) else if "%choice%"=="6" (
    echo.
    echo Installing Missing Dependencies...
    call .venv\Scripts\activate.bat
    python scripts\install_dependencies.py
    echo.
    echo Installation completed.
    pause
) else if "%choice%"=="0" (
    echo.
    echo Goodbye!
) else (
    echo.
    echo Invalid choice. Please run the script again and select 0-6.
    pause
)

echo.
echo Script completed.
