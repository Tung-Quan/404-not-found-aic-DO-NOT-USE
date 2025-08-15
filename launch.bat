@echo off
title Enhanced Video Search System

if "%1"=="--once" goto once_mode

:start
echo ========================================================
echo        Enhanced Video Search System v2.0
echo ========================================================
echo.
echo Quick Launch Options:
echo.
echo [1]  Complete Setup (First Time Users)
echo [2] Enhanced API Server
echo [3] Advanced Web Interface  
echo [4] Interactive Demo
echo [5]  System Status
echo [6]  Run Once Mode (no loop)
echo [7]  Install Missing Dependencies
echo [0] Exit
echo.
set /p choice="Choose option [0-7]: "

if "%choice%"=="1" (
    echo.
    echo Running Complete Setup...
    call .venv\Scripts\activate.bat
    python scripts\setup_complete.py
    echo.
    echo Press any key to return to menu or Ctrl+C to exit...
    pause >nul
    goto start
) else if "%choice%"=="2" (
    echo.
    echo Starting Enhanced API Server...
    call scripts\start_server.bat
    echo.
    echo Server stopped. Press any key to return to menu...
    pause >nul
    goto start
) else if "%choice%"=="3" (
    echo.
    echo Starting Advanced Web Interface...
    call .venv\Scripts\activate.bat
    streamlit run src\ui\enhanced_web_interface.py --server.port 8501
    echo.
    echo Interface stopped. Press any key to return to menu...
    pause >nul
    goto start
) else if "%choice%"=="4" (
    echo.
    echo Starting Interactive Demo...
    call .venv\Scripts\activate.bat
    python demos\enhanced_video_demo.py
    echo.
    echo Demo completed. Press any key to return to menu...
    pause >nul
    goto start
) else if "%choice%"=="5" (
    echo.
    echo Checking System Status...
    call .venv\Scripts\activate.bat
    python scripts\check_status.py
    echo.
    echo Status check completed. Press any key to return to menu or Ctrl+C to exit...
    pause >nul
    goto start
) else if "%choice%"=="6" (
    goto once_mode
) else if "%choice%"=="7" (
    echo.
    echo Installing Missing Dependencies...
    call .venv\Scripts\activate.bat
    python scripts\install_dependencies.py
    echo.
    echo Installation completed. Press any key to return to menu...
    pause >nul
    goto start
) else if "%choice%"=="0" (
    echo Goodbye!
    exit /b 0
) else (
    echo Invalid choice, please try again.
    echo Press any key to return to menu...
    pause >nul
    goto start
)

:once_mode
echo ========================================================
echo        ONE-TIME MODE - No Loop
echo ========================================================
echo.
echo [1]  Complete Setup
echo [2] Start Enhanced API Server 
echo [3] Start Advanced Web Interface
echo [4] Start Interactive Demo
echo [5]  Check System Status
echo [6]  Install Missing Dependencies
echo [0] Exit
echo.
set /p oncechoice="Choose option [0-6]: "

if "%oncechoice%"=="1" (
    echo.
    echo Running Complete Setup...
    call .venv\Scripts\activate.bat
    python scripts\setup_complete.py
) else if "%oncechoice%"=="2" (
    echo.
    echo Starting Enhanced API Server...
    call scripts\start_server.bat
) else if "%oncechoice%"=="3" (
    echo.
    echo Starting Advanced Web Interface...
    call .venv\Scripts\activate.bat
    streamlit run src\ui\enhanced_web_interface.py --server.port 8501
) else if "%oncechoice%"=="4" (
    echo.
    echo Starting Interactive Demo...
    call .venv\Scripts\activate.bat
    python demos\enhanced_video_demo.py
) else if "%oncechoice%"=="5" (
    echo.
    echo Checking System Status...
    call .venv\Scripts\activate.bat
    python scripts\check_status.py
) else if "%oncechoice%"=="6" (
    echo.
    echo Installing Missing Dependencies...
    call .venv\Scripts\activate.bat
    python scripts\install_dependencies.py
) else if "%oncechoice%"=="0" (
    echo Goodbye!
) else (
    echo Invalid choice.
)

echo.
echo Script completed.
