@echo off
title AI Video Search API Server
echo ================================================
echo        AI Video Search API Server
echo ================================================
echo.
echo Starting server...
echo Please wait for "Application startup complete" message...
echo.
echo Available endpoints after startup:
echo   http://localhost:8000/docs     - API Documentation
echo   http://localhost:8000/health   - Health Check
echo   http://localhost:8000/         - Root Info
echo.
echo Press Ctrl+C to stop the server
echo ================================================
echo.

cd /d "E:\Disk D\BK LEARNING\LEARNING\react\Project"

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv .venv
    pause
    exit /b 1
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Check if required files exist
if not exist "api\app.py" (
    echo ERROR: api\app.py not found!
    pause
    exit /b 1
)

if not exist "index\meta.parquet" (
    echo WARNING: index\meta.parquet not found!
    echo You may need to run: python build_meta.py
    echo.
)

REM Start the server
echo Starting uvicorn server...
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

REM Handle graceful shutdown
echo.
echo ================================================
echo Server stopped.
echo ================================================
pause
