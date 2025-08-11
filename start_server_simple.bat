@echo off
title AI Video Search API - Simple Server
echo =====================================================
echo     AI Video Search API - SIMPLE VERSION  
echo =====================================================
echo.
echo ✅ MEMORY OPTIMIZED FEATURES:
echo   - Fast startup, low memory usage
echo   - Individual frame search
echo   - Basic video search without heavy operations
echo   - Stable and reliable
echo.
echo 🚀 RECOMMENDED for production use and systems with limited memory
echo.
echo Available endpoints after startup:
echo   http://localhost:8001/docs           - API Documentation
echo   http://localhost:8001/health         - Health Check
echo   http://localhost:8001/search_frames  - Individual frame search
echo   http://localhost:8001/search_simple  - Basic video search
echo.
echo Press Ctrl+C to stop the server
echo =====================================================
echo.

REM Change to the directory where the batch file is located
cd /d "%~dp0"

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
if not exist "api\app_simple.py" (
    echo ERROR: api\app_simple.py not found!
    pause
    exit /b 1
)

if not exist "index\meta.parquet" (
    echo WARNING: index\meta.parquet not found!
    echo You may need to run: python build_meta.py
    echo.
)

if not exist "index\faiss\ip_flat.index" (
    echo WARNING: FAISS index not found!
    echo You may need to run: python build_embeddings.py
    echo.
)

REM Start the simple server
echo Starting MEMORY-OPTIMIZED uvicorn server...
echo This version loads faster and uses less memory...
python -m uvicorn api.app_simple:app --host 0.0.0.0 --port 8001 --reload

REM Handle graceful shutdown
echo.
echo Server stopped.
pause
