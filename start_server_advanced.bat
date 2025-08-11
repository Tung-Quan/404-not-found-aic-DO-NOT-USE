@echo off
title AI Video Search API - Advanced Server
echo =====================================================
echo     AI Video Search API - ADVANCED VERSION
echo =====================================================
echo.
echo 🔥 ADVANCED FEATURES:
echo   - Full video-level search with aggregation
echo   - Top 5 frames per video in results
echo   - TF-IDF text search integration
echo   - Advanced scoring algorithms
echo.
echo ⚠️  WARNING: May have memory issues on some systems
echo    If you get memory errors, use start_server_simple.bat instead
echo.
echo Available endpoints after startup:
echo   http://localhost:8000/docs       - API Documentation
echo   http://localhost:8000/health     - Health Check
echo   http://localhost:8000/search     - Advanced video search
echo   http://localhost:8000/search_frames - Individual frame search
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

if not exist "index\embeddings\frames.f16.mmap" (
    echo WARNING: Embedding files not found!
    echo You may need to run: python build_embeddings.py
    echo.
)

REM Start the advanced server
echo Starting ADVANCED uvicorn server...
echo Loading embeddings into memory (may take time)...
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

REM Handle graceful shutdown
echo.
echo Server stopped.
pause
