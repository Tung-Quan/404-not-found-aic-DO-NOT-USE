@echo off
title AI Video Search - Full Setup
echo ================================================
echo     AI Video Search System - Full Setup
echo ================================================
echo.
echo This script will run the complete pipeline:
echo 1. Extract frames from videos
echo 2. Build metadata index  
echo 3. Generate AI embeddings
echo 4. Build search index
echo 5. Start API server
echo.
set /p confirm="Continue? (y/n): "
if /i "%confirm%" neq "y" exit /b 0

REM Change to the directory where the batch file is located
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv .venv
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo.
echo ================================================
echo Step 1: Extracting frames from videos...
echo ================================================
powershell -Command "& {$ErrorActionPreference = 'Stop'; Get-ChildItem videos -File | ForEach-Object { $name = $_.BaseName; New-Item -ItemType Directory -Force -Path \"frames/$name\" | Out-Null; ffmpeg -y -i $_.FullName -vf fps=1 \"frames/$name/frame_%%06d.jpg\" }}"

if %errorlevel% neq 0 (
    echo ERROR: Frame extraction failed!
    pause
    exit /b 1
)

echo.
echo ================================================
echo Step 2: Building metadata index...
echo ================================================
python build_meta.py

if %errorlevel% neq 0 (
    echo ERROR: Metadata building failed!
    pause
    exit /b 1
)

echo.
echo ================================================
echo Step 3: Generating AI embeddings...
echo ================================================
python scripts/encode_siglip.py

if %errorlevel% neq 0 (
    echo ERROR: Embedding generation failed!
    pause
    exit /b 1
)

echo.
echo ================================================
echo Step 4: Building search index...
echo ================================================
python scripts/build_faiss.py

if %errorlevel% neq 0 (
    echo ERROR: Search index building failed!
    pause
    exit /b 1
)

echo.
echo ================================================
echo Step 5: Starting API server...
echo ================================================
echo Server will start in 3 seconds...
timeout /t 3 /nobreak

echo.
echo Available endpoints after startup:
echo   http://localhost:8000/docs     - API Documentation  
echo   http://localhost:8000/health   - Health Check
echo   http://localhost:8000/         - Root Info
echo.
echo Press Ctrl+C to stop the server
echo ================================================
echo.

python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

echo.
echo ================================================
echo Setup completed and server stopped.
echo ================================================
pause
