@echo off
title AI Video Search API - Advanced Server
echo =====================================================
echo    🚀 AI Video Search API - ADVANCED VERSION
echo =====================================================
echo.
echo 🔥 ADVANCED FEATURES:
echo   - TensorFlow Hub Universal Sentence Encoder
echo   - EfficientNet V2 for visual features
echo   - Multi-modal search (text + visual)
echo   - Language detection (Vietnamese/English)
echo   - Dynamic score weighting
echo   - Enhanced metadata processing
echo.
echo ⚠️  WARNING: Memory intensive with TF Hub models
echo    Use start_server_simple.bat for lighter version
echo.
echo Available endpoints after startup:
echo   http://localhost:8000/docs           - API Documentation  
echo   http://localhost:8000/health         - Health Check
echo   http://localhost:8000/enhanced_search - Enhanced TF Hub search
echo   http://localhost:8000/analyze_query  - Query analysis
echo.
echo Press Ctrl+C to stop the server
echo =====================================================
echo.

REM Change to the directory where the batch file is located
cd /d "%~dp0"

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Checking system status...
python -c "import os; print('✅ Virtual environment active' if 'VIRTUAL_ENV' in os.environ else '❌ venv not active')"

echo.
echo Checking TensorFlow Hub availability...
python -c "import tensorflow_hub as hub; print('✅ TensorFlow Hub ready')" 2>nul || echo "❌ TensorFlow Hub not installed"

echo.
echo 🔄 Starting Enhanced TF Hub API...
echo This may take longer as TF Hub models are loaded...
echo.

cd api
python app.py

echo.
echo Server stopped. Press any key to exit.
pause
