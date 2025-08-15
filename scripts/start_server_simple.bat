@echo off
title Simple Enhanced Video Search API
echo ========================================
echo   🚀 Simple Enhanced Video Search API
echo ========================================
echo.
echo ✅ Fast startup, no TensorFlow Hub required
echo 📡 API will run on: http://localhost:8000
echo 📚 Documentation: http://localhost:8000/docs
echo.

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Checking system status...
python -c "import os; print('✅ Virtual environment active' if 'VIRTUAL_ENV' in os.environ else '❌ venv not active')"

echo.
echo 🔄 Starting Simple Enhanced API...
echo.

cd api
python simple_enhanced_api.py

echo.
echo Server stopped. Press any key to exit.
pause
