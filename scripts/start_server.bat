@echo off
title Enhanced Video Search System

:start
echo ================    echo     echo � Starting Interactive CLI Demo...
    echo.
    call .venv\Scripts\activate.bat
    python demos\interactive_search_demo.pydvanced TensorFlow Hub model selection
    echo.
    call .venv\Scripts\activate.bat
    streamlit run src\ui\enhanced_web_interface.py --server.port 8501=====================================
echo        🎯 Enhanced Video Search System - Main Menu
echo =========================================================
echo.
echo 🚀 ENHANCED SYSTEM WITH TENSORFLOW HUB INTEGRATION
echo.
echo Available Options:
echo [1] 🔥 Enhanced API with TensorFlow Hub (Port 8000) - RECOMMENDED
echo     - Universal Sentence Encoder Multilingual
echo     - EfficientNet V2 visual features
echo     - Advanced query analysis
echo.
echo [2] � Simple Enhanced API (Port 8000) - Fast startup
echo     - No TensorFlow Hub required
echo     - Text similarity search
echo     - Enhanced metadata processing
echo.
echo [3] 🌐 Web Interface (Port 5000) - Browser UI
echo     - Beautiful responsive interface
echo     - Real-time search results
echo     - Visual result display
echo.
echo [4] 🎥 Enhanced Video Processing Interface (Port 8501)
echo     - Interactive TensorFlow Hub model selection
echo     - Intelligent overlap detection
echo     - Advanced video processing
echo.
echo [5] 💻 Interactive CLI Demo
echo     - Command-line interface
echo     - Real-time search testing
echo.
echo [6] 🎬 Enhanced Video Processing Demo
echo     - TensorFlow Hub model selection demo
echo     - Interactive video processing test
echo.
echo [7] 🛠️  Complete System Setup & Dependencies
echo     - One-click setup với TensorFlow Hub
echo     - Dependency installation và testing
echo     - Project structure verification
echo [8] 📊 System Status Check
echo [0] Exit
echo.
set /p choice="Choose option [0-8]: "

if "%choice%"=="1" (
    echo.
    echo 🔄 Starting Enhanced API with TensorFlow Hub...
    echo ⚠️  Note: Requires TensorFlow Hub dependencies
    echo 📡 API: http://localhost:8000
    echo 📚 Docs: http://localhost:8000/docs
    echo.
    call .venv\Scripts\activate.bat
    cd src\api
    python app.py
) else if "%choice%"=="2" (
    echo.
    echo � Starting Simple Enhanced API...
    echo 📡 API: http://localhost:8000
    echo 📚 Docs: http://localhost:8000/docs
    echo.
    call .venv\Scripts\activate.bat
    cd src\api
    python simple_enhanced_api.py
) else if "%choice%"=="3" (
    echo.
    echo 🔄 Starting Web Interface...
    echo 🌐 Web UI: http://localhost:5000
    echo.
    call .venv\Scripts\activate.bat
    python src\ui\web_search_app.py
) else if "%choice%"=="4" (
    echo.
    echo 🎥 Starting Enhanced Video Processing Interface...
    echo 🌐 Streamlit UI: http://localhost:8501
    echo � Advanced TensorFlow Hub model selection
    echo.
    call .venv\Scripts\activate.bat
    streamlit run enhanced_web_interface.py --server.port 8501
) else if "%choice%"=="5" (
    echo.
    echo �🔄 Starting Interactive CLI Demo...
    echo.
    call .venv\Scripts\activate.bat
    python interactive_search_demo.py
) else if "%choice%"=="6" (
    echo.
    echo 🎬 Starting Enhanced Video Processing Demo...
    echo 🤖 Interactive TensorFlow Hub model selection
    echo.
    call .venv\Scripts\activate.bat
    python demos\enhanced_video_demo.py
) else if "%choice%"=="7" (
    echo.
    echo 🛠️  Running Complete System Setup...
    echo 📦 Installing dependencies và testing TensorFlow Hub
    echo.
    call .venv\Scripts\activate.bat
    python setup_complete.py
    echo.
    echo Setup completed. Press any key to return to menu.
    pause >nul
    goto :start
) else if "%choice%"=="8" (
    echo.
    echo 📊 Checking System Status...
    call .venv\Scripts\activate.bat
    python -c "
import os, sys
print('✅ Virtual environment:', 'Active' if 'VIRTUAL_ENV' in os.environ else 'Not active')
try:
    import pandas as pd
    meta = pd.read_parquet('index/meta.parquet')
    print(f'✅ Metadata: {len(meta)} frames loaded')
except: print('❌ Metadata not found')
try:
    import faiss
    idx = faiss.read_index('index/faiss/ip_flat_chinese_clip.index')
    print(f'✅ FAISS index: {idx.ntotal} vectors')
except: print('❌ FAISS index not found')
try:
    import json
    with open('index/frames_meta.json') as f: meta = json.load(f)
    print(f'✅ Enhanced metadata: {len(meta)} records')
except: print('❌ Enhanced metadata not found')
try:
    import tensorflow as tf, tensorflow_hub as hub
    print('✅ TensorFlow Hub: Available')
except: print('⚠️  TensorFlow Hub: Not installed')
"
    echo.
    echo Status check completed. Press any key to return to menu.
    pause >nul
    goto :start
) else if "%choice%"=="0" (
    echo.
    echo 👋 Goodbye!
    exit /b 0
) else (
    echo.
    echo ❌ Invalid choice. Please select 0-6.
    timeout /t 2 >nul
    goto :start
)

pause
    echo Starting Simple Server...
    call start_server_simple.bat
) else if "%choice%"=="2" (
    echo Starting Advanced Server...
    call start_server_advanced.bat
) else (
    echo Goodbye!
    pause
    exit /b 0
)

echo.
echo Server selection complete.
pause
echo ================================================
echo Server stopped.
echo ================================================
pause
