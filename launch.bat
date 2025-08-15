@echo off
title Enhanced Video Search System

echo ========================================================
echo        🎥 Enhanced Video Search System v2.0
echo ========================================================
echo.
echo 🚀 Quick Launch Options:
echo.
echo [1] 🛠️  Complete Setup (First Time Users)
echo [2] 🔥 Enhanced API Server
echo [3] 🌐 Advanced Web Interface  
echo [4] 🎬 Interactive Demo
echo [5] ⚙️  System Status
echo [0] Exit
echo.
set /p choice="Choose option [0-5]: "

if "%choice%"=="1" (
    echo.
    echo 🛠️  Running Complete Setup...
    python scripts\setup_complete.py
    pause
) else if "%choice%"=="2" (
    echo.
    echo 🔥 Starting Enhanced API Server...
    call scripts\start_server.bat
) else if "%choice%"=="3" (
    echo.
    echo 🌐 Starting Advanced Web Interface...
    call .venv\Scripts\activate.bat
    streamlit run src\ui\enhanced_web_interface.py --server.port 8501
) else if "%choice%"=="4" (
    echo.
    echo 🎬 Starting Interactive Demo...
    call .venv\Scripts\activate.bat
    python demos\enhanced_video_demo.py
) else if "%choice%"=="5" (
    echo.
    echo ⚙️  Checking System Status...
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
    import tensorflow_hub as hub
    print('✅ TensorFlow Hub available')
except: print('❌ TensorFlow Hub not installed')
"
    pause
) else if "%choice%"=="0" (
    echo Goodbye!
    exit /b 0
) else (
    echo Invalid choice, please try again.
    pause
    goto start
)

:start
goto start
