@echo off
title AI Video Search - Installation Guide
echo =====================================================
echo     AI Video Search System - Installation
echo =====================================================
echo.
echo 📋 System Requirements:
echo   - Python 3.8+ installed
echo   - FFmpeg installed (for video processing)
echo   - 4GB+ RAM (8GB+ recommended for Advanced server)
echo   - Internet connection (for AI model download)
echo.
echo 🚀 Quick Setup Steps:
echo   1. Clone/download this project
echo   2. Run: python -m venv .venv
echo   3. Run: .venv\Scripts\activate
echo   4. Run: pip install -r requirements.txt
echo   5. Put your videos in videos/ folder
echo   6. Run: setup_and_run.bat (first time)
echo   7. Run: start_server_simple.bat (daily use)
echo.
echo 📁 Project Structure:
echo   start_server_simple.bat   - Memory optimized server (recommended)
echo   start_server_advanced.bat - Full featured server
echo   setup_and_run.bat         - Complete setup from videos
echo   requirements.txt          - Python dependencies
echo.
echo ⚠️  Note: All batch files now use relative paths
echo    They work from any location on any computer!
echo.
pause
