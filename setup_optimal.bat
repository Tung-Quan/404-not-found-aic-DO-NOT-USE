@echo off
REM Enhanced AI Video Search - Optimal Setup Script
REM Automatically creates venv with Python 3.10 for best compatibility

echo ====================================================================
echo    Enhanced AI Video Search - Optimal Python Setup
echo ====================================================================
echo.

REM Check if Python 3.10 is available
echo Checking for Python 3.10...
py -3.10 --version >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo ✅ Python 3.10 found!
    py -3.10 --version
    set PYTHON_CMD=py -3.10
    goto :create_venv
)

python3.10 --version >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo ✅ Python 3.10 found!
    python3.10 --version
    set PYTHON_CMD=python3.10
    goto :create_venv
)

echo ⚠️ Python 3.10 not found!
echo.
echo 📋 RECOMMENDATION: Install Python 3.10 for best AI compatibility
echo.
echo 🔗 Download Python 3.10.11 from:
echo    https://www.python.org/downloads/release/python-31011/
echo.
echo ⚡ Or continue with current Python (may have compatibility issues)
python --version

set /p choice="Continue with current Python? (y/n): "
if /i "%choice%"=="n" (
    echo.
    echo 👋 Please install Python 3.10 and run this script again
    echo 🔗 https://www.python.org/downloads/release/python-31011/
    pause
    exit /b 1
)

echo.
echo ⚠️ Using current Python - some AI features may not work
set PYTHON_CMD=python

:create_venv
echo.
echo 📦 Creating virtual environment with optimal Python...

REM Remove old venv if exists
if exist .venv (
    echo 🔄 Removing existing virtual environment...
    rmdir /s /q .venv
)

REM Create new venv with Python 3.10
echo 🚀 Creating new virtual environment...
%PYTHON_CMD% -m venv .venv

if %ERRORLEVEL% neq 0 (
    echo ❌ Failed to create virtual environment
    pause
    exit /b 1
)

echo ✅ Virtual environment created successfully!
echo.

REM Activate venv
echo 🔧 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Verify Python version in venv
echo 🐍 Python version in venv:
python --version

echo.
echo 📦 Installing dependencies...
echo This may take several minutes...
echo.

REM Run setup
python setup.py

if %ERRORLEVEL% == 0 (
    echo.
    echo 🎉 SETUP COMPLETE!
    echo ====================================================================
    echo.
    echo 🚀 Next steps:
    echo    1. Launch the system: python main_launcher.py
    echo    2. Choose option 1 for Full AI experience
    echo    3. Or choose option 2 for Lite version
    echo.
    echo 🔑 Optional: Configure API keys in .env file
    echo    cp .env.example .env
    echo.
    echo 📚 Full documentation: README.md
    echo ====================================================================
) else (
    echo.
    echo ⚠️ Setup completed with some issues
    echo 💡 Try running: python main_launcher.py
    echo    Choose option 3 to auto-install missing dependencies
)

echo.
echo 👉 Press any key to launch the system...
pause >nul

REM Launch the system
python main_launcher.py

pause
