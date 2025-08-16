#!/bin/bash
# Enhanced AI Video Search - Optimal Setup Script for macOS
# Automatically creates venv with Python 3.10 for best compatibility

echo "======================================================================"
echo "    Enhanced AI Video Search - Optimal Python Setup (macOS)"
echo "======================================================================"
echo

# Check if Python 3.10 is available
echo "Checking for Python 3.10..."
if command -v python3.10 &> /dev/null; then
    echo "[OK] Python 3.10 found!"
    python3.10 --version
    PYTHON_CMD=python3.10
elif command -v python3 &> /dev/null && python3 --version | grep -q "3.10"; then
    echo "[OK] Python 3.10 found!"
    python3 --version
    PYTHON_CMD=python3
else
    echo "[WARNING] Python 3.10 not found!"
    echo
    echo "[RECOMMENDATION] Install Python 3.10 for best AI compatibility"
    echo
    echo "[DOWNLOAD] Install Python 3.10 via Homebrew:"
    echo "    brew install python@3.10"
    echo
    echo "[ALTERNATIVE] Download from:"
    echo "    https://www.python.org/downloads/release/python-31011/"
    echo
    echo "[OPTION] Or continue with current Python (may have compatibility issues)"
    
    if command -v python3 &> /dev/null; then
        python3 --version
        read -p "Continue with current Python? (y/n): " choice
        if [[ "$choice" != "y" && "$choice" != "Y" ]]; then
            echo
            echo "[EXIT] Please install Python 3.10 and run this script again"
            echo "[BREW] brew install python@3.10"
            exit 1
        fi
        PYTHON_CMD=python3
    else
        echo "[ERROR] No Python found! Please install Python 3.10"
        exit 1
    fi
fi

echo
echo "[WARNING] Using current Python - some AI features may not work"

echo
echo "[SETUP] Creating virtual environment with optimal Python..."

# Remove old venv if exists
if [[ -d ".venv" ]]; then
    echo "[CLEANUP] Removing existing virtual environment..."
    rm -rf .venv
fi

# Create new venv with Python 3.10
echo "[CREATE] Creating new virtual environment..."
$PYTHON_CMD -m venv .venv

if [[ $? -ne 0 ]]; then
    echo "[ERROR] Failed to create virtual environment"
    exit 1
fi

echo "[OK] Virtual environment created successfully!"
echo

# Activate venv
echo "[ACTIVATE] Activating virtual environment..."
source .venv/bin/activate

# Verify Python version in venv
echo "[VERSION] Python version in venv:"
python --version

echo
echo "[INSTALL] Installing dependencies..."
echo "This may take several minutes..."
echo

# Run setup
python setup.py

if [[ $? -eq 0 ]]; then
    echo
    echo "[SUCCESS] SETUP COMPLETE!"
    echo "======================================================================"
    echo
    echo "[NEXT STEPS] Next steps:"
    echo "    1. Launch the system: python main_launcher.py"
    echo "    2. Choose option 1 for Full AI experience"
    echo "    3. Or choose option 2 for Lite version"
    echo
    echo "[OPTIONAL] Optional: Configure API keys in .env file"
    echo "    cp .env.example .env"
    echo
    echo "[DOCS] Full documentation: README.md"
    echo "======================================================================"
else
    echo
    echo "[WARNING] Setup completed with some issues"
    echo "[TIP] Try running: python main_launcher.py"
    echo "    Choose option 3 to auto-install missing dependencies"
fi

echo
echo "[LAUNCH] Press any key to launch the system..."
read -n 1

# Launch the system
python main_launcher.py
