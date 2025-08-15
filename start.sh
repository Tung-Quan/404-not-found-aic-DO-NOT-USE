#!/bin/bash
# Enhanced Video Search System - Smart Auto-Install
# Cross-platform launcher for Linux/macOS

echo "================================================================"
echo "    Enhanced Video Search System - Smart Auto-Install"
echo "================================================================"
echo

# Function to detect Python command
detect_python() {
    if command -v python3 &> /dev/null; then
        echo "python3"
    elif command -v python &> /dev/null; then
        if python --version 2>&1 | grep -q "Python 3"; then
            echo "python"
        else
            echo ""
        fi
    else
        echo ""
    fi
}

# Check Python
PYTHON_CMD=$(detect_python)
if [ -z "$PYTHON_CMD" ]; then
    echo "[ERROR] Python 3.8+ not found. Please install Python 3.8 or higher."
    echo "Ubuntu/Debian: sudo apt install python3"
    echo "macOS: brew install python3"
    echo "Or download from: https://www.python.org/downloads/"
    exit 1
fi

echo "[INFO] Using Python: $PYTHON_CMD"
$PYTHON_CMD --version

# Check if we're in the right directory
if [ ! -f "start.py" ]; then
    echo "[ERROR] start.py not found. Please run from project root directory."
    exit 1
fi

# Auto-activate virtual environment if exists
if [ -f ".venv/bin/activate" ]; then
    echo "[INFO] Activating virtual environment..."
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    echo "[INFO] Activating virtual environment..."
    source venv/bin/activate
fi

# Set environment variables
export PYTHONIOENCODING=utf-8
export PYTHONPATH="$PWD:$PYTHONPATH"

# Run smart launcher
echo "[INFO] Starting Smart Auto-Install Launcher..."
echo
$PYTHON_CMD start.py

# Check exit status
if [ $? -ne 0 ]; then
    echo
    echo "[ERROR] Failed to start the system"
    echo "Check the error messages above for details."
    read -p "Press Enter to exit..."
    exit 1
fi

echo
echo "[INFO] System exited normally."
