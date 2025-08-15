#!/bin/bash
# Enhanced Video Search System - Unix Launcher

echo "================================================================"
echo "    Enhanced Video Search System - Quick Start"
echo "================================================================"

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    if [[ $PYTHON_VERSION == *"Python 3"* ]]; then
        PYTHON_CMD="python"
    else
        echo "[ERROR] Python 3 required. Found: $PYTHON_VERSION"
        exit 1
    fi
else
    echo "[ERROR] Python not found. Please install Python 3.8+"
    exit 1
fi

# Check directory
if [ ! -d "scripts" ]; then
    echo "[ERROR] Please run from project root directory"
    exit 1
fi

# Set environment
export PYTHONIOENCODING=utf-8

# Run unified launcher
echo "[INFO] Starting unified launcher..."
echo
$PYTHON_CMD start.py
