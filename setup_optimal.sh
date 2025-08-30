#!/bin/bash
# Enhanced AI Video Search - Optimal Setup Script
# Automatically creates venv with Python 3.10 for best compatibility

echo "===================================================================="
echo "   Enhanced AI Video Search - Optimal Python Setup"
echo "===================================================================="
echo

# Check if Python 3.10 is available
echo "Checking for Python 3.10..."

if command -v python3.10 &> /dev/null; then
    echo "✅ Python 3.10 found!"
    python3.10 --version
    PYTHON_CMD="python3.10"
elif command -v python3 &> /dev/null && python3 --version | grep -q "3.10"; then
    echo "✅ Python 3.10 found!"
    python3 --version
    PYTHON_CMD="python3"
else
    echo "⚠️ Python 3.10 not found!"
    echo
    echo "📋 RECOMMENDATION: Install Python 3.10 for best AI compatibility"
    echo
    echo "🔗 Download Python 3.10.11 from:"
    echo "   https://www.python.org/downloads/release/python-31011/"
    echo
    echo "📦 Or install via package manager:"
    echo "   Ubuntu/Debian: sudo apt install python3.10 python3.10-venv"
    echo "   CentOS/RHEL:   sudo yum install python3.10"
    echo "   macOS:         brew install python@3.10"
    echo
    echo "⚡ Or continue with current Python (may have compatibility issues)"
    if command -v python3 &> /dev/null; then
        python3 --version
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        python --version
        PYTHON_CMD="python"
    else
        echo "❌ No Python found!"
        exit 1
    fi
    
    read -p "Continue with current Python? (y/n): " choice
    if [[ "$choice" != "y" && "$choice" != "Y" ]]; then
        echo
        echo "👋 Please install Python 3.10 and run this script again"
        echo "🔗 https://www.python.org/downloads/release/python-31011/"
        exit 1
    fi
    
    echo
    echo "⚠️ Using current Python - some AI features may not work"
fi

echo
echo "📦 Creating virtual environment with optimal Python..."

# Remove old venv if exists
if [ -d ".venv" ]; then
    echo "🔄 Removing existing virtual environment..."
    rm -rf .venv
fi

# Create new venv with Python 3.10
echo "🚀 Creating new virtual environment..."
$PYTHON_CMD -m venv .venv

if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment"
    echo "💡 Try installing python3-venv: sudo apt install python3-venv"
    exit 1
fi

echo "✅ Virtual environment created successfully!"
echo

# Activate venv
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Verify Python version in venv
echo "🐍 Python version in venv:"
python --version

echo
echo "📦 Installing dependencies..."
echo "This may take several minutes..."
echo

# Run setup
python setup.py

if [ $? -eq 0 ]; then
    echo
    echo "🎉 SETUP COMPLETE!"
    echo "===================================================================="
    echo
    echo "🚀 Next steps:"
    echo "   1. Launch the system: python main_launcher.py"
    echo "   2. Choose option 1 for Full AI experience"
    echo "   3. Or choose option 2 for Lite version"
    echo
    echo "🔑 Optional: Configure API keys in .env file"
    echo "   cp .env.example .env"
    echo
    echo "📚 Full documentation: README.md"
    echo "===================================================================="
else
    echo
    echo "⚠️ Setup completed with some issues"
    echo "💡 Try running: python main_launcher.py"
    echo "   Choose option 3 to auto-install missing dependencies"
fi

echo
read -p "👉 Press Enter to launch the system..."

# Launch the system
python main_launcher.py
