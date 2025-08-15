#!/usr/bin/env python3
"""
Enhanced Video Search - Easy Setup Script
==========================================
One-command setup for all platforms
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    print("🚀" + "=" * 58 + "🚀")
    print("    Enhanced Video Search - Easy Setup")
    print("🚀" + "=" * 58 + "🚀")
    print(f"📍 Platform: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print()

def check_python():
    """Check Python version"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        print("Please upgrade Python and try again.")
        return False
    return True

def setup_virtual_environment():
    """Setup virtual environment"""
    venv_path = Path('.venv')
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    print("📦 Creating virtual environment...")
    try:
        subprocess.run([sys.executable, '-m', 'venv', '.venv'], check=True)
        print("✅ Virtual environment created")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to create virtual environment")
        return False

def activate_and_install():
    """Install basic dependencies"""
    print("🔧 Installing basic dependencies...")
    
    # Determine activation script
    if platform.system() == "Windows":
        activate_script = Path('.venv/Scripts/python.exe')
    else:
        activate_script = Path('.venv/bin/python')
    
    if not activate_script.exists():
        print("❌ Virtual environment activation failed")
        return False
    
    # Install basic packages
    basic_packages = [
        'fastapi', 'uvicorn[standard]', 'pandas', 
        'numpy', 'pillow', 'python-multipart'
    ]
    
    try:
        for package in basic_packages:
            print(f"  → Installing {package}...")
            subprocess.run([str(activate_script), '-m', 'pip', 'install', package], 
                         check=True, capture_output=True)
        print("✅ Basic dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_launcher_shortcuts():
    """Create platform-specific shortcuts"""
    print("🔗 Creating launcher shortcuts...")
    
    shortcuts_created = []
    
    # Cross-platform shortcuts
    shortcuts = {
        'Windows': ['start.bat', 'start.ps1'],
        'Linux': ['start.sh'],
        'Darwin': ['start.sh']  # macOS
    }
    
    current_platform = platform.system()
    for shortcut in shortcuts.get(current_platform, ['start.sh']):
        if Path(shortcut).exists():
            shortcuts_created.append(shortcut)
    
    if shortcuts_created:
        print(f"✅ Platform shortcuts: {', '.join(shortcuts_created)}")
    else:
        print("⚠️  No platform-specific shortcuts found, use: python start.py")
    
    return True

def show_usage_instructions():
    """Show platform-specific usage instructions"""
    print("\n🎯 USAGE INSTRUCTIONS:")
    print("=" * 50)
    
    current_platform = platform.system()
    
    if current_platform == "Windows":
        print("🪟 Windows:")
        print("  Option 1: start.bat")
        print("  Option 2: powershell -ExecutionPolicy Bypass -File start.ps1")
        print("  Option 3: python start.py")
    elif current_platform == "Darwin":  # macOS
        print("🍎 macOS:")
        print("  chmod +x start.sh && ./start.sh")
        print("  OR: python3 start.py")
    else:  # Linux
        print("🐧 Linux:")
        print("  chmod +x start.sh && ./start.sh") 
        print("  OR: python3 start.py")
    
    print("\n🔥 FEATURES:")
    print("• Smart Auto-Install: Automatically installs TensorFlow when needed")
    print("• Mode Switching: Simple Mode ↔ Full Mode (AI)")
    print("• Cross-Platform: Works on Windows, Linux, macOS")
    print("• API Server: FastAPI with auto-documentation")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Run the launcher using one of the methods above")
    print("2. Choose option 2 to install TensorFlow for AI features")
    print("3. Enjoy intelligent video search!")

def main():
    print_banner()
    
    # Pre-flight checks
    if not check_python():
        return 1
    
    # Setup steps
    steps = [
        ("Checking Python version", lambda: True),
        ("Setting up virtual environment", setup_virtual_environment),
        ("Installing dependencies", activate_and_install),
        ("Creating launcher shortcuts", create_launcher_shortcuts),
    ]
    
    for step_name, step_func in steps:
        print(f"⚙️  {step_name}...")
        if not step_func():
            print(f"❌ Setup failed at: {step_name}")
            return 1
    
    print("\n🎉 SETUP COMPLETED SUCCESSFULLY!")
    show_usage_instructions()
    
    return 0

if __name__ == "__main__":
    exit(main())
