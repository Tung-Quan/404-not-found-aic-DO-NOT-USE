#!/usr/bin/env python3
"""
Enhanced AI Video Search System - Complete Setup
Tự động cài đặt tất cả dependencies từ config/requirements.txt
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print setup banner"""
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║        Enhanced AI Video Search System - Setup            ║")
    print("║              Complete Installation                        ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()

def check_python():
    """Check Python version"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
    return True

def check_venv():
    """Check if running in virtual environment"""
    print("\n🔧 Checking virtual environment...")
    
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
        return True
    else:
        print("⚠️ Not in virtual environment")
        print("Recommendation: Create virtual environment first:")
        print("  python -m venv .venv")
        print("  .venv\\Scripts\\activate  # Windows")
        print("  source .venv/bin/activate  # Linux/macOS")
        return False

def check_gpu():
    """Check GPU availability"""
    print("\n🎮 Checking GPU...")
    
    try:
        result = subprocess.run("nvidia-smi", capture_output=True, text=True)
        if result.returncode == 0:
            # Extract GPU info
            lines = result.stdout.split('\\n')
            for line in lines:
                if 'GeForce' in line or 'RTX' in line or 'GTX' in line:
                    gpu_info = line.split('|')[1].strip() if '|' in line else "GPU detected"
                    print(f"✅ {gpu_info}")
                    return True
            print("✅ NVIDIA GPU detected")
            return True
        else:
            print("❌ No NVIDIA GPU detected")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
        return False

def run_command(command, description):
    """Run shell command with error handling"""
    print(f"🔄 {description}...")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True,
            cwd=os.getcwd()  # Ensure correct working directory
        )
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def install_requirements():
    """Install all requirements from config"""
    print("\\n📦 Installing Dependencies...")
    
    # Check for Python 3.13 compatible requirements first
    compatible_file = Path("config/requirements_compatible.txt")
    requirements_file = Path("config/requirements.txt")
    lite_file = Path("config/requirements_lite.txt")
    
    # Try compatible version first for Python 3.13
    if compatible_file.exists():
        print(f"📋 Installing Python 3.13 compatible packages from {compatible_file}")
        
        # Upgrade pip first
        if not run_command(f'"{sys.executable}" -m pip install --upgrade pip', "Upgrading pip"):
            print("⚠️ pip upgrade failed, continuing...")
        
        # Install compatible requirements
        command = f'"{sys.executable}" -m pip install -r "{compatible_file}" --upgrade'
        if run_command(command, "Installing compatible dependencies"):
            print("✅ Compatible dependencies installed successfully")
            print("\\n🔧 For advanced AI features, install these separately:")
            print("   pip install torch --index-url https://download.pytorch.org/whl/cpu")
            print("   pip install tensorflow")
            print("   pip install transformers")
            print("   pip install openai anthropic")
            return True
        else:
            print("❌ Failed to install compatible requirements")
    
    # Fallback to original requirements
    if not requirements_file.exists():
        print("❌ config/requirements.txt not found!")
        return False
    
    print(f"📋 Installing from {requirements_file}")
    
    # Upgrade pip first
    if not run_command(f'"{sys.executable}" -m pip install --upgrade pip', "Upgrading pip"):
        print("⚠️ pip upgrade failed, continuing...")
    
    # Install requirements
    command = f'"{sys.executable}" -m pip install -r "{requirements_file}" --upgrade'
    if run_command(command, "Installing all dependencies"):
        print("✅ All dependencies installed successfully")
        return True
    else:
        print("❌ Failed to install full requirements")
        
        # Try lite version
        if lite_file.exists():
            print(f"🔄 Trying lite version from {lite_file}")
            lite_command = f'"{sys.executable}" -m pip install -r "{lite_file}" --upgrade'
            if run_command(lite_command, "Installing lite dependencies"):
                print("✅ Lite dependencies installed successfully")
                return True
        
        print("❌ Failed to install dependencies")
        return False

def install_gpu_packages():
    """Install GPU-specific packages"""
    print("\\n🎮 Installing GPU packages...")
    
    gpu_available = check_gpu()
    
    if gpu_available:
        print("Installing GPU-optimized packages...")
        
        # PyTorch with CUDA
        pytorch_command = (
            f'"{sys.executable}" -m pip install torch torchvision torchaudio '
            "--index-url https://download.pytorch.org/whl/cu118"
        )
        
        if run_command(pytorch_command, "Installing PyTorch with CUDA"):
            print("✅ PyTorch GPU support installed")
        
        # FAISS GPU
        if run_command(f'"{sys.executable}" -m pip install faiss-gpu', "Installing FAISS GPU"):
            print("✅ FAISS GPU support installed")
            
        # TensorFlow GPU
        if run_command(f'"{sys.executable}" -m pip install tensorflow[and-cuda]', "Installing TensorFlow GPU"):
            print("✅ TensorFlow GPU support installed")
    else:
        print("⚠️ No GPU detected, CPU versions will be used")

def create_directories():
    """Create necessary directories"""
    print("\\n📁 Creating directories...")
    
    directories = [
        "frames",
        "index", 
        "embeddings",
        "models_cache",
        "logs",
        "data",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created {directory}/")

def create_env_template():
    """Create .env template file"""
    print("\\n🔑 Creating .env template...")
    
    env_content = '''# API Keys for AI Agents (Optional)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Model Settings
MODEL_CACHE_DIR=./models_cache
TRANSFORMERS_CACHE=./models_cache

# GPU Settings
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

# Application Settings
DEBUG=False
LOG_LEVEL=INFO
'''
    
    env_file = Path(".env.example")
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ Created .env.example")
        print("🔧 Copy to .env and add your API keys")
    else:
        print("✅ .env.example already exists")

def verify_installation():
    """Verify installation by importing key modules"""
    print("\\n🧪 Verifying installation...")
    
    tests = [
        ("torch", "PyTorch"),
        ("tensorflow", "TensorFlow"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("transformers", "Transformers"),
        ("fastapi", "FastAPI")
    ]
    
    success_count = 0
    for module, name in tests:
        try:
            __import__(module)
            print(f"✅ {name}")
            success_count += 1
        except ImportError:
            print(f"❌ {name}")
    
    print(f"\\n📊 Installation verification: {success_count}/{len(tests)} packages working")
    
    if success_count >= len(tests) * 0.8:  # 80% success rate
        print("✅ Installation successful!")
        return True
    else:
        print("⚠️ Some packages failed to install")
        return False

def test_gpu_functionality():
    """Test GPU functionality"""
    print("\\n🎮 Testing GPU functionality...")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            print(f"✅ GPU: {device_name} ({memory}GB)")
            
            # Test tensor operation
            x = torch.randn(10, 10).cuda()
            y = torch.randn(10, 10).cuda()
            z = torch.matmul(x, y)
            print("✅ GPU tensor operations working")
            
            return True
        else:
            print("❌ CUDA not available in PyTorch")
            return False
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def print_completion_message():
    """Print completion message with next steps"""
    print("\\n🎉 INSTALLATION COMPLETE!")
    print("=" * 60)
    print("Next steps:")
    print("1. Configure API keys in .env file (optional):")
    print("   cp .env.example .env")
    print("   # Edit .env with your API keys")
    print()
    print("2. Launch the system:")
    print("   python main_launcher.py")
    print()
    print("3. Test individual components:")
    print("   python -c \"from enhanced_hybrid_manager import EnhancedHybridModelManager; print('✅ System ready')\"")
    print()
    print("4. Start web interface:")
    print("   cd api && python main.py")
    print("   # Open http://localhost:8000")
    print()
    print("📚 Documentation: README.md")
    print("🚀 Quick start: Just run 'python main_launcher.py'")
    print("=" * 60)

def main():
    """Main setup function"""
    print_banner()
    
    # Basic checks
    if not check_python():
        sys.exit(1)
    
    check_venv()  # Warning only, don't exit
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_requirements():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # GPU packages (optional)
    install_gpu_packages()
    
    # Create config files
    create_env_template()
    
    # Verify installation
    if not verify_installation():
        print("⚠️ Installation completed with warnings")
    
    # Test GPU (optional)
    test_gpu_functionality()
    
    # Completion message
    print_completion_message()

if __name__ == "__main__":
    main()
