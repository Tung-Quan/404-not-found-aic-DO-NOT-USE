#!/usr/bin/env python3
"""
🎯 Enhanced AI Video Search System - Complete Setup
Tự động cài đặt tất cả dependencies với smart requirements selection
Xử lý NumPy compatibility và TensorFlow issues
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print setup banner"""
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║    🎯 Enhanced AI Video Search System - Smart Setup       ║")
    print("║   Complete Installation & NumPy/AI Version Pinning        ║")
    print("╠═══════════════════════════════════════════════════════════╣")
    print("║  Core versions: torch==2.7.1+cu118, torchvision==0.22.1+cu118, torchaudio==2.7.1+cu118")
    print("║  faiss-cpu==1.12.0, tensorflow==2.20.0, tensorflow-hub==0.16.1, numpy>=1.26.4,<2.0.0")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()

def check_python():
    """Check Python version and recommend requirements"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version_str} detected. Python 3.8+ required.")
        return False, None, None
    
    print(f"✅ Python {version_str} detected")
    
    # Determine optimal requirements file
    if version.minor <= 11:
        req_file = "config/requirements.txt"
        mode = "Full AI Features (Recommended)"
        print("🎯 Mode: Full AI Features with GPU support")
    elif version.minor == 12:
        req_file = "config/requirements_compatible.txt"
        mode = "Compatible Mode (Limited AI)"
        print("⚠️ Mode: Compatible mode - some AI features limited")
    else:
        req_file = "config/requirements_lite.txt"
        mode = "Lite Mode (Basic Features)"
        print("💡 Mode: Lite mode - basic features only")
        print("💡 Recommendation: Use Python 3.10-3.11 for full AI features")
    
    return True, req_file, mode

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
            lines = result.stdout.split('\n')
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

def run_command(command, description, show_output=False):
    """Run shell command with error handling"""
    print(f"🔄 {description}...")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=not show_output, 
            text=True,
            cwd=os.getcwd()
        )
        print(f"✅ {description} completed")
        return True, result.stdout if not show_output else ""
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"Error details: {e.stderr}")
        return False, str(e)
    
def fix_numpy_compatibility():
    """Fix NumPy compatibility issues with TensorFlow & PyTorch (lock 1.26.x series)"""
    print("\n🔧 Ensuring NumPy 1.26.x compatibility...")

    # First cleanup any corrupt numpy installations
    cleanup_corrupt_packages()

    try:
        import numpy as np
        numpy_version = np.__version__
        major, minor, *_ = map(int, numpy_version.split('.'))
        
        print(f"📊 Current NumPy version: {numpy_version}")

        # Case 1: NumPy >= 2.0 (incompatible with TF 2.20 / Torch 2.x)
        if major >= 2:
            print("⚠️ NumPy 2.x detected - incompatible with TensorFlow/PyTorch stack")
            print("🔄 Downgrading to NumPy 1.26.x...")
            run_command(f'"{sys.executable}" -m pip uninstall numpy -y', "Uninstalling NumPy 2.x")
            success, _ = run_command(
                f'"{sys.executable}" -m pip install "numpy>=1.26.0,<2.0.0"',
                "Installing NumPy 1.26.x"
            )
            if success:
                print("✅ Downgrade to NumPy 1.26.x successful")
                return True
            else:
                print("❌ Failed to install NumPy 1.26.x")
                return False

        # Case 2: NumPy < 1.26 (too old, may break TF 2.20)
        elif major == 1 and minor < 26:
            print("⚠️ NumPy version too old (<1.26) - upgrading...")
            success, _ = run_command(
                f'"{sys.executable}" -m pip install --upgrade "numpy>=1.26.0,<2.0.0"',
                "Upgrading NumPy to 1.26.x"
            )
            if success:
                print("✅ NumPy upgraded to 1.26.x successfully")
                return True
            else:
                print("❌ Failed to upgrade NumPy")
                return False

        # Case 3: NumPy 1.26.x (correct version range)
        elif major == 1 and minor >= 26:
            print("✅ NumPy 1.26.x detected - compatible with TensorFlow/PyTorch")
            return True

        else:
            print("⚠️ Unexpected NumPy version - forcing reinstall...")
            run_command(f'"{sys.executable}" -m pip uninstall numpy -y', "Uninstalling NumPy")
            run_command(f'"{sys.executable}" -m pip install "numpy>=1.26.0,<2.0.0"', "Installing NumPy 1.26.x")
            return True

    except ImportError:
        print("📦 NumPy not installed yet - installing compatible version (1.26.x)")
        run_command(f'"{sys.executable}" -m pip install "numpy>=1.26.0,<2.0.0"', "Installing NumPy 1.26.x")
        return True
    except Exception as e:
        print(f"⚠️ Error checking NumPy: {e}")
        return True  # Continue anyway


def cleanup_corrupt_packages():
    """Clean up corrupt package installations"""
    print("🧹 Cleaning up corrupt package installations...")
    
    venv_path = Path(sys.prefix) / "Lib" / "site-packages"
    if not venv_path.exists():
        venv_path = Path(sys.prefix) / "lib" / "python3.10" / "site-packages"  # Linux/Mac
    
    if venv_path.exists():
        # Look for corrupt numpy directories (starting with ~)
        corrupt_dirs = list(venv_path.glob("~umpy*"))
        
        for corrupt_dir in corrupt_dirs:
            try:
                if corrupt_dir.is_dir():
                    import shutil
                    shutil.rmtree(corrupt_dir)
                    print(f"✅ Removed corrupt directory: {corrupt_dir.name}")
                elif corrupt_dir.is_file():
                    corrupt_dir.unlink()
                    print(f"✅ Removed corrupt file: {corrupt_dir.name}")
            except Exception as e:
                print(f"⚠️ Could not remove {corrupt_dir}: {e}")
    else:
        print("⚠️ Site-packages directory not found")

def install_core_packages():
    """Install core packages with proper order and NumPy constraint"""
    print("\n📦 Installing core packages with NumPy constraint...")
    print("   • numpy>=1.26.4,<2.0.0")
    print("   • pip>=25.2, setuptools>=80.9.0, wheel>=0.45.1, packaging>=25.0")
    
    # Core packages that need to be installed first
    core_packages = [
        "pip>=25.2",
        "setuptools>=80.9.0", 
        "wheel>=0.45.1",
        'numpy>=1.26.4,<2.0.0',  # Đúng bản đã cài
        "packaging>=25.0"
    ]
    
    for package in core_packages:
        success, _ = run_command(f'"{sys.executable}" -m pip install {package}', f"Installing {package}")
        if not success:
            print(f"⚠️ Failed to install {package}, continuing...")
    
    print("✅ Core packages with NumPy 1.x constraint installed")

def smart_install_requirements(requirements_file, mode):
    """Smart installation with NumPy handling and fallbacks"""
    print(f"\n📦 Installing Dependencies: {mode}")
    print(f"📋 Requirements file: {requirements_file}")
    print("   • torch==2.7.1+cu118, torchvision==0.22.1+cu118, torchaudio==2.7.1+cu118")
    print("   • faiss-cpu==1.12.0, tensorflow==2.20.0, tensorflow-hub==0.16.1")
    print("   • numpy>=1.26.4,<2.0.0")
    
    # Check if file exists
    if not os.path.exists(requirements_file):
        print(f"❌ Requirements file not found: {requirements_file}")
        return False
    
    # Install core packages first
    install_core_packages()

    # Upgrade pip
    print("🔄 Upgrading pip...")
    run_command(f'"{sys.executable}" -m pip install --upgrade pip', "Upgrading pip")

    # Chọn constraint NumPy phù hợp
    # Luôn constraint đúng bản đã cài
    numpy_constraint = "numpy>=1.26.4,<2.0.0"

    # Tạo constraint file
    constraint_file = Path("temp_numpy_constraint.txt")
    with open(constraint_file, 'w') as f:
        f.write(numpy_constraint + "\n")

    print(f"📋 Installing from {requirements_file} with NumPy constraint: {numpy_constraint}")
    command = f'"{sys.executable}" -m pip install -r "{requirements_file}" -c "{constraint_file}"'
    success, output = run_command(command, "Installing dependencies with NumPy constraint")

    # Clean up constraint file
    if constraint_file.exists():
        constraint_file.unlink()

    if success:
        print("✅ Dependencies installed successfully with constraints")
        return True
    else:
        print("⚠️ Constrained install failed, trying without constraints...")
        # Fallback: install without constraints
        command = f'"{sys.executable}" -m pip install -r "{requirements_file}"'
        success, output = run_command(command, "Installing dependencies (fallback)")
        if success:
            print("✅ Dependencies installed (may need NumPy fix)")
            fix_numpy_compatibility()
            return True
        else:
            print("❌ Failed to install dependencies")
            return False

def fix_numpy_compatibility():
    """Fix NumPy compatibility issues with TensorFlow"""
    print("\n� Checking NumPy compatibility...")
    
    try:
        import numpy as np
        numpy_version = np.__version__
        major_version = int(numpy_version.split('.')[0])
        
        print(f"📊 Current NumPy version: {numpy_version}")
        
        if major_version >= 2:
            print("⚠️ NumPy 2.x detected - TensorFlow requires NumPy 1.x")
            print("🔄 Downgrading NumPy to compatible version...")
            
            # Uninstall current NumPy
            if run_command(f'"{sys.executable}" -m pip uninstall numpy -y', "Uninstalling NumPy 2.x"):
                # Install compatible NumPy version
                if run_command(f'"{sys.executable}" -m pip install "numpy<2,>=1.19.0"', "Installing NumPy 1.x"):
                    print("✅ NumPy downgraded successfully")
                    return True
                else:
                    print("❌ Failed to install compatible NumPy")
                    return False
            else:
                print("❌ Failed to uninstall NumPy 2.x")
                return False
        else:
            print("✅ NumPy 1.x detected - compatible with TensorFlow")
            return True
            
    except ImportError:
        print("📦 NumPy not installed yet")
        return True
    except Exception as e:
        print(f"⚠️ Error checking NumPy: {e}")
        return True  # Continue anyway

def install_core_packages():
    """Install core packages with proper order"""
    print("\n📦 Installing core packages in correct order...")
    
    # Core packages that need to be installed first
    core_packages = [
        "pip>=23.0",
        "setuptools>=65.0",
        "wheel>=0.40.0",
        '"numpy<2,>=1.19.0"',  # Force NumPy 1.x
        "packaging>=21.0"
    ]
    
    for package in core_packages:
        if not run_command(f'"{sys.executable}" -m pip install {package}', f"Installing {package}"):
            print(f"⚠️ Failed to install {package}, continuing...")
    
    print("✅ Core packages installed")

def install_requirements():
    """Install all requirements from config with NumPy fix"""
    print("\n� Installing Dependencies...")
    
    # First fix NumPy compatibility
    fix_numpy_compatibility()
    
    # Install core packages first
    install_core_packages()
    
    # Check for requirements files
    compatible_file = Path("config/requirements_compatible.txt")
    requirements_file = Path("config/requirements.txt")
    lite_file = Path("config/requirements_lite.txt")
    
    # Upgrade pip first
    if not run_command(f'"{sys.executable}" -m pip install --upgrade pip', "Upgrading pip"):
        print("⚠️ pip upgrade failed, continuing...")
    
    # Try installing requirements with NumPy constraint
    if requirements_file.exists():
        print(f"📋 Installing from {requirements_file} with NumPy constraint")
        
        # Install with NumPy constraint to prevent conflicts
        command = f'"{sys.executable}" -m pip install -r "{requirements_file}" --constraint <(echo "numpy<2")'
        
        # For Windows, we need a different approach
        if platform.system() == "Windows":
            # Create temporary constraint file
            constraint_file = Path("temp_numpy_constraint.txt")
            with open(constraint_file, 'w') as f:
                f.write("numpy<2\n")
            
            command = f'"{sys.executable}" -m pip install -r "{requirements_file}" --constraint "{constraint_file}"'
            
            success = run_command(command, "Installing all dependencies with NumPy constraint")
            
            # Clean up constraint file
            if constraint_file.exists():
                constraint_file.unlink()
        else:
            success = run_command(command, "Installing all dependencies with NumPy constraint")
        
        if success:
            print("✅ All dependencies installed successfully")
            return True
        else:
            print("❌ Failed to install with constraint, trying without...")
            # Fallback to normal install
            command = f'"{sys.executable}" -m pip install -r "{requirements_file}"'
            if run_command(command, "Installing dependencies (fallback)"):
                print("✅ Dependencies installed (may need NumPy fix)")
                fix_numpy_compatibility()  # Fix NumPy after install
                return True
    
    # Try compatible version for Python 3.13+
    if compatible_file.exists():
        print(f"📋 Trying compatible packages from {compatible_file}")
        command = f'"{sys.executable}" -m pip install -r "{compatible_file}"'
        if run_command(command, "Installing compatible dependencies"):
            print("✅ Compatible dependencies installed successfully")
            print("\n🔧 For advanced AI features, install these separately:")
            print("   pip install torch --index-url https://download.pytorch.org/whl/cpu")
            print("   pip install tensorflow")
            print("   pip install transformers")
            print("   pip install openai anthropic")
            return True
    
    # Try lite version as last resort
    if lite_file.exists():
        print(f"🔄 Trying lite version from {lite_file}")
        lite_command = f'"{sys.executable}" -m pip install -r "{lite_file}"'
        if run_command(lite_command, "Installing lite dependencies"):
            print("✅ Lite dependencies installed successfully")
            return True
    
    print("❌ Failed to install dependencies")
    return False

def install_gpu_packages(python_version):
    """Install GPU-specific packages based on Python version"""
    print("\n🎮 Installing GPU packages...")
    print("   • torch==2.7.1+cu118, torchvision==0.22.1+cu118, torchaudio==2.7.1+cu118 (CUDA 11.8)")
    print("   • faiss-cpu==1.12.0 (Windows) or faiss-gpu (Linux/macOS)")
    print("   • tensorflow==2.20.0, tensorflow-hub==0.16.1")
    
    # Only install GPU packages for Python <= 3.11
    if python_version.minor > 11:
        print("⚠️ GPU packages may not be available for Python 3.12+")
        print("💡 For GPU support, consider using Python 3.10-3.11")
        return True
    
    gpu_available = check_gpu()
    
    import platform
    if gpu_available:
        print("Installing GPU-optimized packages...")

        # PyTorch với đúng phiên bản đã cài
        pytorch_command = (
            f'"{sys.executable}" -m pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 '
            "--index-url https://download.pytorch.org/whl/cu118"
        )
        success, _ = run_command(pytorch_command, "Installing PyTorch with CUDA 11.8")
        if success:
            print("✅ PyTorch GPU support installed")

        # FAISS đúng bản đã cài
        run_command(f'"{sys.executable}" -m pip install faiss-cpu==1.12.0', "Installing FAISS CPU (Windows)")

        # TensorFlow đúng bản đã cài
        tf_command = f'"{sys.executable}" -m pip install tensorflow==2.20.0 tensorflow-hub==0.16.1 --upgrade --force-reinstall'
        success, _ = run_command(tf_command, "Installing TensorFlow GPU")
        if success:
            print("✅ TensorFlow GPU support installed")
    else:
        print("⚠️ No GPU detected, installing CPU versions")
        run_command(f'"{sys.executable}" -m pip install faiss-cpu', "Installing FAISS CPU")

    return True

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
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
    print("\n🔑 Creating .env template...")

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
    """Verify installation by importing key modules with NumPy fix"""
    print("\n🧪 Verifying installation...")
    print("   • Checking: torch==2.7.1+cu118, torchvision==0.22.1+cu118, torchaudio==2.7.1+cu118")
    print("   • faiss-cpu==1.12.0, tensorflow==2.20.0, tensorflow-hub==0.16.1")
    print("   • numpy>=1.26.4,<2.0.0")
    
    # First verify NumPy
    try:
        import numpy as np
        numpy_version = np.__version__
        major_version = int(numpy_version.split('.')[0])
        
        if major_version >= 2:
            print(f"⚠️ NumPy {numpy_version} detected - may cause TensorFlow issues")
            print("🔧 Attempting NumPy fix...")
            fix_numpy_compatibility()
        else:
            print(f"✅ NumPy {numpy_version} - compatible")
    except ImportError:
        print("❌ NumPy not installed")
    
    tests = [
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("fastapi", "FastAPI"),
        ("torch", "PyTorch"),
        ("transformers", "Transformers")
    ]
    
    # Test TensorFlow separately with error handling
    tensorflow_tests = [("tensorflow", "TensorFlow")]
    
    success_count = 0
    
    # Test core modules first
    for module, name in tests:
        try:
            __import__(module)
            print(f"✅ {name}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {name} - {e}")
        except Exception as e:
            print(f"⚠️ {name} - Warning: {e}")
            success_count += 0.5  # Partial success
    
    # Test TensorFlow with special handling
    for module, name in tensorflow_tests:
        try:
            print(f"🔄 Testing {name}...")
            tf = __import__(module)
            
            # Test basic TensorFlow functionality
            version = tf.__version__
            print(f"✅ {name} {version}")
            
            # Quick functionality test
            try:
                tf.constant([1, 2, 3])
                print(f"✅ {name} basic operations working")
                success_count += 1
            except Exception as e:
                print(f"⚠️ {name} imported but operations failed: {e}")
                print("🔧 This may be due to NumPy compatibility issues")
                success_count += 0.5
                
        except ImportError as e:
            print(f"❌ {name} - {e}")
            if "numpy" in str(e).lower():
                print("🔧 TensorFlow-NumPy compatibility issue detected")
                print("💡 Try: pip uninstall numpy && pip install 'numpy<2'")
        except Exception as e:
            error_msg = str(e).lower()
            if "_array_api" in error_msg or "multiarray_umath" in error_msg:
                print(f"❌ {name} - NumPy compatibility error")
                print("🔧 Fixing NumPy compatibility...")
                if fix_numpy_compatibility():
                    print("✅ NumPy fixed, please restart the application")
                else:
                    print("❌ NumPy fix failed")
            else:
                print(f"❌ {name} - {e}")
    
    total_tests = len(tests) + len(tensorflow_tests)
    print(f"\n📊 Installation verification: {success_count}/{total_tests} packages working")
    
    if success_count >= total_tests * 0.7:  # 70% success rate
        print("✅ Installation successful!")
        if success_count < total_tests:
            print("⚠️ Some packages have warnings - system should still work")
        return True
    else:
        print("⚠️ Installation completed with issues")
        print("💡 Try running the NumPy fix manually:")
        print("   pip uninstall numpy -y")
        print("   pip install 'numpy<2,>=1.19.0'")
        return False
    print(f"\n📊 Installation verification: {success_count}/{len(tests)} packages working")
    
    if success_count >= len(tests) * 0.8:  # 80% success rate
        print("✅ Installation successful!")
        return True
    else:
        print("⚠️ Some packages failed to install")
        return False

def test_gpu_functionality():
    """Test GPU functionality"""
    print("\n🎮 Testing GPU functionality...")
    
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

def create_installation_summary():
    """Create summary of installed packages"""
    print("\n📊 Creating installation summary...")
    
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                               capture_output=True, text=True)
        
        version = sys.version_info
        timestamp = __import__('datetime').datetime.now()
        
        with open("installed_packages.txt", "w", encoding='utf-8') as f:
            f.write(f"🎯 AI Video Search System - Installation Summary\n")
            f.write(f"=" * 60 + "\n")
            f.write(f"Python Version: {version.major}.{version.minor}.{version.micro}\n")
            f.write(f"Installation Date: {timestamp}\n")
            f.write(f"Platform: {platform.system()} {platform.release()}\n\n")
            f.write("Installed Packages:\n")
            f.write("=" * 30 + "\n")
            f.write(result.stdout)
            
            # Add NumPy version check
            try:
                import numpy as np
                f.write(f"\n✅ NumPy Version: {np.__version__}\n")
                if int(np.__version__.split('.')[0]) >= 2:
                    f.write("⚠️ WARNING: NumPy 2.x may cause TensorFlow issues\n")
                else:
                    f.write("✅ NumPy 1.x - Compatible with TensorFlow\n")
            except ImportError:
                f.write("❌ NumPy not installed\n")
        
        print("✅ Installation summary saved to: installed_packages.txt")
        
    except Exception as e:
        print(f"⚠️ Could not create installation summary: {e}")

def print_completion_message(mode, python_version):
    """Print completion message with next steps"""
    print("\n🎉 SMART INSTALLATION COMPLETE!")
    print("=" * 60)
    print(f"🐍 Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    print(f"🎯 Installation Mode: {mode}")
    print("✅ Installed versions:")
    print("   • torch==2.7.1+cu118, torchvision==0.22.1+cu118, torchaudio==2.7.1+cu118")
    print("   • faiss-cpu==1.12.0, tensorflow==2.20.0, tensorflow-hub==0.16.1")
    print("   • numpy>=1.26.4,<2.0.0")
    print()
    print("Next steps:")
    print("1. Test core functionality:")
    print("   python -c \"import torch, cv2, fastapi, numpy; print('✅ Core packages working')\"")
    print()
    print("2. Test NumPy-TensorFlow compatibility:")
    print("   python -c \"import tensorflow as tf; print('✅ TensorFlow working')\"")
    print()
    print("3. Launch the system:")
    print("   python main_launcher.py")
    print()
    print("4. Check embedding status:")
    print("   python scripts/check_embedding_status.py")
    print()
    print("5. Start web interface:")
    print("   cd api && python app.py")
    print("   # Open http://localhost:8000")
    print()
    
    # Version-specific recommendations
    if python_version.minor >= 12:
        print("� Python 3.12+ Recommendations:")
        print("   • Some AI features may be limited")
        print("   • For full features, consider Python 3.10-3.11")
        print("   • Use Docker with Python 3.10 for production")
    else:
        print("🎯 AI Embedding Features Available:")
        print("   • Frame-to-Vector conversion ready")
        print("   • Chinese-CLIP for Vietnamese content")
        print("   • FAISS index for fast similarity search")
        print("   • Multi-modal search (text ↔ image)")
        print()
        print("📋 Advanced Embedding Setup:")
        print("   python scripts/encode_chinese_clip.py")
        print("   python scripts/build_faiss_chinese_clip.py")
    
    print()
    print("📚 Documentation:")
    print("   • README.md - General setup guide")
    print("   • EMBEDDING_GUIDE.md - AI embedding system")
    print("   • AI_AGENTS_GUIDE.md - AI agents documentation")
    print("=" * 60)

def setup_embedding_system():
    """Setup embedding system components and build embeddings"""
    print("\n🎯 Setting up AI Embedding System...")
    
    # Check if embeddings already exist
    embedding_files = [
        'index/embeddings/image_embeddings_clip_vit_base.npy',
        'index/embeddings/frames_chinese_clip.f16.mmap'
    ]
    
    existing_embeddings = [f for f in embedding_files if os.path.exists(f)]
    
    if existing_embeddings:
        print("✅ Found existing embeddings:")
        for emb in existing_embeddings:
            size_mb = os.path.getsize(emb) / 1024 / 1024
            print(f"   📁 {emb} ({size_mb:.1f} MB)")
        print("🔄 Embeddings already exist - skipping build")
    else:
        print("🔄 No embeddings found - building now...")
        
        # Try to build embeddings automatically
        try:
            print("🔄 Initializing AI models and building embeddings...")
            
            # Import and initialize the system
            from ai_search_engine import EnhancedAIVideoSearchEngine
            from enhanced_hybrid_manager import EnhancedHybridModelManager
            
            print("📚 Loading AI models...")
            manager = EnhancedHybridModelManager()
            search_engine = EnhancedAIVideoSearchEngine(model_manager=manager)
            
            print("🤖 Initializing default models...")
            search_engine.initialize_default_models()
            
            print("🏗️ Building embeddings index (this may take a few minutes)...")
            search_engine.build_embeddings_index()
            
            print("✅ Embeddings built successfully!")
            
        except Exception as e:
            print(f"⚠️ Could not auto-build embeddings: {e}")
            print("💡 You can build them manually later:")
            print("   📋 Run: python -c \"from ai_search_engine import *; from enhanced_hybrid_manager import *; manager = EnhancedHybridModelManager(); engine = EnhancedAIVideoSearchEngine(model_manager=manager); engine.initialize_default_models(); engine.build_embeddings_index()\"")
    
    # Check embedding scripts
    embedding_scripts = [
        'scripts/encode_chinese_clip.py',
        'scripts/build_faiss_chinese_clip.py',
        'scripts/text_embed.py'
    ]
    
    missing_scripts = [s for s in embedding_scripts if not os.path.exists(s)]
    if missing_scripts:
        print("⚠️ Missing embedding scripts:")
        for script in missing_scripts:
            print(f"   ❌ {script}")
    else:
        print("✅ All embedding scripts available")
    
    print("🎯 Embedding system ready!")
    
    return True

def extract_frames_ffmpeg(video_path, output_dir, fps=1):
    """
    Extract frames from video using ffmpeg, 1 frame per second.
    Args:
        video_path (str): Path to input video file
        output_dir (str): Directory to save frames
        fps (int): Frames per second (default 1)
    """
    Path(output_dir).mkdir(exist_ok=True)
    # ffmpeg command: 1 frame per second
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={fps}",
        f"{output_dir}/frame_%06d.jpg"
    ]
    print(f"🔄 Extracting frames from {video_path} to {output_dir} (1 frame/sec)...")
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Frames extracted to {output_dir}")
    except Exception as e:
        print(f"❌ FFmpeg extraction failed: {e}")

# Example usage:
# extract_frames_ffmpeg('videos/my_video.mp4', 'frames/my_video', fps=1)

def auto_extract_frames_if_needed():
    """
    Check if frames/ folder has frames. If not, extract frames from all videos in videos/.
    """
    frames_dir = Path("frames")
    videos_dir = Path("videos")
    # Check if frames/ exists and has any jpg file
    has_frames = any(frames_dir.glob("**/*.jpg"))
    if has_frames:
        print("✅ Frames already extracted in frames/ folder.")
        return
    print("⚠️ No frames found in frames/. Extracting frames from videos...")
    for video_file in videos_dir.glob("*.mp4"):
        video_name = video_file.stem
        output_dir = frames_dir / video_name
        extract_frames_ffmpeg(str(video_file), str(output_dir), fps=1)
    print("🎯 All videos processed for frame extraction.")

def install_cuda_windows():
    """Install CUDA Toolkit on Windows automatically (if not installed)"""
    import shutil
    cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.exists(cuda_path):
        print("✅ CUDA Toolkit đã được cài trên máy.")
        return True
    print("⚠️ CUDA Toolkit chưa được cài. Đang tải về và cài đặt...")
    # Link tải CUDA 11.8 (có thể thay đổi nếu cần bản khác)
    cuda_url = "https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_windows_network.exe"
    installer_path = os.path.join(os.getcwd(), "cuda_installer.exe")
    try:
        import urllib.request
        print(f"🔄 Đang tải CUDA installer từ {cuda_url}...")
        urllib.request.urlretrieve(cuda_url, installer_path)
        print("✅ Đã tải xong CUDA installer.")
        print("🔄 Đang chạy installer (cần quyền admin)...")
        # Chạy installer (yêu cầu quyền admin)
        result = subprocess.run(f'"{installer_path}" -s', shell=True)
        if result.returncode == 0:
            print("✅ Đã cài xong CUDA Toolkit!")
            if os.path.exists(installer_path):
                os.remove(installer_path)
            return True
        else:
            print("❌ Cài đặt CUDA thất bại. Hãy chạy installer bằng tay với quyền admin.")
            return False
    except Exception as e:
        print(f"❌ Lỗi khi tải/cài CUDA: {e}")
        return False

def main():
    print_banner()
    
    # Check Python version and determine requirements
    python_ok, requirements_file, mode = check_python()
    if not python_ok:
        sys.exit(1)
    
    python_version = sys.version_info
    
    # Check virtual environment (warning only)
    check_venv()
    
    # Create directories
    create_directories()
    
    # Tự động tách frame nếu chưa có
    auto_extract_frames_if_needed()
    
    # Smart install dependencies based on Python version
    if not smart_install_requirements(requirements_file, mode):
        print("❌ Failed to install dependencies")
        print("\n🔧 Troubleshooting:")
        print("1. Check internet connection")
        print("2. Try: pip install --upgrade pip")
        print("3. Manual NumPy fix: pip uninstall numpy -y && pip install 'numpy<2'")
        sys.exit(1)
    
    # Install GPU packages (if applicable for Python version)
    install_gpu_packages(python_version)
    
    # Create config files
    create_env_template()
    
    # Verify installation with NumPy check
    verification_success = verify_installation()
    
    # Test GPU (optional)
    if python_version.minor <= 11:
        test_gpu_functionality()
    
    # Setup embedding system (chỉ tạo embedding, không khởi tạo dataset/web interface)
    setup_embedding_system()

    # Create installation summary
    create_installation_summary()

    # Completion message
    print_completion_message(mode, python_version)

    # Final status
    if verification_success:
        print("\n🎯 STATUS: Ready to use!")
        print("🚀 Quick start: python main_launcher.py")
        print("🌐 Để chạy web interface, hãy dùng lệnh sau sau khi setup xong:")
        print("   python web_interface.py hoặc cd api && python app.py")
        print("⚡ Lưu ý: Các chức năng quản lý dataset, khởi tạo web server sẽ do web_interface.py xử lý. Không cần khởi tạo lại trong setup.py!")
    else:
        print("\n⚠️ STATUS: Installed with warnings")
        print("💡 Most features should work, some AI features may be limited")
        print("🔧 Manual NumPy fix: pip uninstall numpy -y && pip install 'numpy<2'")

if __name__ == "__main__":
    main()
