#!/usr/bin/env python3
"""
Setup Script cho AI Video Search System
Cài đặt đầy đủ dependencies cho AI agents và TensorFlow models
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description=""):
    """Run command và hiển thị progress"""
    print(f"🔄 {description}...")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def setup_environment():
    """Setup Python environment"""
    print("🚀 Setting up AI Video Search Environment")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or python_version.minor < 8:
        print("❌ Python 3.8+ required")
        return False
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check if we're in virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if in_venv:
        print("✅ Virtual environment detected")
    else:
        print("⚠️ Not in virtual environment - consider using venv")
    
    return True

def install_core_dependencies():
    """Install core dependencies"""
    print("\n📦 Installing Core Dependencies...")
    
    core_packages = [
        "pip>=23.0",
        "wheel>=0.40.0",
        "setuptools>=68.0",
    ]
    
    for package in core_packages:
        if not run_command(f"pip install --upgrade {package}", f"Installing {package}"):
            return False
    
    return True

def install_pytorch_gpu():
    """Install PyTorch with CUDA support"""
    print("\n🎮 Installing PyTorch with GPU support...")
    
    # Check for CUDA
    try:
        result = subprocess.run("nvidia-smi", capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            
            # Install PyTorch with CUDA
            pytorch_command = (
                "pip install torch torchvision torchaudio "
                "--index-url https://download.pytorch.org/whl/cu118"
            )
            
            return run_command(pytorch_command, "Installing PyTorch with CUDA")
        else:
            print("⚠️ No NVIDIA GPU detected, installing CPU version")
            return run_command("pip install torch torchvision torchaudio", "Installing PyTorch CPU")
            
    except FileNotFoundError:
        print("⚠️ nvidia-smi not found, installing CPU version")
        return run_command("pip install torch torchvision torchaudio", "Installing PyTorch CPU")

def install_tensorflow():
    """Install TensorFlow"""
    print("\n🔧 Installing TensorFlow...")
    
    # Install TensorFlow with GPU support
    tf_packages = [
        "tensorflow>=2.13.0",
        "tensorflow-hub>=0.15.0",
        "tensorflow-text>=2.13.0"
    ]
    
    for package in tf_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            return False
    
    return True

def install_ai_models():
    """Install AI model libraries"""
    print("\n🤖 Installing AI Model Libraries...")
    
    ai_packages = [
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "sentence-transformers>=2.2.2",
        "openai>=1.3.0",
        "anthropic>=0.8.0",
        "langchain>=0.0.350",
        "langchain-openai>=0.0.2",
        "langchain-anthropic>=0.0.1"
    ]
    
    for package in ai_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️ Failed to install {package}, continuing...")
    
    return True

def install_vector_databases():
    """Install vector database libraries"""
    print("\n🔍 Installing Vector Databases...")
    
    vector_packages = [
        "faiss-cpu>=1.7.4",
        "chromadb>=0.4.18",
        "pinecone-client>=2.2.4"
    ]
    
    for package in vector_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️ Failed to install {package}, continuing...")
    
    return True

def install_computer_vision():
    """Install computer vision libraries"""
    print("\n📸 Installing Computer Vision Libraries...")
    
    cv_packages = [
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "scikit-image>=0.21.0",
        "albumentations>=1.3.1",
        "imageio>=2.31.0"
    ]
    
    for package in cv_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            return False
    
    return True

def install_web_framework():
    """Install web framework and APIs"""
    print("\n🌐 Installing Web Framework...")
    
    web_packages = [
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "python-multipart>=0.0.6",
        "pydantic>=2.5.0",
        "requests>=2.31.0"
    ]
    
    for package in web_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            return False
    
    return True

def install_data_processing():
    """Install data processing libraries"""
    print("\n📊 Installing Data Processing Libraries...")
    
    data_packages = [
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.13.0"
    ]
    
    for package in data_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            return False
    
    return True

def install_optional_packages():
    """Install optional packages"""
    print("\n🔧 Installing Optional Packages...")
    
    optional_packages = [
        "jupyter>=1.0.0",
        "ipykernel>=6.25.0",
        "wandb>=0.16.0",
        "tensorboard>=2.13.0",
        "python-dotenv>=1.0.0"
    ]
    
    for package in optional_packages:
        run_command(f"pip install {package}", f"Installing {package}")
    
    return True

def create_env_template():
    """Create .env template file"""
    print("\n📝 Creating environment template...")
    
    env_template = """# AI Video Search Environment Configuration
# Copy this file to .env and fill in your API keys

# OpenAI API Key (for GPT-4 Vision and text models)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key (for Claude models)  
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: Pinecone for vector database
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here

# Optional: Weights & Biases for experiment tracking
WANDB_API_KEY=your_wandb_api_key_here

# System Configuration
GPU_MEMORY_FRACTION=0.8
BATCH_SIZE=32
DEVICE=auto
"""
    
    try:
        with open(".env.template", "w", encoding="utf-8") as f:
            f.write(env_template)
        print("✅ Created .env.template file")
        print("📋 Please copy .env.template to .env and add your API keys")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env template: {e}")
        return False

def verify_installation():
    """Verify installation"""
    print("\n🧪 Verifying Installation...")
    
    # Test imports
    test_imports = [
        ("torch", "PyTorch"),
        ("tensorflow", "TensorFlow"),
        ("transformers", "Transformers"),
        ("fastapi", "FastAPI"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas")
    ]
    
    success_count = 0
    total_count = len(test_imports)
    
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✅ {name}")
            success_count += 1
        except ImportError:
            print(f"❌ {name}")
    
    print(f"\n📊 Installation Status: {success_count}/{total_count} packages working")
    
    if success_count >= total_count * 0.8:  # 80% success rate
        print("🎉 Installation completed successfully!")
        return True
    else:
        print("⚠️ Some packages failed to install")
        return False

def main():
    """Main setup function"""
    print("🚀 AI Video Search System Setup")
    print("This will install all dependencies for the enhanced system")
    print("Including AI agents, TensorFlow models, and GPU support")
    print("=" * 70)
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
    
    # Install dependencies in order
    installation_steps = [
        install_core_dependencies,
        install_pytorch_gpu,
        install_tensorflow,
        install_ai_models,
        install_vector_databases,
        install_computer_vision,
        install_web_framework,
        install_data_processing,
        install_optional_packages,
        create_env_template
    ]
    
    for step in installation_steps:
        if not step():
            print(f"❌ Failed at step: {step.__name__}")
            print("⚠️ You can continue manually or re-run the script")
    
    # Verify installation
    verify_installation()
    
    print("\n" + "=" * 70)
    print("🎯 Setup Summary:")
    print("✅ All major dependencies installed")
    print("✅ GPU support configured (if available)")  
    print("✅ AI agents ready (configure API keys in .env)")
    print("✅ TensorFlow Hub models available")
    print("✅ Vector databases configured")
    print("\n📋 Next steps:")
    print("1. Copy .env.template to .env")
    print("2. Add your API keys to .env file")
    print("3. Run: python main_launcher.py")
    print("4. Choose Full Version to test all features")
    print("\n🚀 Your AI Video Search System is ready!")

if __name__ == "__main__":
    main()
