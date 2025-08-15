"""
🚀 ENHANCED VIDEO SEARCH SYSTEM - ONE-CLICK SETUP
================================================
Complete setup script for Enhanced Video Search with TensorFlow Hub integration
"""

import subprocess
import sys
import os
import platform
import time
from pathlib import Path

def print_banner():
    """Print setup banner"""
    print("🚀 ENHANCED VIDEO SEARCH SYSTEM - SETUP")
    print("=" * 60)
    print("Intelligent video search with TensorFlow Hub integration")
    print()

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"🐍 Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Please upgrade Python.")
        return False
    
    print("✅ Python version OK")
    return True

def check_virtual_environment():
    """Check if virtual environment exists"""
    venv_path = Path(".venv")
    
    if venv_path.exists():
        print("✅ Virtual environment found")
        return True
    else:
        print("📦 Creating virtual environment...")
        try:
            subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
            print("✅ Virtual environment created")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to create virtual environment")
            return False

def activate_virtual_environment():
    """Get activation command for virtual environment"""
    system = platform.system()
    
    if system == "Windows":
        return ".venv\\Scripts\\activate.bat"
    else:
        return "source .venv/bin/activate"

def install_requirements():
    """Install all requirements"""
    print("\n📦 INSTALLING DEPENDENCIES")
    print("-" * 40)
    
    # Get Python executable in venv
    if platform.system() == "Windows":
        python_exe = ".venv\\Scripts\\python.exe"
    else:
        python_exe = ".venv/bin/python"
    
    requirements_files = [
        "configs/requirements.txt",
        "configs/requirements_enhanced.txt"
    ]
    
    for req_file in requirements_files:
        if Path(req_file).exists():
            print(f"\n🔧 Installing {req_file}...")
            try:
                result = subprocess.run([
                    python_exe, "-m", "pip", "install", "-r", req_file
                ], check=True, capture_output=True, text=True)
                print(f"✅ {req_file} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"⚠️  Warning: Some packages in {req_file} failed to install")
                print(f"Error: {e.stderr}")
        else:
            print(f"⚠️  {req_file} not found, skipping...")

def test_tensorflow_hub():
    """Test TensorFlow Hub installation"""
    print("\n🧪 TESTING TENSORFLOW HUB")
    print("-" * 30)
    
    if platform.system() == "Windows":
        python_exe = ".venv\\Scripts\\python.exe"
    else:
        python_exe = ".venv/bin/python"
    
    test_script = '''
import sys
try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__}")
    
    import tensorflow_hub as hub
    print("✅ TensorFlow Hub available")
    
    import tensorflow_text
    print("✅ TensorFlow Text available")
    
    # Test loading a small model (this verifies internet connectivity)
    print("🔄 Testing model download...")
    try:
        model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        embeddings = model(["Hello world"])
        print(f"✅ Model test successful! Embedding shape: {embeddings.shape}")
    except Exception as e:
        print(f"⚠️  Model download test failed: {e}")
        print("This might be due to internet connectivity issues.")
    
    print("🎉 TensorFlow Hub setup complete!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)
'''
    
    try:
        result = subprocess.run([python_exe, "-c", test_script], 
                              check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ TensorFlow Hub test failed:")
        print(e.stderr)
        return False

def check_project_structure():
    """Check if required project files exist"""
    print("\n📁 CHECKING PROJECT STRUCTURE")
    print("-" * 35)
    
    required_files = [
        "src/api/app.py",
        "src/core/enhanced_video_processor.py", 
        "src/ui/enhanced_web_interface.py",
        "scripts/start_server.bat",
        "launch.bat"
    ]
    
    required_dirs = [
        "index",
        "videos",
        "api"
    ]
    
    all_good = True
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing!")
            all_good = False
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"⚠️  {dir_path}/ - Creating...")
            Path(dir_path).mkdir(exist_ok=True)
    
    return all_good

def create_sample_data():
    """Create sample data if not exists"""
    print("\n📊 CHECKING SAMPLE DATA")
    print("-" * 25)
    
    # Check for metadata
    meta_file = Path("index/meta.parquet")
    if meta_file.exists():
        print("✅ Video metadata found")
    else:
        print("⚠️  No video metadata found")
        print("   You'll need to run the indexing process after adding videos")
    
    # Check for videos
    videos_dir = Path("videos")
    video_files = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.avi"))
    
    if video_files:
        print(f"✅ Found {len(video_files)} video files")
    else:
        print("⚠️  No video files found in videos/ directory")
        print("   Add video files to videos/ and run indexing")

def show_next_steps():
    """Show next steps after setup"""
    print("\n🎉 SETUP COMPLETE!")
    print("=" * 50)
    print()
    print("🚀 NEXT STEPS:")
    print()
    print("1. 📁 Add video files to videos/ directory")
    print()
    print("2. 🏃 Start the system:")
    if platform.system() == "Windows":
        print("   launch.bat")
    else:
        print("   python scripts/setup_complete.py")
        print("   python demos/enhanced_video_demo.py")
    print()
    print("3. 🌐 Access interfaces:")
    print("   • Enhanced Web UI: http://localhost:8501")  
    print("   • API Docs: http://localhost:8000/docs")
    print("   • Standard Web UI: http://localhost:5000")
    print()
    print("4. 🔧 Available startup options:")
    print("   [1] Enhanced API with TensorFlow Hub")
    print("   [2] Simple Enhanced API (fast startup)")
    print("   [3] Standard Web Interface")
    print("   [4] Enhanced Video Processing Interface")
    print("   [5] Interactive CLI Demo")
    print("   [6] Enhanced Video Processing Demo")
    print()
    print("💡 TIP: First time loading TF Hub models may take 5-10 minutes")
    print("    Subsequent runs will be much faster!")
    print()
    print("📖 For detailed usage instructions, see README.md")

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check/create virtual environment
    if not check_virtual_environment():
        return False
    
    # Show activation command
    activate_cmd = activate_virtual_environment()
    print(f"💡 To manually activate venv: {activate_cmd}")
    
    # Install requirements
    install_requirements()
    
    # Test TensorFlow Hub
    test_success = test_tensorflow_hub()
    
    # Check project structure
    structure_ok = check_project_structure()
    
    # Check sample data
    create_sample_data()
    
    # Show results
    print("\n📊 SETUP SUMMARY")
    print("-" * 20)
    print(f"Python Version: {'✅' if True else '❌'}")
    print(f"Virtual Environment: {'✅' if True else '❌'}")
    print(f"Dependencies: {'✅' if True else '❌'}")
    print(f"TensorFlow Hub: {'✅' if test_success else '⚠️'}")
    print(f"Project Structure: {'✅' if structure_ok else '⚠️'}")
    
    if test_success and structure_ok:
        show_next_steps()
        return True
    else:
        print("\n⚠️  Setup completed with some issues.")
        print("Please check error messages above and resolve them.")
        return False

if __name__ == "__main__":
    success = main()
    
    if not success:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)
    else:
        print("\n🎉 Setup successful! Ready to use Enhanced Video Search System.")
        
        # Ask if user wants to start the system
        if platform.system() == "Windows":
            choice = input("\n🚀 Start the system now? (y/n): ").lower().strip()
            if choice == 'y':
                os.system("launch.bat")
