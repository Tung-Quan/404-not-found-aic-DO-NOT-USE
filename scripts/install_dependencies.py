#!/usr/bin/env python3
"""
Auto-installer for missing dependencies
Enhanced Video Search System
"""
import subprocess
import sys
import os

def check_and_install_dependency(module_name, package_name, import_name=None):
    """Check if a dependency is installed and install if missing"""
    if import_name is None:
        import_name = module_name
    
    try:
        __import__(import_name)
        print(f"✓ {module_name}: Already installed")
        return True
    except ImportError:
        print(f"✗ {module_name}: Not found, installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✓ {module_name}: Successfully installed")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ {module_name}: Installation failed - {e}")
            return False

def main():
    print("========================================================")
    print("        Auto-Installer for Enhanced Video Search")
    print("========================================================")
    print()
    
    # Check if virtual environment is active
    if 'VIRTUAL_ENV' not in os.environ:
        print("⚠️  Warning: Virtual environment not detected!")
        print("   It's recommended to activate .venv first:")
        print("   .venv\\Scripts\\activate.bat")
        print()
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Installation cancelled.")
            return
    else:
        print(f"✓ Virtual environment active: {os.environ['VIRTUAL_ENV']}")
    
    print()
    print("=== Installing Core Dependencies ===")
    
    # Core dependencies
    dependencies = [
        ('FastAPI', 'fastapi'),
        ('Uvicorn', 'uvicorn'),
        ('Streamlit', 'streamlit'),
        ('NumPy', 'numpy'),
        ('Pandas', 'pandas'),
        ('OpenCV', 'opencv-python', 'cv2'),
        ('Pillow', 'Pillow', 'PIL'),
        ('Requests', 'requests'),
    ]
    
    success_count = 0
    for name, package, *import_name in dependencies:
        import_module = import_name[0] if import_name else package
        if check_and_install_dependency(name, package, import_module):
            success_count += 1
    
    print()
    print("=== Installing FAISS (Vector Search) ===")
    
    # FAISS - different package names for different systems
    faiss_installed = False
    faiss_packages = ['faiss-cpu', 'faiss']
    
    for package in faiss_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ FAISS: Successfully installed ({package})")
            faiss_installed = True
            break
        except subprocess.CalledProcessError:
            continue
    
    if not faiss_installed:
        print("✗ FAISS: Installation failed - may need manual installation")
    else:
        success_count += 1
    
    print()
    print("=== Optional: TensorFlow Hub (Enhanced Features) ===")
    
    # TensorFlow Hub (optional but recommended)
    tf_packages = [
        ('TensorFlow', 'tensorflow'),
        ('TensorFlow Hub', 'tensorflow-hub'),
    ]
    
    # TensorFlow Text is optional and may not be available for all platforms
    tf_text_optional = ('TensorFlow Text', 'tensorflow-text')
    
    install_tf = input("Install TensorFlow Hub for enhanced features? (y/n): ")
    if install_tf.lower() == 'y':
        # Install core TensorFlow packages first
        for name, package in tf_packages:
            if check_and_install_dependency(name, package):
                success_count += 1
        
        # Try to install TensorFlow Text (optional)
        print()
        print("Attempting to install TensorFlow Text (optional)...")
        print("Note: This may fail on some systems/Python versions")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", tf_text_optional[1]])
            print(f"✓ {tf_text_optional[0]}: Successfully installed")
            success_count += 1
        except subprocess.CalledProcessError:
            print(f"⚠️  {tf_text_optional[0]}: Installation failed (this is optional)")
            print("   System will work without tensorflow-text")
            print("   Some advanced text processing features may be limited")
    
    print()
    print("=== Installation Summary ===")
    total_possible = len(dependencies) + 1 + (len(tf_packages) if install_tf.lower() == 'y' else 0)
    print(f"Successfully installed: {success_count}/{total_possible} packages")
    
    if success_count == total_possible:
        print("🎉 All dependencies installed successfully!")
    elif success_count > total_possible * 0.8:
        print("✅ Most dependencies installed. Check errors above for any issues.")
    else:
        print("⚠️  Some installations failed. Please check errors above.")
    
    print()
    print("=== Next Steps ===")
    print("1. Run system status check: python scripts\\check_status.py")
    print("2. Start the system: .\\launch.bat")
    print("3. For enhanced features, ensure TensorFlow Hub is installed")
    
    print()
    print("Installation completed!")

if __name__ == "__main__":
    main()
