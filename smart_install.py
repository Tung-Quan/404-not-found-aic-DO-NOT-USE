"""
🔧 Smart Installation Script
===========================
Handles dependency conflicts with multiple strategies
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_with_pip(packages):
    """Install packages with pip"""
    if isinstance(packages, str):
        packages = [packages]
    
    for package in packages:
        try:
            logger.info(f"📦 Installing: {package}")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                logger.info(f"✅ Successfully installed: {package}")
            else:
                logger.warning(f"⚠️ Failed to install {package}: {result.stderr[:200]}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Timeout installing: {package}")
        except Exception as e:
            logger.error(f"❌ Error installing {package}: {e}")

def main():
    logger.info("🚀 Starting smart installation...")
    
    # Step 1: Core ML stack
    logger.info("📊 Installing core ML libraries...")
    core_ml = [
        "torch>=2.0.0,<2.3.0",
        "torchvision>=0.15.0,<0.18.0", 
        "numpy>=1.21.0,<1.27.0"
    ]
    install_with_pip(core_ml)
    
    # Step 2: Basic utilities
    logger.info("🔧 Installing basic utilities...")
    utilities = [
        "Pillow>=9.0.0,<11.0.0",
        "requests>=2.25.0,<3.0.0"
    ]
    install_with_pip(utilities)
    
    # Step 3: Computer vision
    logger.info("👁️ Installing computer vision...")
    cv_packages = ["opencv-python>=4.5.0,<4.10.0"]
    install_with_pip(cv_packages)
    
    # Step 4: NLP and transformers
    logger.info("🤖 Installing NLP libraries...")
    nlp_packages = [
        "transformers>=4.30.0,<4.45.0",
        "ftfy>=6.0.0",
        "regex>=2023.0.0"
    ]
    install_with_pip(nlp_packages)
    
    # Step 5: Vector search
    logger.info("🔍 Installing vector search...")
    vector_packages = ["faiss-cpu>=1.7.0,<1.9.0"]
    install_with_pip(vector_packages)
    
    # Step 6: Web framework
    logger.info("🌐 Installing web framework...")
    web_packages = [
        "fastapi>=0.95.0,<0.115.0",
        "uvicorn>=0.20.0,<0.25.0",
        "jinja2>=3.0.0,<3.2.0",
        "python-multipart>=0.0.5,<0.1.0",
        "pydantic>=2.0.0,<3.0.0"
    ]
    install_with_pip(web_packages)
    
    # Step 7: Additional utilities
    logger.info("🛠️ Installing additional utilities...")
    additional = [
        "aiofiles>=0.8.0,<24.0.0",
        "python-dotenv>=0.19.0,<2.0.0"
    ]
    install_with_pip(additional)
    
    # Step 8: OCR (optional)
    logger.info("📝 Installing OCR (optional)...")
    try:
        install_with_pip("vietocr>=0.3.12")
    except:
        logger.warning("⚠️ VietOCR installation failed, continuing without OCR")
    
    logger.info("✅ Installation completed!")
    
    # Test imports
    logger.info("🧪 Testing imports...")
    test_modules = [
        'torch', 'torchvision', 'numpy', 'PIL', 'cv2', 
        'transformers', 'faiss', 'fastapi', 'uvicorn'
    ]
    
    for module in test_modules:
        try:
            __import__(module)
            logger.info(f"✅ {module} imported successfully")
        except ImportError:
            logger.warning(f"⚠️ {module} import failed")

if __name__ == "__main__":
    main()
