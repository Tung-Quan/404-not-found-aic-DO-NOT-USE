"""
🚀 Quick Start Script for Simplified AI Search System
===================================================
Script khởi động nhanh hệ thống tìm kiếm AI đơn giản hóa
- Check embedding status
- Auto build index if needed
- Complete workflow from setup to web launch
"""

import os
import sys
import subprocess
import logging
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check Python version compatibility"""
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8+ required. Current version: %s", sys.version)
        return False
    logger.info("✅ Python version check passed: %s", sys.version.split()[0])
    return True

def install_requirements():
    """Install required packages with smart conflict resolution"""
    logger.info("📦 Installing requirements with conflict resolution...")
    
    # Strategy 1: Try simplified requirements first
    if try_install_requirements_file("requirements_simplified.txt"):
        return True
    
    logger.warning("⚠️ Simplified requirements failed, trying step-by-step installation...")
    
    # Strategy 2: Try core requirements
    if try_install_requirements_file("requirements_core.txt"):
        logger.info("✅ Core requirements installed")
        
        # Try to install OCR separately
        logger.info("📝 Installing OCR dependencies...")
        if try_install_requirements_file("requirements_ocr.txt"):
            logger.info("✅ OCR dependencies installed")
        else:
            logger.warning("⚠️ OCR dependencies failed, continuing without OCR")
        
        return True
    
    logger.warning("⚠️ Requirements file installation failed, trying individual packages...")
    
    # Strategy 3: Install individual packages
    return install_packages_individually()

def try_install_requirements_file(req_file: str) -> bool:
    """Try to install from requirements file"""
    if not os.path.exists(req_file):
        logger.warning("⚠️ Requirements file not found: %s", req_file)
        return False
    
    try:
        logger.info("📋 Installing from %s...", req_file)
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", req_file
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info("✅ Successfully installed from %s", req_file)
            return True
        else:
            logger.warning("⚠️ Failed to install from %s:", req_file)
            logger.warning(result.stderr[:500])  # Show first 500 chars of error
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Installation timeout for %s", req_file)
        return False
    except Exception as e:
        logger.error("❌ Error installing from %s: %s", req_file, e)
        return False

def install_packages_individually() -> bool:
    """Install packages one by one to handle conflicts"""
    logger.info("🔧 Installing packages individually...")
    
    # Core packages in order of dependency
    package_groups = [
        # Group 1: Core ML
        ["torch>=2.0.0,<2.3.0", "torchvision>=0.15.0,<0.18.0", "numpy>=1.21.0,<1.27.0"],
        
        # Group 2: Basic utilities
        ["Pillow>=9.0.0,<11.0.0", "requests>=2.25.0,<3.0.0"],
        
        # Group 3: Computer vision
        ["opencv-python>=4.5.0,<4.10.0"],
        
        # Group 4: NLP
        ["transformers>=4.30.0,<4.45.0", "ftfy>=6.0.0", "regex>=2023.0.0"],
        
        # Group 5: Vector search
        ["faiss-cpu>=1.7.0,<1.9.0"],
        
        # Group 6: Web framework
        ["fastapi>=0.95.0,<0.115.0", "uvicorn>=0.20.0,<0.25.0", "jinja2>=3.0.0,<3.2.0"],
        
        # Group 7: Additional
        ["python-multipart>=0.0.5,<0.1.0", "pydantic>=2.0.0,<3.0.0", "aiofiles>=0.8.0,<24.0.0", "python-dotenv>=0.19.0,<2.0.0"]
    ]
    
    installed_groups = 0
    
    for i, group in enumerate(package_groups, 1):
        logger.info("📦 Installing group %d/%d: %s", i, len(package_groups), group)
        
        if install_package_group(group):
            installed_groups += 1
            logger.info("✅ Group %d installed successfully", i)
        else:
            logger.warning("⚠️ Group %d failed, continuing...", i)
    
    # Check if we have enough for basic functionality
    if installed_groups >= 5:  # Core ML + utilities + vector search + web
        logger.info("✅ Sufficient packages installed (%d/%d groups)", installed_groups, len(package_groups))
        
        # Try OCR separately
        logger.info("📝 Attempting OCR installation...")
        ocr_packages = ["vietocr>=0.3.12"]
        if install_package_group(ocr_packages):
            logger.info("✅ OCR installed successfully")
        else:
            logger.warning("⚠️ OCR installation failed, continuing without OCR")
        
        return True
    else:
        logger.error("❌ Too few packages installed (%d/%d groups)", installed_groups, len(package_groups))
        return False

def install_package_group(packages: List[str]) -> bool:
    """Install a group of packages"""
    for package in packages:
        if not install_single_package(package):
            # If one package fails, try others in group
            logger.warning("⚠️ Failed to install %s, trying alternatives...", package)
            continue
    
    # Check if at least one package in group was installed
    return True

def install_single_package(package: str) -> bool:
    """Install a single package"""
    try:
        logger.info("📄 Installing: %s", package)
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            logger.info("✅ Installed: %s", package)
            return True
        else:
            logger.warning("⚠️ Failed: %s - %s", package, result.stderr[:200])
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Timeout installing: %s", package)
        return False
    except Exception as e:
        logger.error("❌ Error installing %s: %s", package, e)
        return False

def create_directories():
    """Create necessary directories"""
    logger.info("📁 Creating directories...")
    
    directories = [
        "static",
        "templates", 
        "index",
        "frames",
        "datasets",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info("📂 Created/verified: %s", directory)

def check_gpu_availability():
    """Check if GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info("🚀 GPU detected: %s", gpu_name)
            return True
        else:
            logger.info("💻 Using CPU (no GPU detected)")
            return False
    except ImportError:
        logger.warning("⚠️ PyTorch not installed yet")
        return False

def check_embedding_status() -> Dict[str, any]:
    """Check if embeddings and index are already built"""
    logger.info("🔍 Checking embedding status...")
    
    status = {
        'index_exists': False,
        'metadata_exists': False,
        'frames_found': False,
        'frames_count': 0,
        'index_count': 0,
        'index_path': 'index',
        'frames_paths': [],
        'need_rebuild': False,
        'embedding_stats': {
            'total_processed': 0,
            'with_text': 0,
            'without_text': 0,
            'processing_time': None,
            'average_time_per_image': None
        }
    }
    
    # Check index directory
    index_dir = Path('index')
    if index_dir.exists():
        # Check FAISS index file
        faiss_file = index_dir / 'visual_index.faiss'
        metadata_file = index_dir / 'metadata.pkl'
        
        if faiss_file.exists():
            status['index_exists'] = True
            # Get index file size
            index_size = faiss_file.stat().st_size / (1024 * 1024)  # MB
            logger.info("✅ FAISS index found: %s (%.2f MB)", faiss_file, index_size)
        
        if metadata_file.exists():
            status['metadata_exists'] = True
            try:
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                status['index_count'] = len(metadata)
                
                # Calculate detailed statistics
                total_processed = len(metadata)
                with_text = sum(1 for item in metadata if item.get('has_text', False))
                without_text = total_processed - with_text
                
                status['embedding_stats'].update({
                    'total_processed': total_processed,
                    'with_text': with_text,
                    'without_text': without_text
                })
                
                logger.info("✅ Metadata found: %d items", len(metadata))
                logger.info("📊 Embedding statistics:")
                logger.info("   📷 Total images processed: %d", total_processed)
                logger.info("   📝 Images with text: %d (%.1f%%)", with_text, (with_text/total_processed)*100 if total_processed > 0 else 0)
                logger.info("   🖼️ Images without text: %d (%.1f%%)", without_text, (without_text/total_processed)*100 if total_processed > 0 else 0)
                
            except Exception as e:
                logger.warning("⚠️ Could not read metadata: %s", e)
    
    # Check frames directories
    possible_frame_dirs = ['frames', 'datasets', 'data', 'images']
    for dir_name in possible_frame_dirs:
        if os.path.exists(dir_name):
            frame_count = count_images_in_directory(dir_name)
            if frame_count > 0:
                status['frames_found'] = True
                status['frames_count'] += frame_count
                status['frames_paths'].append(dir_name)
                logger.info("📁 Found %d images in %s", frame_count, dir_name)
    
    # Determine if rebuild is needed
    if not status['index_exists'] or not status['metadata_exists']:
        status['need_rebuild'] = True
        logger.info("🔨 Index needs to be built")
    elif status['frames_count'] > status['index_count']:
        status['need_rebuild'] = True
        logger.info("🔨 Index needs rebuild: %d frames vs %d indexed", 
                   status['frames_count'], status['index_count'])
    else:
        logger.info("✅ Index is up to date")
    
    return status

def count_images_in_directory(directory: str) -> int:
    """Count image files in directory recursively"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    count = 0
    
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    count += 1
    except Exception as e:
        logger.warning("⚠️ Error counting images in %s: %s", directory, e)
    
    return count

def find_best_frames_directory(status: Dict[str, any]) -> Optional[str]:
    """Find the best directory for building index"""
    if not status['frames_paths']:
        return None
    
    # Priority: frames > datasets > data > images
    priority_order = ['frames', 'datasets', 'data', 'images']
    
    for preferred in priority_order:
        for path in status['frames_paths']:
            if preferred in path:
                return path
    
    # Return the one with most images
    best_path = None
    max_count = 0
    
    for path in status['frames_paths']:
        count = count_images_in_directory(path)
        if count > max_count:
            max_count = count
            best_path = path
    
    return best_path

def build_embeddings_index(frames_dir: str) -> bool:
    """Build embeddings index from frames directory with progress tracking"""
    logger.info("🔨 Building embeddings index from %s", frames_dir)
    
    try:
        # Import the simplified search engine
        from simplified_search_engine import SimplifiedSearchEngine
        
        # Initialize engine
        logger.info("🚀 Initializing search engine...")
        engine = SimplifiedSearchEngine()
        
        # Count total images first
        total_images = count_images_in_directory(frames_dir)
        logger.info("📊 Total images to process: %d", total_images)
        
        if total_images == 0:
            logger.error("❌ No images found in %s", frames_dir)
            return False
        
        # Estimate processing time
        estimated_minutes = (total_images * 2) / 60  # Roughly 2 seconds per image
        logger.info("⏱️ Estimated processing time: %.1f minutes", estimated_minutes)
        
        # Build index with progress tracking
        logger.info("📊 Building index (this may take a while)...")
        start_time = time.time()
        
        # Build index (this will handle progress internally)
        engine.build_index(frames_dir, "index")
        
        build_time = time.time() - start_time
        logger.info("✅ Index built successfully in %.2f seconds", build_time)
        
        # Show final statistics
        show_embedding_statistics()
        
        return True
        
    except Exception as e:
        logger.error("❌ Failed to build index: %s", e)
        return False

def show_embedding_statistics():
    """Show detailed embedding statistics"""
    logger.info("\n" + "="*50)
    logger.info("📊 EMBEDDING STATISTICS")
    logger.info("="*50)
    
    try:
        # Read metadata
        metadata_file = Path('index') / 'metadata.pkl'
        if not metadata_file.exists():
            logger.warning("⚠️ No metadata file found")
            return
        
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        total_processed = len(metadata)
        with_text = sum(1 for item in metadata if item.get('has_text', False))
        without_text = total_processed - with_text
        
        # Check index file size
        faiss_file = Path('index') / 'visual_index.faiss'
        index_size = 0
        if faiss_file.exists():
            index_size = faiss_file.stat().st_size / (1024 * 1024)  # MB
        
        # Text embeddings file
        text_embeddings_file = Path('index') / 'text_embeddings.npy'
        text_embeddings_size = 0
        if text_embeddings_file.exists():
            text_embeddings_size = text_embeddings_file.stat().st_size / (1024 * 1024)  # MB
        
        logger.info("📷 Total images processed: %d", total_processed)
        logger.info("📝 Images with text (OCR): %d (%.1f%%)", 
                   with_text, (with_text/total_processed)*100 if total_processed > 0 else 0)
        logger.info("🖼️ Images without text: %d (%.1f%%)", 
                   without_text, (without_text/total_processed)*100 if total_processed > 0 else 0)
        logger.info("💾 Index file size: %.2f MB", index_size)
        logger.info("📄 Text embeddings size: %.2f MB", text_embeddings_size)
        logger.info("🎯 Total storage: %.2f MB", index_size + text_embeddings_size)
        
        # Show sample extracted texts
        text_samples = [item['extracted_text'] for item in metadata if item.get('has_text', False) and item.get('extracted_text', '').strip()]
        if text_samples:
            logger.info("\n📝 Sample extracted texts:")
            for i, text in enumerate(text_samples[:5], 1):
                clean_text = text.strip()[:100] + "..." if len(text.strip()) > 100 else text.strip()
                logger.info("   %d. %s", i, clean_text)
        
        # Show directory breakdown
        logger.info("\n📁 Directory breakdown:")
        dir_counts = {}
        for item in metadata:
            path_parts = Path(item['path']).parts
            if len(path_parts) > 1:
                dir_name = path_parts[0]
                dir_counts[dir_name] = dir_counts.get(dir_name, 0) + 1
        
        for dir_name, count in sorted(dir_counts.items()):
            logger.info("   📂 %s: %d images", dir_name, count)
        
    except Exception as e:
        logger.error("❌ Failed to show statistics: %s", e)

def test_imports():
    """Test if all required modules can be imported"""
    logger.info("🧪 Testing imports...")
    
    required_modules = [
        'torch',
        'transformers', 
        'PIL',
        'cv2',
        'numpy',
        'fastapi',
        'uvicorn'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            logger.info("✅ Import OK: %s", module)
        except ImportError as e:
            logger.error("❌ Import failed: %s - %s", module, e)
            failed_imports.append(module)
    
    # Test OCR imports
    try:
        import vietocr
        logger.info("✅ VietOCR available")
    except ImportError:
        logger.warning("⚠️ VietOCR not available (will install if needed)")
    
    try:
        import easyocr
        logger.info("✅ EasyOCR available")
    except ImportError:
        logger.warning("⚠️ EasyOCR not available (alternative OCR)")
    
    if failed_imports:
        logger.error("❌ Some imports failed. Please install missing packages.")
        return False
    
    logger.info("✅ All core imports successful")
    return True

def start_system():
    """Start the simplified search system"""
    logger.info("🚀 Starting Simplified AI Search System...")
    
    try:
        # Import and run the web interface
        from simplified_web_interface import app
        import uvicorn
        
        logger.info("🌐 Starting web server on http://localhost:8000")
        logger.info("📱 Open your browser and navigate to: http://localhost:8000")
        logger.info("🛑 Press Ctrl+C to stop the server")
        
        uvicorn.run(
            "simplified_web_interface:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        logger.info("👋 System stopped by user")
    except Exception as e:
        logger.error("❌ Failed to start system: %s", e)
        return False
    
    return True

def complete_workflow():
    """Complete workflow from setup to web launch"""
    print("🔍 Simplified AI Search System - Complete Workflow")
    print("=" * 60)
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Create directories
    create_directories()
    
    # Step 3: Install requirements
    if not install_requirements():
        logger.error("❌ Setup failed during requirements installation")
        sys.exit(1)
    
    # Step 4: Check GPU
    has_gpu = check_gpu_availability()
    
    # Step 5: Test imports
    if not test_imports():
        logger.error("❌ Setup failed during import testing")
        sys.exit(1)
    
    # Step 6: Check embedding status
    logger.info("\n" + "="*50)
    logger.info("📊 CHECKING EMBEDDING STATUS")
    logger.info("="*50)
    
    embedding_status = check_embedding_status()
    
    # Step 7: Build index if needed
    if embedding_status['need_rebuild']:
        if not embedding_status['frames_found']:
            logger.error("❌ No image frames found!")
            logger.error("💡 Please add images to one of these directories:")
            logger.error("   - frames/")
            logger.error("   - datasets/")
            logger.error("   - data/")
            logger.error("   - images/")
            sys.exit(1)
        
        best_frames_dir = find_best_frames_directory(embedding_status)
        logger.info("🎯 Selected frames directory: %s", best_frames_dir)
        logger.info("📊 Total images to process: %d", embedding_status['frames_count'])
        
        # Ask user confirmation for index building
        try:
            build_now = input(f"\n🔨 Build embeddings index from {best_frames_dir}? (Y/n): ").lower().strip()
            if build_now not in ['n', 'no']:
                logger.info("\n" + "="*50)
                logger.info("🔨 BUILDING EMBEDDINGS INDEX")
                logger.info("="*50)
                
                if not build_embeddings_index(best_frames_dir):
                    logger.error("❌ Failed to build embeddings index")
                    sys.exit(1)
                
                logger.info("✅ Embeddings index built successfully!")
            else:
                logger.info("⚠️ Skipping index building. System may not work properly.")
        except KeyboardInterrupt:
            logger.info("\n👋 Setup cancelled by user")
            sys.exit(1)
    else:
        logger.info("✅ Embeddings index is ready!")
        logger.info("📊 Index contains %d items", embedding_status['index_count'])
    
    # Step 8: Final status check
    logger.info("\n" + "="*50)
    logger.info("🚀 SYSTEM READY TO LAUNCH")
    logger.info("="*50)
    
    final_status = check_embedding_status()
    if final_status['index_exists'] and final_status['metadata_exists']:
        logger.info("✅ All components ready:")
        logger.info("   🔍 Search engine: Ready")
        logger.info("   📊 Embeddings index: %d items", final_status['index_count'])
        logger.info("   🖼️ Image frames: %d found", final_status['frames_count'])
        logger.info("   🌐 Web interface: Ready to start")
        
        # Step 9: Launch web interface
        try:
            launch_now = input(f"\n🚀 Launch web interface at http://localhost:8000? (Y/n): ").lower().strip()
            if launch_now not in ['n', 'no']:
                logger.info("\n" + "="*50)
                logger.info("🌐 STARTING WEB INTERFACE")
                logger.info("="*50)
                logger.info("🔗 Open your browser: http://localhost:8000")
                logger.info("🛑 Press Ctrl+C to stop")
                
                start_system()
            else:
                logger.info("💡 To start web interface later:")
                logger.info("   python quick_start.py --start")
                logger.info("   python simplified_web_interface.py")
        except KeyboardInterrupt:
            logger.info("\n👋 Setup completed. Launch manually when ready.")
    else:
        logger.error("❌ System not ready. Please check the issues above.")
        sys.exit(1)

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Simplified AI Search System')
    parser.add_argument('--install', action='store_true', 
                       help='Install dependencies only')
    parser.add_argument('--build-index', type=str, metavar='FRAMES_DIR',
                       help='Build search index from frames directory')
    parser.add_argument('--start', action='store_true',
                       help='Start web interface')
    parser.add_argument('--status', action='store_true',
                       help='Show system status only')
    parser.add_argument('--info', action='store_true',
                       help='Show detailed embedding information and statistics')
    
    args = parser.parse_args()
    
    # If no arguments, run default workflow
    if not any(vars(args).values()):
        complete_workflow()
    elif args.install:
        install_dependencies()
    elif args.build_index:
        frames_dir = args.build_index
        if Path(frames_dir).exists():
            build_embeddings_index(frames_dir)
        else:
            print(f"❌ Directory not found: {frames_dir}")
    elif args.start:
        start_web_interface()
    elif args.status:
        show_status_only()
    elif args.info:
        show_embedding_info()

def print_manual_commands():
    """Print manual commands for step-by-step execution"""
    print("\n" + "="*60)
    print("📋 MANUAL COMMANDS - Step by Step Execution")
    print("="*60)
    
    print("\n🔧 1. Setup Environment:")
    print("   pip install -r requirements_simplified.txt")
    
    print("\n🧪 2. Test Installation:")
    print("   python -c \"import torch, transformers, PIL, cv2, numpy, fastapi, uvicorn; print('✅ All imports OK')\"")
    
    print("\n🔍 3. Check GPU (Optional):")
    print("   python -c \"import torch; print('GPU Available:', torch.cuda.is_available())\"")
    
    print("\n📁 4. Check Image Frames:")
    print("   # Make sure you have images in one of these directories:")
    print("   #   frames/")
    print("   #   datasets/") 
    print("   #   data/")
    print("   #   images/")
    
    print("\n🔨 5. Build Embeddings Index:")
    print("   python -c \"")
    print("   from simplified_search_engine import SimplifiedSearchEngine")
    print("   engine = SimplifiedSearchEngine()")
    print("   engine.build_index('frames', 'index')  # Change 'frames' to your directory")
    print("   print('✅ Index built successfully')")
    print("   \"")
    
    print("\n🌐 6. Start Web Interface:")
    print("   python simplified_web_interface.py")
    print("   # Or:")
    print("   python quick_start.py --start")
    
    print("\n🔗 7. Open Browser:")
    print("   http://localhost:8000")
    
    print("\n" + "="*60)
    print("💡 TIP: Run 'python quick_start.py' for automatic workflow")
    print("="*60)

def show_status_only():
    """Show current system status without making changes"""
    print("🔍 Simplified AI Search System - Status Check")
    print("=" * 50)
    
    # Check embedding status
    embedding_status = check_embedding_status()
    
    print("\n📊 SYSTEM STATUS:")
    print("-" * 30)
    print(f"✅ Index exists: {embedding_status['index_exists']}")
    print(f"✅ Metadata exists: {embedding_status['metadata_exists']}")
    print(f"📁 Frames found: {embedding_status['frames_found']}")
    print(f"📊 Frames count: {embedding_status['frames_count']}")
    print(f"📋 Index count: {embedding_status['index_count']}")
    print(f"🔨 Needs rebuild: {embedding_status['need_rebuild']}")
    
    if embedding_status['frames_paths']:
        print(f"📂 Frame directories: {', '.join(embedding_status['frames_paths'])}")
    
    # Check dependencies
    print("\n🧪 DEPENDENCIES:")
    print("-" * 30)
    test_imports()
    
    # Check GPU
    print("\n🚀 GPU STATUS:")
    print("-" * 30)
    check_gpu_availability()
    
    print("\n💡 NEXT STEPS:")
    print("-" * 30)
    if embedding_status['need_rebuild']:
        if embedding_status['frames_found']:
            best_dir = find_best_frames_directory(embedding_status)
            print(f"🔨 Run: python quick_start.py --build-index {best_dir}")
        else:
            print("❌ Add images to frames/, datasets/, data/, or images/ directory")
    else:
        print("🚀 Run: python quick_start.py --start")

def show_embedding_info():
    """Show comprehensive embedding information"""
    print("� Simplified AI Search System - Embedding Information")
    print("=" * 60)
    
    # Check embedding status
    embedding_status = check_embedding_status()
    
    print("\n� INDEX STATUS:")
    print("-" * 30)
    print(f"✅ Index exists: {embedding_status['index_exists']}")
    print(f"✅ Metadata exists: {embedding_status['metadata_exists']}")
    print(f"📁 Frames found: {embedding_status['frames_found']}")
    print(f"📊 Frames count: {embedding_status['frames_count']}")
    print(f"📋 Index count: {embedding_status['index_count']}")
    print(f"🔨 Needs rebuild: {embedding_status['need_rebuild']}")
    
    if embedding_status['frames_paths']:
        print(f"📂 Frame directories: {', '.join(embedding_status['frames_paths'])}")
    
    # Show detailed statistics if index exists
    if embedding_status['index_exists'] and embedding_status['metadata_exists']:
        print("\n📊 DETAILED STATISTICS:")
        print("-" * 30)
        stats = embedding_status['embedding_stats']
        total = stats['total_processed']
        with_text = stats['with_text']
        without_text = stats['without_text']
        
        print(f"📷 Total processed: {total}")
        print(f"📝 With text: {with_text} ({(with_text/total)*100:.1f}%)" if total > 0 else "📝 With text: 0")
        print(f"🖼️ Without text: {without_text} ({(without_text/total)*100:.1f}%)" if total > 0 else "🖼️ Without text: 0")
        
        # File sizes
        index_dir = Path('index')
        faiss_file = index_dir / 'visual_index.faiss'
        metadata_file = index_dir / 'metadata.pkl'
        text_embeddings_file = index_dir / 'text_embeddings.npy'
        
        print(f"\n💾 FILE SIZES:")
        print("-" * 30)
        if faiss_file.exists():
            size_mb = faiss_file.stat().st_size / (1024 * 1024)
            print(f"🔍 Visual index: {size_mb:.2f} MB")
        
        if metadata_file.exists():
            size_mb = metadata_file.stat().st_size / (1024 * 1024)
            print(f"📋 Metadata: {size_mb:.2f} MB")
        
        if text_embeddings_file.exists():
            size_mb = text_embeddings_file.stat().st_size / (1024 * 1024)
            print(f"📝 Text embeddings: {size_mb:.2f} MB")
        
        # Show sample texts
        show_sample_extracted_texts()
    
    # GPU status
    print(f"\n🚀 GPU STATUS:")
    print("-" * 30)
    check_gpu_availability()
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 30)
    if embedding_status['need_rebuild']:
        if embedding_status['frames_found']:
            best_dir = find_best_frames_directory(embedding_status)
            print(f"🔨 Run: python quick_start.py --build-index {best_dir}")
        else:
            print("❌ Add images to frames/, datasets/, data/, or images/ directory")
    else:
        print("🚀 Ready to search! Run: python quick_start.py --start")

def show_sample_extracted_texts():
    """Show sample extracted texts from metadata"""
    try:
        metadata_file = Path('index') / 'metadata.pkl'
        if not metadata_file.exists():
            return
        
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        text_samples = []
        for item in metadata:
            if item.get('has_text', False) and item.get('extracted_text', '').strip():
                text_samples.append({
                    'text': item['extracted_text'].strip(),
                    'path': Path(item['path']).name
                })
        
        if text_samples:
            print(f"\n📝 SAMPLE EXTRACTED TEXTS ({len(text_samples)} total):")
            print("-" * 30)
            for i, sample in enumerate(text_samples[:5], 1):
                clean_text = sample['text'][:80] + "..." if len(sample['text']) > 80 else sample['text']
                print(f"{i}. [{sample['path']}] {clean_text}")
            
            if len(text_samples) > 5:
                print(f"... and {len(text_samples) - 5} more")
    except Exception as e:
        print(f"⚠️ Could not show sample texts: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simplified AI Search System Quick Start")
    parser.add_argument("--start", action="store_true", help="Start web interface only")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--info", action="store_true", help="Show detailed embedding information and statistics")
    parser.add_argument("--build-index", type=str, help="Build index from specified directory")
    parser.add_argument("--manual", action="store_true", help="Show manual commands")
    parser.add_argument("--setup-only", action="store_true", help="Setup dependencies only")
    
    args = parser.parse_args()
    
    if args.manual:
        print_manual_commands()
    elif args.status:
        show_status_only()
    elif args.info:
        show_embedding_info()
    elif args.start:
        start_system()
    elif args.build_index:
        if not os.path.exists(args.build_index):
            logger.error("❌ Directory not found: %s", args.build_index)
            sys.exit(1)
        if build_embeddings_index(args.build_index):
            logger.info("✅ Index built successfully!")
        else:
            logger.error("❌ Failed to build index")
            sys.exit(1)
    elif args.setup_only:
        # Setup only
        if check_python_version():
            create_directories()
            if install_requirements():
                logger.info("✅ Setup completed!")
                test_imports()
            else:
                sys.exit(1)
    else:
        # Full workflow
        main()
