#!/usr/bin/env python3
"""
AI Video Search - Main Launcher
Enhanced launcher với hỗ trợ Python version check và full installation options
"""

import os
import sys
import time
import traceback
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

def print_banner():
    """Print application banner"""
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║                 AI VIDEO SEARCH SYSTEM                    ║") 
    print("║         Enhanced Launcher với Python Compatibility        ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()

def check_python_compatibility():
    """Check Python version and recommend optimal version"""
    current_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"🐍 Current Python: {current_version}")
    
    # Recommended versions based on compatibility testing
    recommended_versions = {
        "3.9": {"status": "✅ Excellent", "ai_support": "Full", "tensorflow": "✅", "pytorch": "✅"},
        "3.10": {"status": "✅ Excellent", "ai_support": "Full", "tensorflow": "✅", "pytorch": "✅"}, 
        "3.11": {"status": "✅ Very Good", "ai_support": "Full", "tensorflow": "✅", "pytorch": "✅"},
        "3.12": {"status": "⚠️ Good", "ai_support": "Partial", "tensorflow": "⚠️", "pytorch": "✅"},
        "3.13": {"status": "⚠️ Limited", "ai_support": "Basic", "tensorflow": "❌", "pytorch": "⚠️"}
    }
    
    current_key = f"{sys.version_info.major}.{sys.version_info.minor}"
    if current_key in recommended_versions:
        info = recommended_versions[current_key]
        print(f"   Compatibility: {info['status']}")
        print(f"   AI Support: {info['ai_support']}")
        print(f"   TensorFlow: {info['tensorflow']}")
        print(f"   PyTorch: {info['pytorch']}")
    else:
        print("   ⚠️ Unknown compatibility")
    
    print("\n📋 RECOMMENDED PYTHON VERSIONS FOR FULL AI FEATURES:")
    print("   🥇 Python 3.10.x - BEST CHOICE (Most stable for AI/ML)")
    print("   🥈 Python 3.9.x  - Excellent (All features supported)")
    print("   🥉 Python 3.11.x - Very good (Minor compatibility issues)")
    print("   ⚠️ Python 3.12.x - Limited (Some packages may fail)")
    print("   ❌ Python 3.13.x - Not recommended (Many AI packages incompatible)")
    
    if sys.version_info.major == 3 and sys.version_info.minor >= 13:
        print(f"\n⚠️ WARNING: Python {current_version} has limited AI package support!")
        print("   For full features, consider downgrading to Python 3.10.x")
        return False
    elif sys.version_info.major == 3 and sys.version_info.minor >= 12:
        print(f"\n⚠️ NOTICE: Python {current_version} may have some compatibility issues")
        return True
    else:
        print(f"\n✅ Python {current_version} is well supported")
        return True

def check_system_status() -> Dict[str, Any]:
    """Check system capabilities and dependencies"""
    print("\n🔍 Checking system capabilities...")
    
    status = {
        "python_compatible": check_python_compatibility(),
        "gpu": {"available": False, "name": "", "memory": 0},
        "dependencies": {
            "pytorch": False,
            "tensorflow": False,
            "opencv": False,
            "transformers": False,
            "sentence_transformers": False
        },
        "versions": {
            "full": {"available": False, "errors": []},
            "lite": {"available": False, "errors": []}
        }
    }
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            status["gpu"]["available"] = True
            status["gpu"]["name"] = torch.cuda.get_device_name(0)
            status["gpu"]["memory"] = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            print(f"   ✅ GPU: {status['gpu']['name']} ({status['gpu']['memory']}GB)")
        else:
            print("   ⚠️ GPU: CUDA not available")
    except Exception as e:
        print(f"   ❌ GPU: PyTorch error - {e}")
    
    # Check dependencies
    deps_to_check = [
        ("pytorch", "torch"),
        ("tensorflow", "tensorflow"),
        ("opencv", "cv2"),
        ("transformers", "transformers"),
        ("sentence_transformers", "sentence_transformers")
    ]
    
    for dep_name, module_name in deps_to_check:
        try:
            __import__(module_name)
            status["dependencies"][dep_name] = True
            print(f"   ✅ {dep_name.title()}")
        except ImportError:
            print(f"   ❌ {dep_name.title()}")
    
    # Check Full version
    try:
        from ai_search_engine import EnhancedAIVideoSearchEngine
        from enhanced_hybrid_manager import EnhancedHybridModelManager
        
        # Test enhanced manager with all features
        manager = EnhancedHybridModelManager()
        enhanced_info = manager.get_enhanced_system_info()
        
        status["versions"]["full"]["available"] = True
        status["versions"]["full"]["enhanced_features"] = enhanced_info.get("enhanced_features", {})
        status["versions"]["full"]["ai_agents"] = enhanced_info.get("ai_agents", {})
        status["versions"]["full"]["tensorflow_models"] = enhanced_info.get("tensorflow_models", {})
        
        print("   ✅ Full Version: Available")
        
        # Show enhanced features
        ai_agents_info = enhanced_info.get("ai_agents", {})
        if ai_agents_info.get("available"):
            print("   🤖 AI Agents: Available")
        else:
            reason = ai_agents_info.get("reason", "Not configured")
            print(f"   ⚠️ AI Agents: {reason}")
            
        tf_models_info = enhanced_info.get("tensorflow_models", {})
        if tf_models_info.get("available"):
            print("   🔧 TensorFlow Models: Available")
        else:
            reason = tf_models_info.get("reason", "Not available")
            print(f"   ⚠️ TensorFlow Models: {reason}")
            
    except Exception as e:
        status["versions"]["full"]["errors"].append(str(e))
        print(f"   ⚠️ Full Version: {e}")
    
    # Check Lite version
    try:
        from ai_search_lite import AISearchEngineLite
        status["versions"]["lite"]["available"] = True
        print("   ✅ Lite Version: Available")
    except Exception as e:
        status["versions"]["lite"]["errors"].append(str(e))
        print(f"   ❌ Lite Version: {e}")
    
    print()
    return status

def show_version_info(status: Dict[str, Any]):
    """Display detailed version information"""
    print("📋 VERSION COMPARISON")
    print("─" * 60)
    
    # Full Version
    print("🔥 FULL VERSION (GPU-Optimized + AI Agents)")
    print("   Features:")
    print("   • Advanced AI models (CLIP, BLIP)")
    print("   • GPU acceleration with RTX 3060")
    print("   • Hybrid model management")
    print("   • Semantic search capabilities")
    print("   • High accuracy image understanding")
    
    # Enhanced features display
    if "enhanced_features" in status["versions"]["full"]:
        features = status["versions"]["full"]["enhanced_features"]
        if features.get("ai_agents_available"):
            print("   • 🤖 AI Agents (OpenAI, Anthropic, Local)")
        if features.get("tensorflow_models_available"):
            print("   • 🔧 TensorFlow Hub Models")
        
        total_caps = features.get("total_capabilities", 0)
        if total_caps > 0:
            print(f"   • 📊 Total AI Capabilities: {total_caps}")
    
    if status["versions"]["full"]["available"]:
        print("   Status: ✅ Available")
        if status["gpu"]["available"]:
            print(f"   GPU: ✅ {status['gpu']['name']} ({status['gpu']['memory']}GB)")
        else:
            print("   GPU: ⚠️ Will use CPU fallback")
            
        # AI Agents status
        ai_agents = status["versions"]["full"].get("ai_agents", {})
        if ai_agents.get("available"):
            agents_count = len(ai_agents.get("agents", {}))
            print(f"   AI Agents: ✅ {agents_count} agents configured")
            
            # API Keys status
            api_keys = ai_agents.get("api_keys_configured", {})
            if api_keys.get("openai"):
                print("   OpenAI: ✅ API key configured")
            else:
                print("   OpenAI: ⚠️ API key not set")
                
            if api_keys.get("anthropic"):
                print("   Anthropic: ✅ API key configured")
            else:
                print("   Anthropic: ⚠️ API key not set")
        else:
            print("   AI Agents: ⚠️ Not available")
            
        # TensorFlow models status  
        tf_models = status["versions"]["full"].get("tensorflow_models", {})
        if tf_models.get("available"):
            models_count = len(tf_models.get("models", {}))
            print(f"   TensorFlow: ✅ {models_count} models available")
        else:
            print("   TensorFlow: ⚠️ Not available")
            
    else:
        print("   Status: ⚠️ Dependencies missing")
        for error in status["versions"]["full"]["errors"]:
            print(f"   Error: {error}")
    
    print()
    
    # Lite Version
    print("💡 LITE VERSION (Fast & Reliable)")
    print("   Features:")
    print("   • Image similarity search using OpenCV")
    print("   • Color-based frame search")
    print("   • Basic computer vision analysis")
    print("   • Fast performance without AI models")
    print("   • Low resource usage")
    
    if status["versions"]["lite"]["available"]:
        print("   Status: ✅ Available")
        print("   Dependencies: ✅ All satisfied (OpenCV, PIL, NumPy)")
    else:
        print("   Status: ❌ Not available")
        for error in status["versions"]["lite"]["errors"]:
            print(f"   Error: {error}")
    
    print()

def get_user_choice(status: Dict[str, Any]) -> Optional[str]:
    """Get user's enhanced version choice with detailed options"""
    print("🎯 CHOOSE YOUR AI VIDEO SEARCH EXPERIENCE:")
    print("═" * 60)
    
    choices = []
    
    # Option 1: NEW Web Interface
    choices.append(("1", "web", "🌐 WEB INTERFACE - Multi-Dataset & Model Switching (NEW!)"))
    print("   1. 🌐 WEB INTERFACE - Multi-Dataset & Model Switching (NEW!)")
    print("      • 🎯 Visual interface with drag-drop search")
    print("      • 🔄 Real-time model switching (CLIP Base/Large, BLIP)")
    print("      • 📁 Multi-dataset management (Nature, People, Mixed)")
    print("      • ⚡ GPU-accelerated search with previews")
    print("      • 🖼️ Image upload and similarity search")
    print("      • 🚀 Best for new users and demonstrations")
    print("      • 🌐 Access: http://localhost:8080")
    print()
    
    # Option 2: Full Version API
    if status["versions"]["full"]["available"]:
        choices.append(("2", "full", "🔥 FULL API VERSION - Complete AI Experience"))
        print("   2. 🔥 FULL API VERSION - Complete AI Experience")
        print("      • GPU-optimized deep learning models")
        print("      • Advanced semantic search with transformers")
        print("      • Multi-modal AI (vision + language)")
        
        # Show what's actually available
        ai_agents = status["versions"]["full"].get("ai_agents", {})
        tf_models = status["versions"]["full"].get("tensorflow_models", {})
        
        if ai_agents.get("available"):
            print("      • ✅ OpenAI GPT-4 & Anthropic Claude integration")
        else:
            print("      • ⚠️ AI Agents: Available but not configured (API keys needed)")
            
        if tf_models.get("available"):
            print("      • ✅ TensorFlow Hub pre-trained models")
        else:
            print("      • ⚠️ TensorFlow Hub: Available but not fully loaded")
            
        print("      • ✅ Real-time video analysis with CLIP/BLIP")
        print("      • 🌐 Access: http://localhost:8000")
        
        if status["python_compatible"]:
            print("      ✅ Status: Ready to launch")
        else:
            print("      ⚠️ Status: Limited Python compatibility")
        print()
    else:
        print("   1. 🔥 FULL VERSION - ❌ Not Available")
        print("      • Missing dependencies or incompatible Python version")
        print("      • Run option 3 to auto-install dependencies")
        print()
    
    # Option 2: Lite Version
    if status["versions"]["lite"]["available"]:
        choices.append(("2", "lite", "💡 LITE VERSION - Fast & Reliable"))
        print("   2. 💡 LITE VERSION - Fast & Reliable")
        print("      • Basic computer vision with OpenCV")
        print("      • Color and histogram-based search")
        print("      • Fast performance without heavy AI models")
        print("      • Works on any Python 3.8+")
        print("      • Low resource usage")
        print("      ✅ Status: Ready to launch")
        print()
    else:
        print("   2. 💡 LITE VERSION - ❌ Not Available")
        print("      • Missing basic dependencies (OpenCV, PIL)")
        print()
    
    # Option 3: Auto-Install Full Dependencies
    choices.append(("3", "install", "📦 AUTO-INSTALL Full Dependencies"))
    print("   3. 📦 AUTO-INSTALL Full Dependencies")
    print("      • Automatically install missing AI packages")
    print("      • Configure optimal settings for your Python version")
    print("      • Download and setup pre-trained models")
    print("      • Compatible with Python 3.9-3.13")
    print()
    
    # Option 4: Performance comparison
    if status["versions"]["lite"]["available"]:
        choices.append(("4", "compare", "📊 Performance Comparison"))
        print("   4. 📊 Performance Comparison")
        print("      • Test search speed and accuracy")
        print("      • Compare Full vs Lite versions")
        print()
    
    # Option 5: Fix Dependencies
    choices.append(("5", "fix", "🔧 Diagnose & Fix Issues"))
    print("   5. 🔧 Diagnose & Fix Issues")
    print("      • Check system compatibility")
    print("      • Repair broken installations")
    print("      • Install missing dependencies")
    print()
    
    # Option Q: Quit
    choices.append(("q", "quit", "❌ Quit"))
    print("   q. ❌ Quit")
    print()
    
    # Recommendation based on system status
    print("💡 RECOMMENDATIONS:")
    if status["python_compatible"] and status["versions"]["full"]["available"]:
        print("   🥇 Choose option 1 (Full Version) for complete AI experience")
    elif status["versions"]["lite"]["available"] and not status["versions"]["full"]["available"]:
        print("   🥈 Choose option 2 (Lite Version) or option 3 (Auto-Install)")
    elif not status["python_compatible"]:
        print("   ⚠️ Your Python version has compatibility issues")
        print("   🔧 Best option: Install Python 3.10.x, then choose option 3")
    else:
        print("   📦 Choose option 3 (Auto-Install) to setup dependencies")
    print()
    
    while True:
        choice = input("👉 Enter your choice (1-5, q): ").strip().lower()
        
        for key, value, _ in choices:
            if choice == key:
                return value
        
        print("❌ Invalid choice. Please enter 1, 2, 3, 4, 5, or q")

def start_full_version():
    """Start the full AI search system"""
    print("🚀 Starting Full AI Search System...")
    print("=" * 50)
    
    try:
        # Import and start full backend
        import subprocess
        import sys
        
        print("🔄 Initializing GPU-optimized AI models...")
        print("🌐 Server will be available at: http://localhost:8000")
        print("📖 API documentation at: http://localhost:8000/docs")
        print()
        print("To stop the server, press Ctrl+C")
        print("=" * 50)
        
        # Start full backend
        subprocess.run([sys.executable, "backend_ai.py"], check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting full version: {e}")
        print("💡 Try using Lite version instead")

def start_lite_version():
    """Start the lite search system"""
    print("🚀 Starting Lite AI Search System...")
    print("=" * 50)
    
    try:
        # Import and start lite backend
        import subprocess
        import sys
        
        print("🔄 Initializing OpenCV-based search engine...")
        print("🌐 Server will be available at: http://localhost:8000")
        print("📖 API documentation at: http://localhost:8000/docs")
        print()
        print("To stop the server, press Ctrl+C")
        print("=" * 50)
        
        # Start lite backend
        subprocess.run([sys.executable, "backend_ai_lite.py"], check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting lite version: {e}")

def run_performance_comparison():
    """Run performance comparison"""
    print("📊 Running Performance Comparison...")
    print("=" * 50)
    
    try:
        import subprocess
        import sys
        
        subprocess.run([sys.executable, "compare_performance.py"], check=True)
        
    except Exception as e:
        print(f"❌ Error running comparison: {e}")
    
    input("\n👉 Press Enter to continue...")

def fix_dependencies():
    """Help user fix dependencies"""
    print("🔧 DEPENDENCY FIXING GUIDE")
    print("=" * 50)
    
    print("🎯 For Full Version Support:")
    print("   1. Install PyTorch with CUDA:")
    print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print()
    print("   2. Install AI model dependencies:")
    print("      pip install transformers sentence-transformers tensorflow-hub")
    print()
    print("   3. Install additional packages:")
    print("      pip install faiss-cpu pillow")
    print()
    
    print("⚡ For Lite Version (Minimal):")
    print("   pip install opencv-python pillow numpy fastapi uvicorn python-multipart")
    print()
    
    print("🚀 Auto-fix attempt:")
    choice = input("   Do you want to auto-install missing dependencies? (y/n): ").strip().lower()
    
    if choice == 'y':
        try:
            import subprocess
            import sys
            
            print("🔄 Installing dependencies...")
            
            # Essential packages
            essential_packages = [
                "opencv-python",
                "pillow", 
                "numpy",
                "fastapi",
                "uvicorn",
                "python-multipart"
            ]
            
            subprocess.run([sys.executable, "-m", "pip", "install"] + essential_packages, check=True)
            print("✅ Essential packages installed")
            
            # Try AI packages
            ai_packages = [
                "transformers",
                "sentence-transformers", 
                "tensorflow-hub",
                "faiss-cpu"
            ]
            
            try:
                subprocess.run([sys.executable, "-m", "pip", "install"] + ai_packages, check=True)
                print("✅ AI packages installed")
            except:
                print("⚠️ Some AI packages failed to install")
            
            print("🎉 Dependency installation completed!")
            
        except Exception as e:
            print(f"❌ Installation failed: {e}")
    
    input("\n👉 Press Enter to continue...")

def auto_install_full_dependencies():
    """Auto-install full dependencies based on Python version"""
    print("📦 AUTO-INSTALLING FULL DEPENDENCIES")
    print("=" * 60)
    
    # Check Python version and recommend installation strategy
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"🐍 Detected Python {python_version}")
    
    if python_version in ["3.13"]:
        print("⚠️ Python 3.13 detected - Using compatible packages only")
        requirements_file = "config/requirements_compatible.txt"
    elif python_version in ["3.12"]:
        print("⚠️ Python 3.12 detected - Mixed compatibility mode")
        requirements_file = "config/requirements_compatible.txt"
    else:
        print("✅ Python version supports full AI packages")
        requirements_file = "config/requirements.txt"
    
    print(f"📋 Using requirements file: {requirements_file}")
    print()
    
    try:
        # Run setup.py for comprehensive installation
        print("🚀 Running comprehensive setup...")
        print("This may take several minutes...")
        print()
        
        import subprocess
        result = subprocess.run([sys.executable, "setup.py"], 
                              capture_output=False, 
                              text=True)
        
        if result.returncode == 0:
            print("✅ Auto-installation completed successfully!")
            print("🎉 You can now use the Full Version")
        else:
            print("⚠️ Installation completed with some warnings")
            print("💡 You may still be able to use Lite Version")
            
    except Exception as e:
        print(f"❌ Auto-installation failed: {e}")
        print("🔧 Try manual installation:")
        print("   python setup.py")
    
    input("\n👉 Press Enter to continue...")

def start_web_interface():
    """Start the enhanced web interface"""
    print("🌐 Starting Enhanced Web Interface...")
    print("=" * 50)
    
    try:
        # Check if web interface exists
        if not os.path.exists("web_interface.py"):
            print("❌ web_interface.py not found")
            print("💡 Please ensure all files are properly installed")
            return False
            
        print("🔄 Initializing web interface...")
        print("   • Multi-dataset management")
        print("   • Real-time model switching") 
        print("   • GPU acceleration")
        print("   • Visual search interface")
        print()
        
        print("🚀 Starting server...")
        print("📱 Access URL: http://localhost:8080")
        print("⏹️  Press Ctrl+C to stop")
        print()
        
        # Import and run web interface
        import subprocess
        import sys
        
        result = subprocess.run([sys.executable, "web_interface.py"], 
                              cwd=os.getcwd())
        
        if result.returncode == 0:
            print("✅ Web interface closed successfully")
        else:
            print("⚠️ Web interface exited with errors")
            
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Some dependencies may be missing")
        print("🔧 Try running option 4 (Auto-Install) first")
        return False
    except Exception as e:
        print(f"❌ Failed to start web interface: {e}")
        print("🔧 Check your installation and try again")
        return False

def start_full_version():
    """Start full version with all models"""
    print("🚀 Starting full AI Video Search Engine...")
    try:
        from ai_search_engine import EnhancedAIVideoSearchEngine
        ai_search = EnhancedAIVideoSearchEngine()
        ai_search.run()
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Some dependencies may be missing")
        print("🔧 Try running option 4 (Auto-Install) first")
        return False
    except Exception as e:
        print(f"❌ Failed to start full version: {e}")
        print("🔧 Check your installation and try again")
        return False

def main():
    """Main application entry point"""
    print_banner()
    
    while True:
        # Check system status
        status = check_system_status()
        
        # Show version information
        show_version_info(status)
        
        # Get user choice
        choice = get_user_choice(status)
        
        if choice == "web":
            start_web_interface()
        elif choice == "full":
            start_full_version()
        elif choice == "lite":
            start_lite_version()
        elif choice == "install":
            auto_install_full_dependencies()
        elif choice == "demo":
            # Demo mode - start web interface with datasets
            print("🎁 Starting demo mode with sample datasets...")
            start_web_interface()
        elif choice == "compare":
            run_performance_comparison()
        elif choice == "fix":
            fix_dependencies()
        elif choice == "exit":
            print("👋 Goodbye!")
            break
        
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Application terminated by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print(traceback.format_exc())
