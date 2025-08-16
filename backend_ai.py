#!/usr/bin/env python3
"""
AI Video Search - Full Backend
GPU-optimized backend with complete AI features
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def start_full_backend():
    """Start the full AI backend server"""
    print("🚀 Starting Full AI Video Search Backend...")
    print("=" * 60)
    
    try:
        # Import and initialize the enhanced system
        from ai_search_engine import EnhancedAIVideoSearchEngine
        from enhanced_hybrid_manager import EnhancedHybridModelManager
        
        print("🔄 Initializing Enhanced Hybrid Model Manager...")
        manager = EnhancedHybridModelManager()
        
        print("🔄 Initializing Enhanced AI Video Search Engine...")
        search_engine = EnhancedAIVideoSearchEngine(model_manager=manager)
        
        print("✅ Full AI system initialized successfully!")
        print()
        
        # Check available features
        system_info = manager.get_enhanced_system_info()
        
        print("📊 Available Features:")
        print("─" * 40)
        
        # Core models
        available_models = len(manager.available_models)
        print(f"   🤖 Core AI Models: {available_models}")
        
        # AI Agents
        ai_agents = system_info.get("ai_agents", {})
        if ai_agents.get("available"):
            agents_count = len(ai_agents.get("agents", {}))
            print(f"   🤖 AI Agents: {agents_count} available")
            
            # API Keys status
            api_keys = ai_agents.get("api_keys_configured", {})
            if api_keys.get("openai"):
                print("      ✅ OpenAI GPT-4 Vision ready")
            else:
                print("      ⚠️ OpenAI: API key not configured")
                
            if api_keys.get("anthropic"):
                print("      ✅ Anthropic Claude ready")
            else:
                print("      ⚠️ Anthropic: API key not configured")
        else:
            print("   ⚠️ AI Agents: Not available")
            
        # TensorFlow models
        tf_models = system_info.get("tensorflow_models", {})
        if tf_models.get("available"):
            models_count = len(tf_models.get("models", {}))
            print(f"   🔧 TensorFlow Models: {models_count} available")
        else:
            print("   ⚠️ TensorFlow Models: Not available")
            
        # GPU status
        if manager.device == "cuda":
            print(f"   🎮 GPU: {system_info.get('gpu', {}).get('name', 'NVIDIA GPU')} enabled")
        else:
            print("   💻 GPU: Using CPU fallback")
            
        print("─" * 40)
        print()
        
        # Import and start the API server
        print("🔄 Starting FastAPI server...")
        
        # Try to use the main API app
        try:
            from api.app import app
            print("📡 Using main API application")
        except ImportError:
            try:
                from api.simple_enhanced_api import app
                print("📡 Using enhanced API application")
            except ImportError:
                # Create a minimal API if none available
                from fastapi import FastAPI
                from fastapi.responses import JSONResponse
                
                app = FastAPI(title="AI Video Search API", version="1.0.0")
                
                @app.get("/")
                async def root():
                    return {"message": "AI Video Search - Full Version", "status": "running"}
                
                @app.get("/health")
                async def health():
                    return {"status": "healthy", "version": "full"}
                
                @app.get("/system/info")
                async def system_info_endpoint():
                    return manager.get_enhanced_system_info()
                
                print("📡 Using minimal API application")
        
        print()
        print("🌐 Server starting on: http://localhost:8000")
        print("📖 API Documentation: http://localhost:8000/docs")
        print("📊 System Info: http://localhost:8000/system/info")
        print()
        print("To stop the server, press Ctrl+C")
        print("=" * 60)
        
        # Start the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False,
            access_log=True
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Missing dependencies. Try running:")
        print("   python setup.py")
        print("   or")
        print("   python main_launcher.py  # Choose option 3")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("💡 Try using Lite version instead:")
        print("   python main_launcher.py  # Choose option 2")

if __name__ == "__main__":
    start_full_backend()
