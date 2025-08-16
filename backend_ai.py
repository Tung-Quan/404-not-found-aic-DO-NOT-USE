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
                # Create a comprehensive API with search functionality
                from fastapi import FastAPI, HTTPException
                from fastapi.responses import JSONResponse
                from pydantic import BaseModel
                from typing import List, Optional
                import traceback
                
                app = FastAPI(
                    title="AI Video Search API - Full Version", 
                    version="1.0.0",
                    description="Advanced AI-powered video frame search with multiple models"
                )
                
                # Pydantic models for request/response
                class SearchRequest(BaseModel):
                    query: str
                    top_k: Optional[int] = 10
                    model_name: Optional[str] = None
                    search_type: Optional[str] = "semantic"  # semantic, color, hybrid
                
                class SearchResult(BaseModel):
                    frame_path: str
                    video_path: str
                    score: float
                    metadata: Optional[dict] = None
                
                class SearchResponse(BaseModel):
                    query: str
                    results: List[SearchResult]
                    total_found: int
                    search_time: float
                    model_used: str
                
                @app.get("/")
                async def root():
                    return {
                        "message": "AI Video Search - Full Version", 
                        "status": "running",
                        "features": [
                            "semantic_search",
                            "frame_search", 
                            "gpu_acceleration",
                            "multiple_models",
                            "tensorflow_support"
                        ]
                    }
                
                @app.get("/health")
                async def health():
                    return {"status": "healthy", "version": "full"}
                
                @app.get("/system/info")
                async def system_info_endpoint():
                    return manager.get_enhanced_system_info()
                
                @app.post("/search", response_model=SearchResponse)
                async def search_frames(request: SearchRequest):
                    """
                    Search video frames using AI models
                    """
                    try:
                        import time
                        start_time = time.time()
                        
                        print(f"🔍 Search request: '{request.query}' (top_k={request.top_k})")
                        
                        # Initialize default models if needed
                        if not hasattr(search_engine, '_models_initialized'):
                            print("🔄 Initializing default models...")
                            search_engine.initialize_default_models()
                            print("🔄 Building embeddings index...")
                            search_engine.build_embeddings_index()
                            search_engine._models_initialized = True
                        
                        # Perform search based on type
                        if request.search_type == "semantic":
                            results = search_engine.search_similar_frames(
                                query=request.query,
                                top_k=request.top_k
                            )
                        else:
                            # Fallback to semantic search
                            results = search_engine.search_similar_frames(
                                query=request.query,
                                top_k=request.top_k
                            )
                        
                        search_time = time.time() - start_time
                        
                        # Format results
                        formatted_results = []
                        for result in results:
                            formatted_results.append(SearchResult(
                                frame_path=result.get('frame_path', ''),
                                video_path=result.get('video_path', ''),
                                score=float(result.get('score', 0.0)),
                                metadata=result.get('metadata', {})
                            ))
                        
                        response = SearchResponse(
                            query=request.query,
                            results=formatted_results,
                            total_found=len(formatted_results),
                            search_time=search_time,
                            model_used=request.model_name or "default"
                        )
                        
                        print(f"✅ Search completed: {len(formatted_results)} results in {search_time:.3f}s")
                        return response
                        
                    except Exception as e:
                        print(f"❌ Search error: {e}")
                        traceback.print_exc()
                        raise HTTPException(
                            status_code=500, 
                            detail=f"Search failed: {str(e)}"
                        )
                
                @app.get("/search/stats")
                async def search_stats():
                    """
                    Get search engine statistics
                    """
                    try:
                        stats = search_engine.get_stats()
                        return {
                            "status": "success",
                            "stats": stats,
                            "device": search_engine.device,
                            "models_loaded": search_engine.get_active_models()
                        }
                    except Exception as e:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Failed to get stats: {str(e)}"
                        )
                
                @app.post("/models/initialize")
                async def initialize_models():
                    """
                    Initialize default AI models
                    """
                    try:
                        print("� Initializing AI models...")
                        search_engine.initialize_default_models()
                        search_engine._models_initialized = True
                        
                        active_models = search_engine.get_active_models()
                        return {
                            "status": "success",
                            "message": "Models initialized successfully",
                            "active_models": active_models,
                            "device": search_engine.device
                        }
                    except Exception as e:
                        print(f"❌ Model initialization error: {e}")
                        raise HTTPException(
                            status_code=500,
                            detail=f"Model initialization failed: {str(e)}"
                        )
                
                @app.get("/models/available")
                async def get_available_models():
                    """
                    Get list of available AI models
                    """
                    try:
                        active_models = search_engine.get_active_models()
                        system_info = manager.get_enhanced_system_info()
                        
                        return {
                            "active_models": active_models,
                            "tensorflow_models": system_info.get("tensorflow_models", {}),
                            "ai_agents": system_info.get("ai_agents", {}),
                            "device": search_engine.device,
                            "gpu_available": search_engine.device == "cuda"
                        }
                    except Exception as e:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Failed to get models: {str(e)}"
                        )
                
                @app.post("/embeddings/build")
                async def build_embeddings():
                    """
                    Build embeddings index for all frames
                    """
                    try:
                        print("🔄 Building embeddings index...")
                        
                        # Ensure models are initialized first
                        if not hasattr(search_engine, '_models_initialized'):
                            print("🔄 Initializing models first...")
                            search_engine.initialize_default_models()
                            search_engine._models_initialized = True
                        
                        search_engine.build_embeddings_index()
                        
                        stats = search_engine.get_stats()
                        return {
                            "status": "success",
                            "message": "Embeddings built successfully",
                            "stats": stats,
                            "device": search_engine.device
                        }
                    except Exception as e:
                        print(f"❌ Embeddings build error: {e}")
                        raise HTTPException(
                            status_code=500,
                            detail=f"Embeddings build failed: {str(e)}"
                        )
                
                print("📡 Using enhanced API application with search endpoints")
        
        
        print()
        print("🌐 Server starting on: http://localhost:8000")
        print("📖 API Documentation: http://localhost:8000/docs")
        print("📊 System Info: http://localhost:8000/system/info")
        print("🔍 Search Frames: POST http://localhost:8000/search")
        print("📈 Search Stats: GET http://localhost:8000/search/stats") 
        print("🤖 Available Models: GET http://localhost:8000/models/available")
        print("⚡ Initialize Models: POST http://localhost:8000/models/initialize")
        print("🏗️ Build Embeddings: POST http://localhost:8000/embeddings/build")
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
