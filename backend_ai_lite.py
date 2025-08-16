#!/usr/bin/env python3
"""
AI Video Search - Lite Backend
Fast & reliable backend using OpenCV
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def start_lite_backend():
    """Start the lite backend server"""
    print("🚀 Starting Lite AI Video Search Backend...")
    print("=" * 60)
    
    try:
        # Import and initialize the lite system
        from ai_search_lite import AISearchEngineLite
        
        print("🔄 Initializing Lite Search Engine...")
        search_engine = AISearchEngineLite()
        
        print("✅ Lite system initialized successfully!")
        print()
        
        print("📊 Available Features:")
        print("─" * 40)
        print("   🔍 OpenCV-based image similarity")
        print("   🎨 Color histogram analysis")
        print("   📏 Basic computer vision metrics")
        print("   ⚡ Fast performance")
        print("   💾 Low resource usage")
        print("   🖥️ CPU-only operation")
        print("─" * 40)
        print()
        
        # Import and start the API server
        print("🔄 Starting FastAPI server...")
        
        # Create a minimal API for lite version
        from fastapi import FastAPI, UploadFile, File, Form
        from fastapi.responses import JSONResponse, HTMLResponse
        from fastapi.staticfiles import StaticFiles
        import json
        
        app = FastAPI(title="AI Video Search API - Lite", version="1.0.0")
        
        # Serve static files if templates directory exists
        if os.path.exists("templates"):
            app.mount("/static", StaticFiles(directory="templates"), name="static")
        
        @app.get("/", response_class=HTMLResponse)
        async def root():
            # Serve basic HTML interface
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>AI Video Search - Lite Version</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .header { background: #f0f0f0; padding: 20px; border-radius: 8px; }
                    .feature { margin: 10px 0; }
                    .status { color: green; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🚀 AI Video Search - Lite Version</h1>
                    <p class="status">✅ System is running</p>
                </div>
                
                <h2>📊 Available Features:</h2>
                <div class="feature">🔍 OpenCV-based image similarity</div>
                <div class="feature">🎨 Color histogram analysis</div>
                <div class="feature">📏 Basic computer vision metrics</div>
                <div class="feature">⚡ Fast performance</div>
                <div class="feature">💾 Low resource usage</div>
                
                <h2>📖 API Endpoints:</h2>
                <ul>
                    <li><a href="/docs">📚 Interactive API Documentation</a></li>
                    <li><a href="/health">🔧 Health Check</a></li>
                    <li><a href="/system/info">📊 System Information</a></li>
                </ul>
            </body>
            </html>
            """
            return HTMLResponse(content=html_content)
        
        @app.get("/health")
        async def health():
            return {"status": "healthy", "version": "lite", "engine": "opencv"}
        
        @app.get("/system/info")
        async def system_info():
            return {
                "version": "lite",
                "engine": "OpenCV",
                "features": [
                    "Image similarity search",
                    "Color histogram analysis", 
                    "Basic computer vision",
                    "Fast performance",
                    "Low resource usage"
                ],
                "status": "running"
            }
        
        @app.post("/search/similarity")
        async def search_similarity(
            query_image: UploadFile = File(...),
            top_k: int = Form(default=5)
        ):
            """Search for similar images using OpenCV"""
            try:
                # Save uploaded file temporarily
                temp_path = f"temp_query_{query_image.filename}"
                with open(temp_path, "wb") as f:
                    content = await query_image.read()
                    f.write(content)
                
                # Perform similarity search using lite engine
                results = search_engine.search_similar_images(temp_path, top_k=top_k)
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                return {
                    "query_image": query_image.filename,
                    "results": results,
                    "engine": "lite"
                }
                
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e), "engine": "lite"}
                )
        
        print("📡 Lite API application ready")
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
        print("💡 Missing basic dependencies. Try installing:")
        print("   pip install opencv-python pillow numpy fastapi uvicorn")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("💡 Check if all dependencies are installed correctly")

if __name__ == "__main__":
    start_lite_backend()
