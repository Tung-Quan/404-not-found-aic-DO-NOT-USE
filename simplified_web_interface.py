"""
🌐 Simplified Web Interface - Single Model + OCR
===============================================
Web interface đơn giản hóa sử dụng 1 mô hình chính với OCR tích hợp
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import json
from pathlib import Path
import os
import uvicorn

# Import simplified engine
from simplified_search_engine import SimplifiedSearchEngine, get_engine, initialize_engine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="🔍 Simplified AI Search System",
    description="Hệ thống tìm kiếm AI đơn giản với 1 mô hình + OCR",
    version="2.0.0"
)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Request models
class SearchRequest(BaseModel):
    query: str
    top_k: int = 20
    search_mode: str = "hybrid"  # visual, text, hybrid

class IndexBuildRequest(BaseModel):
    frames_directory: str
    save_path: str = "index"

class ConfigRequest(BaseModel):
    model_name: str = "ViT-B/32"
    device: str = "auto"

# Global state
engine_initialized = False
index_loaded = False

@app.on_event("startup")
async def startup_event():
    """Initialize engine on startup"""
    global engine_initialized
    try:
        initialize_engine()
        engine_initialized = True
        logger.info("✅ Simplified Search Engine initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize engine: {e}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    return templates.TemplateResponse("simplified_index.html", {
        "request": request,
        "engine_initialized": engine_initialized,
        "index_loaded": index_loaded
    })

@app.get("/api/status")
async def get_status():
    """Get system status"""
    return {
        "engine_initialized": engine_initialized,
        "index_loaded": index_loaded,
        "model_info": {
            "name": get_engine().model_name if engine_initialized else None,
            "device": get_engine().device if engine_initialized else None,
            "ocr_available": get_engine().ocr_predictor is not None if engine_initialized else False
        }
    }

@app.post("/api/initialize")
async def initialize_system(config: ConfigRequest):
    """Initialize or reconfigure engine"""
    global engine_initialized
    
    try:
        initialize_engine(config.model_name, config.device)
        engine_initialized = True
        
        return {
            "success": True,
            "message": f"Engine initialized with {config.model_name} on {config.device}",
            "model_name": config.model_name,
            "device": config.device
        }
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/build_index")
async def build_index(request: IndexBuildRequest):
    """Build search index"""
    global index_loaded
    
    if not engine_initialized:
        raise HTTPException(status_code=400, detail="Engine not initialized")
    
    try:
        engine = get_engine()
        
        # Validate frames directory
        if not os.path.exists(request.frames_directory):
            raise HTTPException(status_code=400, detail=f"Directory not found: {request.frames_directory}")
        
        # Build index
        engine.build_index(request.frames_directory, request.save_path)
        index_loaded = True
        
        return {
            "success": True,
            "message": f"Index built from {request.frames_directory}",
            "save_path": request.save_path,
            "total_frames": len(engine.metadata)
        }
    except Exception as e:
        logger.error(f"❌ Index building failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/load_index")
async def load_index(index_path: str = Form(...)):
    """Load existing index"""
    global index_loaded
    
    if not engine_initialized:
        raise HTTPException(status_code=400, detail="Engine not initialized")
    
    try:
        engine = get_engine()
        
        # Validate index path
        if not os.path.exists(index_path):
            raise HTTPException(status_code=400, detail=f"Index path not found: {index_path}")
        
        # Load index
        engine.load_index(index_path)
        index_loaded = True
        
        return {
            "success": True,
            "message": f"Index loaded from {index_path}",
            "total_frames": len(engine.metadata)
        }
    except Exception as e:
        logger.error(f"❌ Index loading failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search")
async def search(request: SearchRequest):
    """Perform search"""
    if not engine_initialized:
        raise HTTPException(status_code=400, detail="Engine not initialized")
    
    if not index_loaded:
        raise HTTPException(status_code=400, detail="Index not loaded")
    
    try:
        engine = get_engine()
        
        # Perform search
        results = engine.search(
            query=request.query,
            top_k=request.top_k,
            search_mode=request.search_mode
        )
        
        return {
            "success": True,
            "query": request.query,
            "search_mode": request.search_mode,
            "total_results": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"❌ Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/datasets")
async def list_datasets():
    """List available datasets/directories"""
    try:
        # Common dataset directories
        common_dirs = [
            "frames",
            "datasets",
            "data",
            "images"
        ]
        
        available_dirs = []
        for dir_name in common_dirs:
            if os.path.exists(dir_name):
                # Count subdirectories
                subdirs = [d for d in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, d))]
                available_dirs.append({
                    "name": dir_name,
                    "path": os.path.abspath(dir_name),
                    "subdirectories": len(subdirs),
                    "subdirs": subdirs[:10]  # Show first 10
                })
        
        return {
            "success": True,
            "datasets": available_dirs
        }
    except Exception as e:
        logger.error(f"❌ Failed to list datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/frame/{path:path}")
async def get_frame(path: str):
    """Get frame image"""
    try:
        # Security: ensure path is within allowed directories
        allowed_dirs = ["frames", "datasets", "data", "images", "static"]
        
        if not any(path.startswith(dir) for dir in allowed_dirs):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Frame not found")
        
        from fastapi.responses import FileResponse
        return FileResponse(path)
    except Exception as e:
        logger.error(f"❌ Failed to get frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process_image")
async def process_image(file: UploadFile = File(...)):
    """Process uploaded image"""
    if not engine_initialized:
        raise HTTPException(status_code=400, detail="Engine not initialized")
    
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        engine = get_engine()
        result = engine.process_image_multimodal(temp_path)
        
        # Clean up temp file
        os.remove(temp_path)
        
        return {
            "success": True,
            "filename": file.filename,
            "extracted_text": result['extracted_text'],
            "has_text": bool(result['extracted_text'].strip())
        }
    except Exception as e:
        logger.error(f"❌ Image processing failed: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "engine_initialized": engine_initialized,
        "index_loaded": index_loaded
    }

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    os.makedirs("index", exist_ok=True)
    
    # Run server
    uvicorn.run(
        "simplified_web_interface:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
