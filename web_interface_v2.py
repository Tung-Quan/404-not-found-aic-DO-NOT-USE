"""
🌐 Updated Web Interface - BLIP-2 Integration
Web interface tương thích với BLIP-2 search engine mới
"""

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
import json
import time

# Import new BLIP-2 search engine
try:
    from ai_search_engine_v2 import EnhancedBLIP2SearchEngine
    BLIP2_AVAILABLE = True
except ImportError:
    # Fallback to original
    from ai_search_engine import AISearchEngine
    BLIP2_AVAILABLE = False
    logging.warning("⚠️  BLIP-2 engine not available, using original engine")

# Import enhanced hybrid manager (preserve for backward compatibility)
try:
    from enhanced_hybrid_manager import EnhancedHybridManager
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False
    logging.warning("⚠️  Enhanced hybrid manager not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="🧠 AI Video Search - BLIP-2 Enhanced",
    description="Advanced video search với BLIP-2 vision-language model và TensorFlow reranking",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
search_engines = {}  # Multiple search engines per dataset
current_dataset = "mixed_collection"
hybrid_manager = None

# Available models
AVAILABLE_MODELS = {
    "blip2-base": {
        "name": "BLIP-2 Base (Flan-T5)",
        "model_id": "Salesforce/blip2-flan-t5-base",
        "description": "Primary BLIP-2 model với complex query understanding",
        "type": "blip2"
    },
    "blip2-large": {
        "name": "BLIP-2 Large (Flan-T5-XL)", 
        "model_id": "Salesforce/blip2-flan-t5-xl",
        "description": "Larger BLIP-2 model for better accuracy",
        "type": "blip2"
    },
    "hybrid": {
        "name": "Hybrid Manager (Legacy)",
        "model_id": "enhanced_hybrid",
        "description": "Original hybrid model with CLIP variants",
        "type": "hybrid"
    }
}

current_model = "blip2-base"

# Available datasets
def get_available_datasets() -> List[str]:
    """Get list of available datasets"""
    datasets_dir = Path("datasets")
    if not datasets_dir.exists():
        return []
    
    datasets = []
    for item in datasets_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            datasets.append(item.name)
    
    return sorted(datasets)

# Initialize search engines
def initialize_search_engines():
    """Initialize search engines"""
    global search_engines, hybrid_manager
    
    logger.info("🚀 Initializing search engines...")
    
    # Initialize BLIP-2 engines
    if BLIP2_AVAILABLE:
        for model_key, model_info in AVAILABLE_MODELS.items():
            if model_info["type"] == "blip2":
                try:
                    logger.info(f"🧠 Loading {model_info['name']}...")
                    engine = EnhancedBLIP2SearchEngine(
                        dataset_name=current_dataset,
                        blip2_model=model_info["model_id"],
                        enable_tensorflow_rerank=True
                    )
                    
                    if engine.setup_models():
                        if engine.load_or_build_index():
                            search_engines[model_key] = engine
                            logger.info(f"✅ {model_info['name']} ready")
                        else:
                            logger.error(f"❌ Failed to load index for {model_info['name']}")
                    else:
                        logger.error(f"❌ Failed to setup {model_info['name']}")
                        
                except Exception as e:
                    logger.error(f"❌ Error initializing {model_info['name']}: {e}")
    
    # Initialize hybrid manager (backward compatibility)
    if HYBRID_AVAILABLE:
        try:
            logger.info("🔄 Loading hybrid manager...")
            hybrid_manager = EnhancedHybridManager()
            if hybrid_manager.load_models():
                logger.info("✅ Hybrid manager ready")
            else:
                logger.error("❌ Failed to load hybrid manager")
                hybrid_manager = None
        except Exception as e:
            logger.error(f"❌ Error initializing hybrid manager: {e}")
            hybrid_manager = None
    
    logger.info(f"🎉 Initialized {len(search_engines)} search engines")

# API Routes

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🧠 AI Video Search - BLIP-2 Enhanced",
        "version": "2.0.0",
        "available_models": list(AVAILABLE_MODELS.keys()),
        "current_model": current_model,
        "current_dataset": current_dataset,
        "blip2_available": BLIP2_AVAILABLE,
        "hybrid_available": HYBRID_AVAILABLE
    }

@app.get("/api/models")
async def get_models():
    """Get available models"""
    models_status = {}
    
    for model_key, model_info in AVAILABLE_MODELS.items():
        status = {
            **model_info,
            "loaded": False,
            "stats": None
        }
        
        if model_key in search_engines:
            status["loaded"] = True
            try:
                status["stats"] = search_engines[model_key].get_stats()
            except Exception as e:
                logger.warning(f"⚠️  Failed to get stats for {model_key}: {e}")
        
        models_status[model_key] = status
    
    return {
        "models": models_status,
        "current": current_model
    }

@app.post("/api/models/{model_key}/switch")
async def switch_model(model_key: str):
    """Switch current model"""
    global current_model
    
    if model_key not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail=f"Model {model_key} not found")
    
    if model_key not in search_engines:
        raise HTTPException(status_code=503, detail=f"Model {model_key} not loaded")
    
    current_model = model_key
    logger.info(f"🔄 Switched to model: {model_key}")
    
    return {
        "success": True,
        "current_model": current_model,
        "model_info": AVAILABLE_MODELS[model_key]
    }

@app.get("/api/datasets")
async def get_datasets():
    """Get available datasets"""
    datasets = get_available_datasets()
    
    dataset_info = []
    for dataset in datasets:
        dataset_path = Path("datasets") / dataset
        image_count = len(list(dataset_path.rglob("*.jpg"))) + len(list(dataset_path.rglob("*.png")))
        
        dataset_info.append({
            "name": dataset,
            "image_count": image_count,
            "current": dataset == current_dataset
        })
    
    return {
        "datasets": dataset_info,
        "current": current_dataset
    }

@app.post("/api/datasets/{dataset_name}/switch")
async def switch_dataset(dataset_name: str):
    """Switch current dataset"""
    global current_dataset, search_engines
    
    available = get_available_datasets()
    if dataset_name not in available:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")
    
    if dataset_name == current_dataset:
        return {
            "success": True,
            "message": f"Already using dataset {dataset_name}",
            "current_dataset": current_dataset
        }
    
    logger.info(f"🔄 Switching dataset to: {dataset_name}")
    
    # Switch all loaded engines
    switch_results = {}
    for model_key, engine in search_engines.items():
        try:
            success = engine.switch_dataset(dataset_name)
            switch_results[model_key] = success
            if not success:
                logger.warning(f"⚠️  Failed to switch {model_key} to {dataset_name}")
        except Exception as e:
            logger.error(f"❌ Error switching {model_key}: {e}")
            switch_results[model_key] = False
    
    current_dataset = dataset_name
    
    return {
        "success": True,
        "current_dataset": current_dataset,
        "switch_results": switch_results
    }

@app.get("/api/search")
async def search_api(
    q: str = Query(..., description="Search query"),
    k: int = Query(20, description="Number of results"),
    model: Optional[str] = Query(None, description="Model to use"),
    rerank: bool = Query(True, description="Use TensorFlow reranking"),
    dataset: Optional[str] = Query(None, description="Dataset to search")
):
    """
    🔍 Main search API với BLIP-2 + TensorFlow reranking
    """
    # Determine model to use
    use_model = model if model and model in search_engines else current_model
    
    # Determine dataset to use  
    use_dataset = dataset if dataset else current_dataset
    
    if use_model not in search_engines:
        raise HTTPException(status_code=503, detail=f"Model {use_model} not available")
    
    # Get search engine
    engine = search_engines[use_model]
    
    # Switch dataset if needed
    if use_dataset != current_dataset:
        try:
            engine.switch_dataset(use_dataset)
        except Exception as e:
            logger.error(f"❌ Failed to switch dataset: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to switch to dataset {use_dataset}")
    
    try:
        start_time = time.time()
        
        # Perform search
        results = engine.search(
            query=q,
            top_k=k,
            use_reranking=rerank,
            return_scores=True
        )
        
        search_time = time.time() - start_time
        
        # Format results for API
        formatted_results = []
        for result in results:
            # Convert absolute path to relative for web serving
            image_path = result['image_path']
            if os.path.isabs(image_path):
                image_path = os.path.relpath(image_path)
            
            formatted_results.append({
                "image_path": image_path.replace("\\", "/"),  # Web-friendly paths
                "similarity": result.get('final_score', result.get('similarity', 0.0)),
                "rank": result.get('rank', 0),
                "stage": result.get('stage', 'unknown'),
                "tf_score": result.get('tf_score'),
                "blip2_score": result.get('similarity'),
                "query_complexity": result.get('query_complexity', 0.0)
            })
        
        return {
            "success": True,
            "query": q,
            "results": formatted_results,
            "metadata": {
                "total_results": len(formatted_results),
                "search_time": round(search_time, 3),
                "model_used": use_model,
                "dataset_used": use_dataset,
                "reranking_enabled": rerank,
                "query_analysis": results[0].get('parsed_query', {}) if results else {}
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/search")
async def search_post(request_data: dict):
    """POST version of search API"""
    query = request_data.get("query", "")
    top_k = request_data.get("top_k", 20)
    model = request_data.get("model")
    rerank = request_data.get("rerank", True)
    dataset = request_data.get("dataset")
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    # Call GET API
    return await search_api(q=query, k=top_k, model=model, rerank=rerank, dataset=dataset)

@app.get("/frames/{path:path}")
async def serve_frame(path: str):
    """Serve frame images"""
    try:
        # Security: ensure path doesn't go outside frames directory
        safe_path = Path("frames") / path
        safe_path = safe_path.resolve()
        
        if not str(safe_path).startswith(str(Path("frames").resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not safe_path.exists():
            raise HTTPException(status_code=404, detail="Frame not found")
        
        return FileResponse(
            safe_path,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=3600"}
        )
        
    except Exception as e:
        logger.error(f"❌ Error serving frame {path}: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve frame")

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    stats = {
        "system": {
            "current_model": current_model,
            "current_dataset": current_dataset,
            "blip2_available": BLIP2_AVAILABLE,
            "hybrid_available": HYBRID_AVAILABLE
        },
        "models": {},
        "datasets": get_available_datasets()
    }
    
    # Get stats from each loaded model
    for model_key, engine in search_engines.items():
        try:
            stats["models"][model_key] = engine.get_stats()
        except Exception as e:
            logger.warning(f"⚠️  Failed to get stats for {model_key}: {e}")
            stats["models"][model_key] = {"error": str(e)}
    
    return stats

@app.post("/api/cache/clear")
async def clear_cache():
    """Clear all model caches"""
    cleared = {}
    
    for model_key, engine in search_engines.items():
        try:
            engine.clear_caches()
            cleared[model_key] = True
        except Exception as e:
            logger.error(f"❌ Failed to clear cache for {model_key}: {e}")
            cleared[model_key] = False
    
    return {
        "success": True,
        "cleared": cleared
    }

@app.post("/api/index/rebuild")
async def rebuild_index(dataset_name: Optional[str] = None):
    """Rebuild search index"""
    target_dataset = dataset_name or current_dataset
    
    rebuilt = {}
    
    for model_key, engine in search_engines.items():
        try:
            # Switch to target dataset
            if target_dataset != engine.dataset_name:
                engine.switch_dataset(target_dataset)
            
            # Rebuild index
            success = engine.rebuild_index()
            rebuilt[model_key] = success
            
        except Exception as e:
            logger.error(f"❌ Failed to rebuild index for {model_key}: {e}")
            rebuilt[model_key] = False
    
    return {
        "success": any(rebuilt.values()),
        "dataset": target_dataset,
        "rebuilt": rebuilt
    }

# Static files (nếu có web interface)
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

if Path("templates").exists():
    app.mount("/templates", StaticFiles(directory="templates"), name="templates")

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(search_engines),
        "current_model": current_model,
        "current_dataset": current_dataset
    }

# ASGI application export (for uvicorn compatibility)
application = app

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    """Initialize search engines on startup"""
    logger.info("🚀 Starting AI Video Search - BLIP-2 Enhanced")
    initialize_search_engines()
    logger.info("✅ Server ready!")

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "web_interface_v2:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
