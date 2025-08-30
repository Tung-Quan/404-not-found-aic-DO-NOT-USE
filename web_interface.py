"""
🌐 Advanced Web Interface for AI Video Search
==============================================
Multi-video, multi-model search interface with real-time switching
"""

from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import os
import json
import shutil
from pathlib import Path
from enhanced_hybrid_manager import EnhancedHybridModelManager
from ai_search_engine import EnhancedAIVideoSearchEngine
import asyncio
from typing import List, Dict, Optional
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Video Search - Web Interface", version="2.0.0")

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global managers
model_manager = None
search_engine = None
current_dataset = "default"
available_datasets = {}
available_models = []

class VideoDatasetManager:
    """Manages multiple video datasets and their embeddings"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.datasets_path = self.base_path / "datasets"
        self.datasets_path.mkdir(exist_ok=True)
        self.load_datasets()
    
    def load_datasets(self):
        """Load all available datasets"""
        global available_datasets
        available_datasets = {}
        
        # Default dataset
        if (self.base_path / "videos").exists():
            video_count = len(list((self.base_path / "videos").glob("*.mp4")))
            available_datasets["default"] = {
                "name": "Default Dataset",
                "path": str(self.base_path),
                "video_count": video_count,
                "description": f"Original dataset with {video_count} videos"
            }
        
        # Additional datasets
        for dataset_dir in self.datasets_path.iterdir():
            if dataset_dir.is_dir() and (dataset_dir / "videos").exists():
                video_count = len(list((dataset_dir / "videos").glob("*.mp4")))
                available_datasets[dataset_dir.name] = {
                    "name": dataset_dir.name.replace("_", " ").title(),
                    "path": str(dataset_dir),
                    "video_count": video_count,
                    "description": f"Dataset with {video_count} videos"
                }
    
    def create_dataset(self, name: str, description: str = ""):
        """Create a new dataset"""
        dataset_path = self.datasets_path / name
        dataset_path.mkdir(exist_ok=True)
        (dataset_path / "videos").mkdir(exist_ok=True)
        (dataset_path / "frames").mkdir(exist_ok=True)
        (dataset_path / "index").mkdir(exist_ok=True)
        
        # Create config file
        config = {
            "name": name,
            "description": description,
            "created_at": str(datetime.now()),
            "video_count": 0
        }
        
        with open(dataset_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        self.load_datasets()
        return str(dataset_path)
    
    def switch_dataset(self, dataset_name: str):
        """Switch to a different dataset"""
        global current_dataset, search_engine
        
        if dataset_name not in available_datasets:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        current_dataset = dataset_name
        dataset_path = available_datasets[dataset_name]["path"]
        
        # Reinitialize search engine with new dataset
        if search_engine:
            search_engine.base_path = dataset_path
            search_engine.frames_path = os.path.join(dataset_path, "frames")
            search_engine.index_path = os.path.join(dataset_path, "index")
            search_engine.load_frame_records()
        
        return dataset_path

# Initialize dataset manager
dataset_manager = VideoDatasetManager()

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    global model_manager, search_engine, available_models
    
    logger.info("🚀 Starting AI Video Search Web Interface...")
    
    # Initialize model manager
    model_manager = EnhancedHybridModelManager()
    logger.info("✅ Model manager initialized")
    
    # Initialize search engine
    search_engine = EnhancedAIVideoSearchEngine(model_manager=model_manager)
    logger.info("✅ Search engine initialized")
    
    # Get available models
    available_models = [
        {
            "id": "clip_vit_base",
            "name": "CLIP ViT Base", 
            "type": "vision_language",
            "description": "Best for general image-text matching"
        },
        {
            "id": "clip_vit_large", 
            "name": "CLIP ViT Large",
            "type": "vision_language", 
            "description": "Higher accuracy, slower performance"
        },
        {
            "id": "chinese_clip",
            "name": "Chinese CLIP",
            "type": "vision_language",
            "description": "Optimized for Chinese/Vietnamese text"
        },
        {
            "id": "sentence_transformers",
            "name": "Sentence Transformers",
            "type": "text_embedding",
            "description": "Pure text embedding model"
        }
    ]
    
    # Auto-load CLIP model for immediate use
    try:
        logger.info("🚀 Loading default CLIP model...")
        success = model_manager.load_model("clip_vit_base")
        if success:
            search_engine.set_active_model("vision_language", "clip_vit_base")
            logger.info("✅ CLIP model loaded and set as active")
            
            # Build/load embeddings for immediate search capability
            logger.info("🚀 Loading embeddings for search...")
            embeddings_success = search_engine.build_embeddings_index()
            if embeddings_success:
                logger.info("✅ Embeddings loaded successfully")
            else:
                logger.warning("⚠️ Failed to load embeddings")
        else:
            logger.warning("⚠️ Failed to load CLIP model")
    except Exception as e:
        logger.error(f"❌ Error loading CLIP model: {e}")
    
    logger.info("🌐 Web interface ready!")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main web interface"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "datasets": available_datasets,
        "current_dataset": current_dataset,
        "models": available_models,
        "total_datasets": len(available_datasets)
    })

@app.get("/api/datasets")
async def get_datasets():
    """Get all available datasets"""
    dataset_manager.load_datasets()
    return {
        "datasets": available_datasets,
        "current": current_dataset,
        "total": len(available_datasets)
    }

@app.post("/api/datasets/switch")
async def switch_dataset(request: Request):
    """Switch to a different dataset"""
    data = await request.json()
    dataset_name = data.get("dataset")
    
    try:
        dataset_path = dataset_manager.switch_dataset(dataset_name)
        
        # Rebuild embeddings for new dataset if needed
        if search_engine:
            await search_engine.build_embeddings_index()
        
        return {
            "success": True,
            "message": f"Switched to dataset: {dataset_name}",
            "dataset_path": dataset_path,
            "current_dataset": current_dataset
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/datasets/create")
async def create_dataset(
    name: str = Form(...),
    description: str = Form(""),
    files: List[UploadFile] = File(...)
):
    """Create a new dataset with uploaded videos"""
    try:
        # Create dataset directory
        dataset_path = dataset_manager.create_dataset(name, description)
        videos_path = Path(dataset_path) / "videos"
        
        # Save uploaded videos
        for file in files:
            if file.content_type.startswith("video/"):
                file_path = videos_path / file.filename
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
        
        # Extract frames and build embeddings
        import cv2
        import numpy as np
        
        def extract_frames_from_videos(videos_path, frames_path):
            """Extract frames from videos"""
            frames_extracted = 0
            for video_file in Path(videos_path).glob("*.mp4"):
                cap = cv2.VideoCapture(str(video_file))
                frame_count = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % 30 == 0:  # Extract every 30th frame
                        frame_name = f"{video_file.stem}_frame_{frame_count:06d}.jpg"
                        frame_path = Path(frames_path) / frame_name
                        cv2.imwrite(str(frame_path), frame)
                        frames_extracted += 1
                    
                    frame_count += 1
                
                cap.release()
            
            return frames_extracted
        
        # Extract frames
        frames_extracted = extract_frames_from_videos(
            str(videos_path), 
            str(Path(dataset_path) / "frames")
        )
        
        # Build embeddings
        if frames_extracted > 0:
            search_engine_temp = EnhancedAIVideoSearchEngine(
                model_manager=model_manager,
                base_path=dataset_path
            )
            await search_engine_temp.build_embeddings_index()
        
        dataset_manager.load_datasets()
        
        return {
            "success": True,
            "message": f"Dataset '{name}' created with {len(files)} videos",
            "frames_extracted": frames_extracted,
            "dataset_path": dataset_path
        }
        
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/models")
async def get_models():
    """Get all available models"""
    return {
        "models": available_models,
        "current_model": getattr(search_engine, 'current_model', 'clip_vit_base')
    }

@app.post("/api/models/switch")
async def switch_model(request: Request):
    """Switch to a different AI model"""
    data = await request.json()
    model_id = data.get("model")
    
    try:
        # Switch model in search engine
        success = False
        
        if model_id == "clip_vit_base":
            success = model_manager.set_vision_language_model("CLIP ViT Base")
        elif model_id == "clip_vit_large":
            success = model_manager.set_vision_language_model("CLIP ViT Large") 
        elif model_id == "chinese_clip":
            success = model_manager.set_vision_language_model("Chinese CLIP")
        elif model_id == "sentence_transformers":
            success = model_manager.set_text_embedding_model("Sentence Transformers")
        
        if not success:
            # Try loading default models first
            await model_manager.initialize_default_models()
            if model_id == "clip_vit_base":
                success = model_manager.set_vision_language_model("CLIP ViT Base")
        
        if success and search_engine:
            # Update search engine's current model
            search_engine.current_model = model_id
            # Optionally rebuild embeddings with new model
            # await search_engine.build_embeddings_index()
        
        return {
            "success": success,
            "message": f"Switched to model: {model_id}" if success else f"Failed to switch to model: {model_id}",
            "current_model": model_id if success else None
        }
        
    except Exception as e:
        logger.error(f"Error switching model: {e}")
        return {"success": False, "error": str(e)}

@app.get("/search")
async def get_search(q: str, topk: int = 12):
    """GET endpoint for search - for compatibility"""
    if not q:
        return {"error": "Query parameter 'q' is required"}
    
    try:
        # Perform search using current search engine
        results = search_engine.search_similar_frames(q, top_k=topk)
        
        # Format results to ensure all fields are present
        formatted_results = []
        for result in results:
            # Safely handle similarity score
            similarity = result.get("similarity_score", result.get("similarity", 0.0))
            if similarity is None:
                similarity = 0.0
            try:
                similarity = float(similarity)
            except (ValueError, TypeError):
                similarity = 0.0
            
            # Calculate timestamp from frame number if available
            timestamp = result.get("timestamp_seconds", 0.0)
            if timestamp == 0.0 and "frame_number" in result:
                frame_number = result.get("frame_number", 0)
                # Calculate timestamp assuming 30 FPS (1 frame = 1/30 second)
                timestamp = frame_number / 30.0
            
            if timestamp is None:
                timestamp = 0.0
            try:
                timestamp = float(timestamp)
            except (ValueError, TypeError):
                timestamp = 0.0
            
            formatted_result = {
                "frame_path": result.get("frame_path", ""),
                "timestamp": timestamp,
                "frame_number": result.get("frame_number", 0),
                "similarity": similarity,
                "video_name": result.get("video_name", "Unknown"),
                "dataset": result.get("dataset", current_dataset)
            }
            formatted_results.append(formatted_result)
        
        return {
            "results": formatted_results,
            "query": q,
            "total": len(formatted_results),
            "dataset": current_dataset,
            "model": getattr(search_engine, 'current_model', 'clip_vit_base')
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {"error": str(e)}

@app.post("/api/search")
async def web_search(request: Request):
    """Search with current dataset and model"""
    data = await request.json()
    query = data.get("query", "")
    top_k = data.get("top_k", 6)
    model_key = data.get("model", "clip_vit_base")
    
    if not query:
        return {"error": "Query is required"}
    
    try:
        # Switch model if requested
        if model_key and model_key != search_engine.get_active_models().get("vision_language"):
            logger.info(f"Switching to model: {model_key}")
            success = model_manager.load_model(model_key)
            if success:
                search_engine.set_active_model("vision_language", model_key)
                # Build embeddings for new model if needed
                search_engine.build_embeddings_index()
                logger.info(f"✅ Switched to model: {model_key}")
            else:
                logger.warning(f"⚠️ Failed to load model: {model_key}")
        
        # Perform search
        results = search_engine.search_similar_frames(query, top_k=top_k)
        
        # Format results for web display
        formatted_results = []
        for result in results:
            # Safely handle similarity score
            similarity = result.get("similarity_score", result.get("similarity", 0.0))
            if similarity is None:
                similarity = 0.0
            try:
                similarity = float(similarity)
            except (ValueError, TypeError):
                similarity = 0.0
            
            # Calculate timestamp from frame number if available
            timestamp = result.get("timestamp_seconds", 0.0)
            if timestamp == 0.0 and "frame_number" in result:
                frame_number = result.get("frame_number", 0)
                # Calculate timestamp assuming 30 FPS (1 frame = 1/30 second)
                timestamp = frame_number / 30.0
            
            if timestamp is None:
                timestamp = 0.0
            try:
                timestamp = float(timestamp)
            except (ValueError, TypeError):
                timestamp = 0.0
            
            # Extract video name from frame_path if needed
            video_name = result.get("video_name", "Unknown")
            if video_name == "Unknown" and result.get("frame_path"):
                frame_path = result.get("frame_path", "")
                # Extract from path like "frames\good willhunting\frame_000285.jpg"
                path_parts = frame_path.replace('\\', '/').split('/')
                if len(path_parts) >= 2:
                    video_name = path_parts[-2]  # Get folder name
            
            frame_path = result.get("frame_path", "")
            file_name = Path(frame_path).name if frame_path else ""
            formatted_results.append({
                "frame_path": frame_path,
                "file_name": file_name,
                "video_name": video_name,
                "timestamp": timestamp,
                "frame_number": result.get("frame_number", 0),
                "similarity_score": similarity,
                "frame_url": f"/frames/{file_name}"
            })
        
        active_models = search_engine.get_active_models()
        current_model = active_models.get("vision_language", model_key)
        
        return {
            "success": True,
            "query": query,
            "results": formatted_results,
            "total_results": len(formatted_results),
            "dataset": current_dataset,
            "model": current_model
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    try:
        # Get frame count
        frame_count = len(search_engine.frame_records) if search_engine else 0
        
        # Get video count  
        dataset_info = available_datasets.get(current_dataset, {})
        video_count = dataset_info.get("video_count", 0)
        
        # Get model info
        current_model = getattr(search_engine, 'current_model', 'clip_vit_base')
        
        return {
            "dataset": {
                "name": current_dataset,
                "video_count": video_count,
                "frame_count": frame_count
            },
            "model": {
                "current": current_model,
                "available_count": len(available_models)
            },
            "system": {
                "gpu_available": model_manager.device.type == "cuda" if model_manager else False,
                "total_datasets": len(available_datasets)
            }
        }
        
    except Exception as e:
        return {"error": str(e)}

# Serve frame images
@app.get("/frames/{frame_name}")
async def serve_frame(frame_name: str):
    """Serve frame images"""
    dataset_path = available_datasets.get(current_dataset, {}).get("path", ".")
    frame_path = Path(dataset_path) / "frames" / frame_name
    
    if frame_path.exists():
        from fastapi.responses import FileResponse
        return FileResponse(frame_path)
    else:
        return {"error": "Frame not found"}

if __name__ == "__main__":
    import uvicorn
    
    print("🌐 Starting AI Video Search Web Interface...")
    print("📱 Access at: http://localhost:8080")
    print("🎮 Multi-dataset & Multi-model support")
    print("📊 Real-time switching and search")
    
    uvicorn.run(app, host="0.0.0.0", port=8080)
