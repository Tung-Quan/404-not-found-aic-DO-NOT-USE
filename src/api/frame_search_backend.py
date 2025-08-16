#!/usr/bin/env python3
"""
🎯 Frame Search Backend API
==========================
Hệ thống backend trả về JSON với top 5 frames sát nhất với query
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import sys
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import traceback
import pandas as pd
import base64
from datetime import datetime

# Setup environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Try to import TensorFlow components
TF_AVAILABLE = False
TF_HUB_AVAILABLE = False

try:
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
    
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    
    import tensorflow_hub as hub
    import numpy as np
    import faiss
    
    TF_AVAILABLE = True
    TF_HUB_AVAILABLE = True
    print("[OK] TensorFlow Hub available - Full AI mode")
    
except Exception as e:
    print(f"[INFO] TensorFlow not available - Simple mode: {str(e)[:50]}...")
    import numpy as np

# Create FastAPI app
app = FastAPI(
    title="🎯 Frame Search Backend API",
    description="Hệ thống backend trả về JSON với top 5 frames sát nhất với query",
    version="3.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frame serving
if Path("frames").exists():
    app.mount("/frames", StaticFiles(directory="frames"), name="frames")

# Global variables for caching
_model = None
_index = None
_metadata = None

# Models
class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5
    mode: Optional[str] = "auto"  # auto, simple, full

class FrameResult(BaseModel):
    video_name: str
    frame_number: int
    frame_path: str
    frame_url: str
    timestamp: str
    similarity_score: float
    frame_base64: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    mode: str
    processing_time_ms: float
    total_results: int
    results: List[FrameResult]
    metadata: Dict[str, Any]

def load_models():
    """Load TensorFlow models and index"""
    global _model, _index, _metadata
    
    if not TF_HUB_AVAILABLE:
        return False
        
    try:
        if _model is None:
            print("[INFO] Loading Universal Sentence Encoder...")
            _model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
            print("[OK] Model loaded successfully")
        
        if _index is None:
            # Try different FAISS index files
            faiss_files = [
                'index/faiss/video_index.faiss',
                'index/faiss/ip_flat.index', 
                'index/faiss/ip_flat_chinese_clip.index',
                'index/faiss/ivf_chinese_clip.index'
            ]
            
            for faiss_file in faiss_files:
                if Path(faiss_file).exists():
                    print(f"[INFO] Loading FAISS index: {faiss_file}")
                    _index = faiss.read_index(faiss_file)
                    print(f"[OK] FAISS index loaded: {_index.ntotal} vectors")
                    break
            
            if _index is None:
                print("[WARNING] No FAISS index found")
        
        if _metadata is None and Path('index/meta.parquet').exists():
            print("[INFO] Loading metadata...")
            _metadata = pd.read_parquet('index/meta.parquet')
            print(f"[OK] Metadata loaded: {len(_metadata)} entries")
            
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        return False

def get_frame_base64(frame_path: str) -> Optional[str]:
    """Convert frame to base64 string"""
    try:
        if Path(frame_path).exists():
            with open(frame_path, "rb") as f:
                image_data = f.read()
                return base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        print(f"[WARNING] Failed to encode frame: {e}")
    return None

def frame_number_to_timestamp(frame_number: int, fps: float = 25.0) -> str:
    """Convert frame number to timestamp"""
    try:
        seconds = frame_number / fps
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    except:
        return "00:00"

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "🎯 Frame Search Backend API",
        "version": "3.0.0",
        "mode": "full" if TF_HUB_AVAILABLE else "simple",
        "capabilities": {
            "tensorflow_hub": TF_HUB_AVAILABLE,
            "faiss_search": TF_AVAILABLE and Path('index/faiss/video_index.faiss').exists(),
            "metadata_search": Path('index/meta.parquet').exists(),
            "frame_serving": Path("frames").exists()
        },
        "endpoints": {
            "search": "/api/search",
            "health": "/health",
            "frame": "/frames/{video_name}/{frame_file}"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "mode": "full" if TF_HUB_AVAILABLE else "simple",
        "components": {
            "tensorflow_hub": TF_HUB_AVAILABLE,
            "faiss_index": _index is not None,
            "metadata": _metadata is not None,
            "frames_directory": Path("frames").exists()
        },
        "stats": {
            "total_vectors": _index.ntotal if _index else 0,
            "metadata_entries": len(_metadata) if _metadata is not None else 0,
            "frame_directories": len(list(Path("frames").iterdir())) if Path("frames").exists() else 0
        }
    }

@app.post("/api/search", response_model=SearchResponse)
async def search_frames(request: SearchRequest):
    """
    🎯 Tìm kiếm top 5 frames gần nhất với query
    
    Args:
        request: SearchRequest với query, limit (default=5), mode
    
    Returns:
        SearchResponse với top 5 frames và metadata
    """
    start_time = datetime.now()
    
    try:
        # Determine search mode
        if request.mode == "auto":
            mode = "full" if TF_HUB_AVAILABLE and load_models() else "simple"
        else:
            mode = request.mode
        
        if mode == "full" and TF_HUB_AVAILABLE:
            results = await search_full_mode(request.query, request.limit)
        else:
            results = await search_simple_mode(request.query, request.limit)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Create response
        response = SearchResponse(
            query=request.query,
            mode=mode,
            processing_time_ms=round(processing_time, 2),
            total_results=len(results),
            results=results,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "tensorflow_available": TF_HUB_AVAILABLE,
                "index_size": _index.ntotal if _index else 0,
                "search_algorithm": "semantic_vector" if mode == "full" else "keyword_matching"
            }
        )
        
        return response
        
    except Exception as e:
        error_details = {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "query": request.query,
            "mode": mode if 'mode' in locals() else "unknown"
        }
        print(f"[ERROR] Search failed: {error_details}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

async def search_full_mode(query: str, limit: int) -> List[FrameResult]:
    """Full AI search with TensorFlow Hub"""
    try:
        # Load models if needed
        if not load_models():
            raise Exception("Failed to load AI models")
        
        # Encode query
        print(f"[INFO] Encoding query: {query}")
        query_embedding = _model([query]).numpy()
        print(f"[OK] Query encoded: {query_embedding.shape}")
        
        # Search with FAISS
        print(f"[INFO] Searching FAISS index...")
        distances, indices = _index.search(query_embedding.astype('float32'), limit)
        print(f"[OK] Found {len(indices[0])} results")
        
        # Process results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(_metadata):
                row = _metadata.iloc[idx]
                
                # Extract information
                video_name = row.get('video_file', 'unknown')
                frame_number = int(row.get('frame_number', 0))
                
                # Build frame path
                frame_filename = f"frame_{frame_number:06d}.jpg"
                frame_path = f"frames/{video_name}/{frame_filename}"
                frame_url = f"/frames/{video_name}/{frame_filename}"
                
                # Calculate similarity (convert distance to similarity)
                similarity = float(1 - distances[0][i]) if distances[0][i] <= 1.0 else float(max(0, 1 - distances[0][i]))
                
                # Get timestamp
                timestamp = frame_number_to_timestamp(frame_number)
                
                # Get base64 image (optional)
                frame_base64 = get_frame_base64(frame_path) if Path(frame_path).exists() else None
                
                result = FrameResult(
                    video_name=video_name,
                    frame_number=frame_number,
                    frame_path=frame_path,
                    frame_url=frame_url,
                    timestamp=timestamp,
                    similarity_score=round(similarity, 4),
                    frame_base64=frame_base64
                )
                
                results.append(result)
                print(f"[OK] Result {i+1}: {video_name} frame {frame_number} (similarity: {similarity:.4f})")
        
        return results
        
    except Exception as e:
        print(f"[ERROR] Full mode search failed: {e}")
        raise e

async def search_simple_mode(query: str, limit: int) -> List[FrameResult]:
    """Simple keyword search"""
    try:
        print(f"[INFO] Simple mode search for: {query}")
        
        # Load metadata if needed
        global _metadata
        if _metadata is None and Path('index/meta.parquet').exists():
            _metadata = pd.read_parquet('index/meta.parquet')
        
        if _metadata is None:
            # Fallback: scan frames directory
            return await scan_frames_directory(query, limit)
        
        # Simple keyword matching
        query_lower = query.lower()
        matches = []
        
        for idx, row in _metadata.iterrows():
            video_name = row.get('video_file', '')
            frame_number = int(row.get('frame_number', 0))
            
            # Simple relevance scoring
            score = 0.0
            if query_lower in video_name.lower():
                score += 0.8
            
            # Add some randomness for demonstration
            import random
            score += random.random() * 0.2
            
            if score > 0.1:  # Minimum threshold
                matches.append((idx, score, video_name, frame_number))
        
        # Sort by score and take top results
        matches.sort(key=lambda x: x[1], reverse=True)
        matches = matches[:limit]
        
        # Convert to FrameResult objects
        results = []
        for idx, score, video_name, frame_number in matches:
            frame_filename = f"frame_{frame_number:06d}.jpg"
            frame_path = f"frames/{video_name}/{frame_filename}"
            frame_url = f"/frames/{video_name}/{frame_filename}"
            timestamp = frame_number_to_timestamp(frame_number)
            
            # Get base64 image (optional)
            frame_base64 = get_frame_base64(frame_path) if Path(frame_path).exists() else None
            
            result = FrameResult(
                video_name=video_name,
                frame_number=frame_number,
                frame_path=frame_path,
                frame_url=frame_url,
                timestamp=timestamp,
                similarity_score=round(score, 4),
                frame_base64=frame_base64
            )
            results.append(result)
            print(f"[OK] Simple result: {video_name} frame {frame_number} (score: {score:.4f})")
        
        return results
        
    except Exception as e:
        print(f"[ERROR] Simple mode search failed: {e}")
        raise e

async def scan_frames_directory(query: str, limit: int) -> List[FrameResult]:
    """Fallback: scan frames directory"""
    try:
        print(f"[INFO] Scanning frames directory for: {query}")
        
        results = []
        frames_dir = Path("frames")
        
        if not frames_dir.exists():
            return results
        
        # Scan video directories
        for video_dir in frames_dir.iterdir():
            if video_dir.is_dir():
                video_name = video_dir.name
                
                # Simple name matching
                if query.lower() in video_name.lower():
                    # Get some sample frames
                    frame_files = list(video_dir.glob("frame_*.jpg"))
                    
                    # Take every 100th frame for sampling
                    sample_frames = frame_files[::100][:limit]
                    
                    for frame_file in sample_frames:
                        # Extract frame number
                        frame_number = int(frame_file.stem.split('_')[1])
                        frame_path = str(frame_file)
                        frame_url = f"/frames/{video_name}/{frame_file.name}"
                        timestamp = frame_number_to_timestamp(frame_number)
                        
                        # Simple score
                        score = 0.5 + (len(results) * 0.1)
                        
                        result = FrameResult(
                            video_name=video_name,
                            frame_number=frame_number,
                            frame_path=frame_path,
                            frame_url=frame_url,
                            timestamp=timestamp,
                            similarity_score=round(score, 4),
                            frame_base64=None  # Skip base64 for fallback mode
                        )
                        results.append(result)
                        
                        if len(results) >= limit:
                            break
            
            if len(results) >= limit:
                break
        
        return results[:limit]
        
    except Exception as e:
        print(f"[ERROR] Directory scan failed: {e}")
        return []

@app.get("/api/videos")
async def list_videos():
    """List available videos"""
    try:
        videos = []
        frames_dir = Path("frames")
        
        if frames_dir.exists():
            for video_dir in frames_dir.iterdir():
                if video_dir.is_dir():
                    frame_count = len(list(video_dir.glob("frame_*.jpg")))
                    videos.append({
                        "name": video_dir.name,
                        "frame_count": frame_count,
                        "sample_frame": f"/frames/{video_dir.name}/frame_001000.jpg"
                    })
        
        return {
            "total_videos": len(videos),
            "videos": videos
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list videos: {str(e)}")

if __name__ == "__main__":
    # Preload models
    if TF_HUB_AVAILABLE:
        print("[INFO] Preloading AI models...")
        load_models()
    
    # Start server
    print("🚀 Starting Frame Search Backend API...")
    print("📍 API Documentation: http://localhost:8000/docs")
    print("🎯 Search Endpoint: http://localhost:8000/api/search")
    print("📊 Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=False,
        access_log=True
    )
