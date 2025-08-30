#!/usr/bin/env python3
"""
Enhanced Video Search API - Unified Server
==========================================
Intelligent API server that auto-detects TensorFlow availability
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os
import sys
from pathlib import Path
from typing import List, Optional
import traceback

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
    TF_AVAILABLE = True
    TF_HUB_AVAILABLE = True
    print("[OK] TensorFlow Hub available - Full mode")
    
except Exception as e:
    print(f"[INFO] TensorFlow not available - Simple mode: {str(e)[:50]}...")

# Import other dependencies
try:
    import numpy as np
    import pandas as pd
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[INFO] FAISS not available")

# Create FastAPI app
app = FastAPI(
    title="Enhanced Video Search API",
    description="Intelligent video search with auto-detection capabilities",
    version="2.0.0"
)

# Models
class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10

class SearchResult(BaseModel):
    video_file: str
    frame_number: int
    timestamp: str
    similarity: float

@app.get("/")
async def root():
    return {
        "message": "Enhanced Video Search API",
        "version": "2.0.0",
        "mode": "full" if TF_AVAILABLE else "simple",
        "tensorflow_hub": TF_HUB_AVAILABLE,
        "faiss": FAISS_AVAILABLE
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "mode": "full" if TF_AVAILABLE else "simple",
        "components": {
            "tensorflow_hub": TF_HUB_AVAILABLE,
            "faiss": FAISS_AVAILABLE,
            "virtual_env": "VIRTUAL_ENV" in os.environ
        }
    }

@app.post("/search")
async def search_videos(request: SearchRequest):
    """Search videos with intelligent mode detection"""
    
    if TF_HUB_AVAILABLE and FAISS_AVAILABLE:
        # Full mode - use TensorFlow Hub
        try:
            return await search_full_mode(request.query, request.limit)
        except Exception as e:
            print(f"[WARNING] Full mode failed: {e}")
            return await search_simple_mode(request.query, request.limit)
    else:
        # Simple mode - basic search
        return await search_simple_mode(request.query, request.limit)

async def search_full_mode(query: str, limit: int):
    """Full search with TensorFlow Hub"""
    try:
        # Load model and index
        model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        index = faiss.read_index('index/video_index.faiss')
        
        # Encode query
        query_embedding = model([query]).numpy()
        
        # Search
        distances, indices = index.search(query_embedding, limit)
        
        # Load metadata
        metadata = pd.read_parquet('index/meta.parquet')
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(metadata):
                row = metadata.iloc[idx]
                results.append({
                    "video_file": row.get('video_file', 'unknown'),
                    "frame_number": int(row.get('frame_number', 0)),
                    "timestamp": row.get('timestamp', '0:00'),
                    "similarity": float(1 - distances[0][i])
                })
        
        return {
            "query": query,
            "mode": "full",
            "results": results,
            "total": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

async def search_simple_mode(query: str, limit: int):
    """Simple search without TensorFlow"""
    try:
        # Basic keyword search in metadata
        if Path('index/meta.parquet').exists():
            metadata = pd.read_parquet('index/meta.parquet')
            
            # Simple text matching
            if 'description' in metadata.columns:
                matches = metadata[metadata['description'].str.contains(query, case=False, na=False)]
            else:
                matches = metadata.head(limit)  # Return first N results
            
            results = []
            for _, row in matches.head(limit).iterrows():
                results.append({
                    "video_file": row.get('video_file', 'unknown'),
                    "frame_number": int(row.get('frame_number', 0)),
                    "timestamp": row.get('timestamp', '0:00'),
                    "similarity": 0.5  # Default similarity
                })
            
            return {
                "query": query,
                "mode": "simple",
                "results": results,
                "total": len(results),
                "note": "Simple mode - install TensorFlow Hub for better results"
            }
        else:
            return {
                "query": query,
                "mode": "simple",
                "results": [],
                "total": 0,
                "note": "No index found - run indexing first"
            }
            
    except Exception as e:
        return {
            "query": query,
            "mode": "error",
            "results": [],
            "error": str(e)
        }

@app.get("/system/info")
async def system_info():
    """Get system information"""
    return {
        "platform": sys.platform,
        "python_version": sys.version,
        "tensorflow_available": TF_AVAILABLE,
        "tensorflow_hub_available": TF_HUB_AVAILABLE,
        "faiss_available": FAISS_AVAILABLE,
        "virtual_env": os.environ.get('VIRTUAL_ENV'),
        "mode": "full" if TF_AVAILABLE else "simple"
    }

if __name__ == "__main__":
    print("Enhanced Video Search API - Unified Server")
    print("=" * 50)
    print(f"Mode: {'Full (TensorFlow Hub)' if TF_AVAILABLE else 'Simple'}")
    print(f"FAISS: {'Available' if FAISS_AVAILABLE else 'Not available'}")
    print()
    print("Access at: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
