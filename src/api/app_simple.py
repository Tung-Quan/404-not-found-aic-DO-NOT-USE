#!/usr/bin/env python3
"""
Simple API Server without TensorFlow dependencies
For testing purposes when TensorFlow has issues
"""

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import uvicorn
import os
from pathlib import Path

app = FastAPI(title="Enhanced Video Search API - Simple Mode")

@app.get("/")
async def root():
    return {"message": "Enhanced Video Search API - Simple Mode", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "mode": "simple",
        "tensorflow_hub": False,
        "message": "Running without TensorFlow dependencies"
    }

@app.get("/search")
async def search_videos(query: str = Query(..., description="Search query")):
    return {
        "query": query,
        "results": [],
        "message": "TensorFlow Hub not available - using simple mode",
        "suggestion": "Install TensorFlow Hub for enhanced search capabilities"
    }

@app.get("/system/status")
async def system_status():
    return {
        "virtual_env": "VIRTUAL_ENV" in os.environ,
        "tensorflow_hub": False,
        "faiss_available": False,
        "mode": "simple",
        "message": "Simple mode - basic functionality only"
    }

if __name__ == "__main__":
    print("Starting Enhanced Video Search API - Simple Mode")
    print("=" * 50)
    print("Access at: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print()
    print("Note: This is simple mode without TensorFlow")
    print("Install TensorFlow Hub for full functionality")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
