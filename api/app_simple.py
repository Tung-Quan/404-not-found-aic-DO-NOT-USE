from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np, pandas as pd, faiss
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from typing import List
import os

print("Loading metadata...")
META = pd.read_parquet('index/meta.parquet')
N = len(META)
print(f"Loaded {N} frames metadata")

print("Loading FAISS index...")
INDEX = faiss.read_index('index/faiss/ip_flat.index')
print("FAISS index loaded successfully")

# Vector file path (don't load into memory immediately)
VEC_PATH = 'index/embeddings/frames.f16.mmap'

# --- Optional TF-IDF (per video)
try:
    with open('index/tfidf.pkl','rb') as f:
        VEC_TOK, X_TFIDF, VIDEO_IDS = pickle.load(f)
    HAS_TFIDF = True
    print("TF-IDF loaded successfully")
except Exception:
    HAS_TFIDF = False
    print("TF-IDF not available")

# --- Text encoder (CLIP)
print("Loading CLIP model...")
import torch
from transformers import AutoProcessor, AutoModel
MODEL_ID = 'openai/clip-vit-base-patch32'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PROC = AutoProcessor.from_pretrained(MODEL_ID)
MODEL = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
print(f"CLIP model loaded on {DEVICE}")

def embed_text(q: str) -> np.ndarray:
    with torch.no_grad():
        ins = PROC(text=[q], return_tensors='pt').to(DEVICE)
        out = MODEL.get_text_features(**ins)
        out = torch.nn.functional.normalize(out, dim=-1)
        return out[0].detach().cpu().numpy().astype('float32')

# --- API schema
class FrameDetail(BaseModel):
    frame_path: str
    timestamp: float
    score: float

class Hit(BaseModel):
    video_id: str
    video_path: str
    score: float
    frames_used: int
    best_frame_path: str = None
    best_frame_timestamp: float = None
    top_frames: List[FrameDetail] = []

class FrameHit(BaseModel):
    frame_path: str
    video_id: str
    timestamp: float
    score: float
    video_path: str

class SearchResponse(BaseModel):
    query: str
    clip_weight: float
    query_weight: float
    topk_mean: int
    results: List[Hit]

class FrameSearchResponse(BaseModel):
    query: str
    total_frames_searched: int
    results: List[FrameHit]

app = FastAPI(title='Text→Video Retrieval API - Memory Optimized')

@app.get('/health')
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "message": "AI Video Search API is running",
        "memory_optimized": True,
        "total_frames": N,
        "device": DEVICE
    }

@app.get('/')
def root():
    """Root endpoint with basic info"""
    return {
        "message": "AI Video Search API - Memory Optimized",
        "docs": "/docs",
        "health": "/health",
        "search": "/search?q=your_query",
        "search_frames": "/search_frames?q=your_query&top_frames=5",
        "total_frames": N
    }

@app.get('/search_frames', response_model=FrameSearchResponse)
def search_frames(q: str,
                 top_frames: int = Query(5, ge=1, le=100)):
    """
    Search for individual frames closest to query.
    Returns top N frames from potentially different videos.
    """
    # Embed query
    qv = embed_text(q)[None, :]

    # Search for top frames using FAISS
    scores, idx = INDEX.search(qv, top_frames)
    scores, idx = scores[0], idx[0]

    # Build frame results
    frame_results = []
    for score, frame_idx in zip(scores, idx):
        row = META.iloc[frame_idx]
        video_id = row['video_id']
        
        frame_results.append({
            'frame_path': row['frame_path'],
            'video_id': video_id,
            'timestamp': float(row['ts']),
            'score': float(score),
            'video_path': f"videos/{video_id}"
        })

    return {
        'query': q,
        'total_frames_searched': top_frames,
        'results': frame_results
    }

# Simple search endpoint without memory-heavy operations
@app.get('/search_simple')
def search_simple(q: str, top_k: int = Query(10, ge=1, le=100)):
    """Simplified search using only FAISS index"""
    try:
        # Embed query
        qv = embed_text(q)[None, :]
        
        # Search using FAISS
        scores, idx = INDEX.search(qv, top_k)
        scores, idx = scores[0], idx[0]
        
        # Group by video and get top result per video
        from collections import defaultdict
        per_video = defaultdict(list)
        
        for score, frame_idx in zip(scores, idx):
            row = META.iloc[frame_idx]
            video_id = row['video_id']
            per_video[video_id].append({
                'score': float(score),
                'timestamp': float(row['ts']),
                'frame_path': row['frame_path']
            })
        
        # Get best frame per video
        results = []
        for video_id, frames in per_video.items():
            best_frame = max(frames, key=lambda x: x['score'])
            results.append({
                'video_id': video_id,
                'video_path': f'videos/{video_id}',
                'best_score': best_frame['score'],
                'best_timestamp': best_frame['timestamp'],
                'frame_count': len(frames)
            })
        
        # Sort by best score
        results.sort(key=lambda x: x['best_score'], reverse=True)
        
        return {
            'query': q,
            'total_searched': top_k,
            'videos_found': len(results),
            'results': results[:10]  # Top 10 videos
        }
        
    except Exception as e:
        return {'error': str(e)}

# --- File serving endpoints
@app.get('/videos/{video_filename}')
def get_video(video_filename: str):
    """Serve video files"""
    video_path = os.path.join('videos', video_filename)
    if os.path.exists(video_path):
        return FileResponse(video_path)
    return {"error": "Video not found"}

@app.get('/frames/{video_id}/{frame_filename}')
def get_frame(video_id: str, frame_filename: str):
    """Serve frame image files"""
    frame_path = os.path.join('frames', video_id, frame_filename)
    if os.path.exists(frame_path):
        return FileResponse(frame_path)
    return {"error": "Frame not found"}

@app.get('/thumbnail/{video_id}')
def get_thumbnail(video_id: str):
    """Get first frame as thumbnail"""
    frame_dir = os.path.join('frames', video_id)
    if os.path.exists(frame_dir):
        frames = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
        if frames:
            return FileResponse(os.path.join(frame_dir, frames[0]))
    return {"error": "Thumbnail not found"}

print("✅ API server ready!")
print("🔧 Memory optimized version loaded")
print(f"📊 Ready to search {N} frames")
