from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np, pandas as pd, faiss
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from typing import List
import os

# Import Vietnamese translator
try:
    from .vietnamese_translator import translate_vietnamese_query
    VIETNAMESE_SUPPORT = True
    print("Vietnamese translation support loaded")
except ImportError:
    try:
        from vietnamese_translator import translate_vietnamese_query
        VIETNAMESE_SUPPORT = True
        print("Vietnamese translation support loaded")
    except ImportError:
        VIETNAMESE_SUPPORT = False
        print("Vietnamese translation not available")

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

class VietnameseSearchResponse(BaseModel):
    original_query: str
    translated_query: str
    total_frames_searched: int
    results: List[FrameHit]
    translation_available: bool

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
    endpoints = {
        "message": "AI Video Search API - Memory Optimized",
        "docs": "/docs",
        "health": "/health",
        "search": "/search?q=your_query",
        "search_frames": "/search_frames?q=your_query&top_frames=5",
        "search_vietnamese": "/search_vietnamese?q=người_đang_đi_bộ&top_frames=5",
        "total_frames": N,
        "vietnamese_support": VIETNAMESE_SUPPORT
    }
    
    if VIETNAMESE_SUPPORT:
        endpoints["vietnamese_examples"] = [
            "/search_vietnamese?q=người đang đi bộ",
            "/search_vietnamese?q=xe hơi đang chạy", 
            "/search_vietnamese?q=cảnh thiên nhiên đẹp"
        ]
    
    return endpoints

@app.get('/search_frames', response_model=FrameSearchResponse)
def search_frames(q: str,
                 top_frames: int = Query(5, ge=1, le=100)):
    """
    Search for individual frames closest to query.
    Returns top N frames from potentially different videos.
    """
    frame_results = _search_frames_internal(q, top_frames)
    
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

# --- Vietnamese Search Endpoints
@app.get('/search_vietnamese', response_model=VietnameseSearchResponse)
def search_vietnamese_frames(q: str, top_frames: int = Query(5, ge=1, le=100)):
    """
    🇻🇳 Tìm kiếm frame bằng tiếng Việt
    
    Endpoint chuyên biệt cho query tiếng Việt:
    1. Tự động dịch tiếng Việt → tiếng Anh
    2. Sử dụng bản dịch để search với CLIP model
    3. Trả về kết quả với cả query gốc và bản dịch
    
    Examples:
    - /search_vietnamese?q=người đang đi bộ
    - /search_vietnamese?q=xe hơi đang chạy&top_frames=10
    """
    if not VIETNAMESE_SUPPORT:
        # Fallback: use original Vietnamese query directly with CLIP
        return {
            'original_query': q,
            'translated_query': q,
            'translation_available': False,
            'total_frames_searched': top_frames,
            'results': _search_frames_internal(q, top_frames)
        }
    
    try:
        # Translate Vietnamese to English
        original_query, translated_query = translate_vietnamese_query(q)
        
        # Use translated query for CLIP search
        search_query = translated_query if translated_query != original_query else original_query
        
        # Perform search
        results = _search_frames_internal(search_query, top_frames)
        
        return {
            'original_query': original_query,
            'translated_query': translated_query,
            'translation_available': True,
            'total_frames_searched': top_frames,
            'results': results
        }
        
    except Exception as e:
        # Fallback: use original query if translation fails
        print(f"Vietnamese search error: {e}")
        return {
            'original_query': q,
            'translated_query': q,
            'translation_available': False,
            'total_frames_searched': top_frames,
            'results': _search_frames_internal(q, top_frames)
        }

def _search_frames_internal(query: str, top_frames: int) -> List[dict]:
    """
    Internal function to perform frame search
    Used by both regular and Vietnamese endpoints
    """
    try:
        # Embed query
        qv = embed_text(query)[None, :]

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

        return frame_results
        
    except Exception as e:
        print(f"Frame search error: {e}")
        return []

print("✅ API server ready!")
print("🔧 Memory optimized version loaded")
print(f"📊 Ready to search {N} frames")
