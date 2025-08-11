from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np, pandas as pd, faiss
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from typing import List
import os

# --- Load metadata & vectors (lazy loading for memory efficiency)
META = pd.read_parquet('index/meta.parquet')
N = len(META)

# Don't load vectors into memory immediately - use lazy loading
VEC_PATH = 'index/embeddings/frames.f16.mmap'
VEC_SHAPE = (N, 512)

# Load FAISS index (this is smaller and safer)
INDEX = faiss.read_index('index/faiss/ip_flat.index')

# --- Optional TF-IDF (per video)
try:
    with open('index/tfidf.pkl','rb') as f:
        VEC_TOK, X_TFIDF, VIDEO_IDS = pickle.load(f)
    HAS_TFIDF = True
except Exception:
    HAS_TFIDF = False

# --- Text encoder (CLIP)
import torch
from transformers import AutoProcessor, AutoModel
MODEL_ID = 'openai/clip-vit-base-patch32'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PROC = AutoProcessor.from_pretrained(MODEL_ID)
MODEL = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()

def embed_text(q: str) -> np.ndarray:
    with torch.no_grad():
        ins = PROC(text=[q], return_tensors='pt').to(DEVICE)
        out = MODEL.get_text_features(**ins)
        out = torch.nn.functional.normalize(out, dim=-1)
        return out[0].detach().cpu().numpy().astype('float32')

def load_vectors_chunk(start_idx: int, end_idx: int) -> np.ndarray:
    """Load a chunk of vectors to avoid memory issues."""
    try:
        # Load only the needed chunk
        chunk_size = end_idx - start_idx
        vec_chunk = np.memmap(VEC_PATH, dtype='float16', mode='r', 
                             shape=(N, 512), offset=start_idx * 512 * 2)[start_idx:end_idx]
        return vec_chunk.astype('float32')
    except Exception as e:
        print(f"Warning: Could not load vector chunk {start_idx}:{end_idx}, error: {e}")
        return np.zeros((end_idx - start_idx, 512), dtype='float32')

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

app = FastAPI(title='Text→Video Retrieval API')

@app.get('/health')
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "AI Video Search API is running"}

@app.get('/')
def root():
    """Root endpoint with basic info"""
    return {
        "message": "AI Video Search API",
        "docs": "/docs",
        "health": "/health",
        "search": "/search?q=your_query",
        "search_frames": "/search_frames?q=your_query&top_frames=5"
    }

@app.get('/search', response_model=SearchResponse)
def search(q: str,
           clip_weight: float = Query(1.0, ge=0.0),
           query_weight: float = Query(0.0, ge=0.0),
           topk_mean: int = Query(200, ge=1, le=2000),
           topk_frames: int = Query(8000, ge=100, le=50000)):
    # Embed query
    qv = embed_text(q)[None, :]

    # Dense frame search
    scores, idx = INDEX.search(qv, topk_frames)
    scores, idx = scores[0], idx[0]

    # Group by video_id → top-K mean + track best frames
    from collections import defaultdict
    per_video_dense = defaultdict(list)
    per_video_frames = defaultdict(list)  # Track frame info
    for s, i in zip(scores, idx):
        row = META.iloc[i]
        vid = row['video_id']
        per_video_dense[vid].append(float(s))
        per_video_frames[vid].append({
            'score': float(s),
            'frame_path': row['frame_path'],
            'timestamp': row['ts']  # Changed from 'timestamp' to 'ts'
        })

    def topkmean(arr, k):
        if not arr:
            return 0.0
        arr = sorted(arr, reverse=True)
        k = min(k, len(arr))
        return float(np.mean(arr[:k]))

    dense_score = {vid: topkmean(vs, topk_mean) for vid, vs in per_video_dense.items()}

    # Lexical TF‑IDF (optional)
    lex_score = {}
    if HAS_TFIDF and query_weight > 0:
        qx = VEC_TOK.transform([q.lower()])
        S = cosine_similarity(qx, X_TFIDF).A[0]
        for vid, s in zip(VIDEO_IDS, S):
            if s > 0:
                lex_score[vid] = float(s)

    # Normalize scores to [0,1]
    def norm_dict(d):
        if not d:
            return {}
        v = np.array(list(d.values()), dtype='float32')
        lo, hi = np.percentile(v, 1), np.percentile(v, 99)
        if hi <= lo + 1e-9:
            return {k: 0.0 for k in d}
        return {k: float(np.clip(val, lo, hi) - lo) / float(hi - lo) for k, val in d.items()}

    d_dense = norm_dict(dense_score)
    d_lex = norm_dict(lex_score)

    # Combine with weights and add video paths
    vids = set(d_dense) | set(d_lex)
    combined = []
    for vid in vids:
        s = clip_weight * d_dense.get(vid, 0.0) + query_weight * d_lex.get(vid, 0.0)
        
        # Get all frames for this video and sort by score
        frames_info = per_video_frames.get(vid, [])
        frames_info_sorted = sorted(frames_info, key=lambda x: x['score'], reverse=True)
        
        # Get best frame
        best_frame = frames_info_sorted[0] if frames_info_sorted else None
        
        # Get top 5 frames for this video
        top_5_frames = []
        for frame in frames_info_sorted[:5]:
            top_5_frames.append({
                'frame_path': frame['frame_path'],
                'timestamp': frame['timestamp'],
                'score': frame['score']
            })
        
        # Construct video path
        video_path = f"videos/{vid}"
        
        combined.append((vid, s, len(per_video_dense.get(vid, [])), video_path, best_frame, top_5_frames))

    combined.sort(key=lambda x: x[1], reverse=True)

    return {
        'query': q,
        'clip_weight': clip_weight,
        'query_weight': query_weight,
        'topk_mean': topk_mean,
        'results': [
            {
                'video_id': vid, 
                'video_path': video_path,
                'score': float(score), 
                'frames_used': n,
                'best_frame_path': best_frame['frame_path'] if best_frame else None,
                'best_frame_timestamp': best_frame['timestamp'] if best_frame else None,
                'top_frames': top_5_frames
            }
            for vid, score, n, video_path, best_frame, top_5_frames in combined[:50]
        ]
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

    # Search for top frames
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