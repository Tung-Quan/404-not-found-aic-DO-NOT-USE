from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np, pandas as pd, faiss
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from typing import List, Optional
import os
import logging
import time
import re

# TensorFlow Hub imports
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow_text  # Required for USE multilingual
    TF_AVAILABLE = True
    print("✅ TensorFlow Hub available")
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow Hub not available")

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

# Import Enhanced Video Processor
try:
    from ..core.enhanced_video_processor import TensorFlowHubVideoManager, ModelType
    ENHANCED_VIDEO_PROCESSOR = TensorFlowHubVideoManager()
    print("✅ Enhanced Video Processor loaded")
except ImportError as e:
    try:
        import sys
        sys.path.append('../..')
        from src.core.enhanced_video_processor import TensorFlowHubVideoManager, ModelType
        ENHANCED_VIDEO_PROCESSOR = TensorFlowHubVideoManager()
        print("✅ Enhanced Video Processor loaded")
    except ImportError as e:
        ENHANCED_VIDEO_PROCESSOR = None
        print(f"⚠️  Enhanced Video Processor not available: {e}")

# --- Load metadata & vectors (lazy loading for memory efficiency)
META = pd.read_parquet('../index/meta.parquet')
N = len(META)

# Load enhanced metadata
try:
    import json
    with open('../index/frames_meta.json', 'r', encoding='utf-8') as f:
        ENHANCED_META = json.load(f)
    print(f"✅ Enhanced metadata loaded: {len(ENHANCED_META)} records")
except Exception as e:
    ENHANCED_META = None
    print(f"⚠️  Enhanced metadata not available: {e}")

# Don't load vectors into memory immediately - use lazy loading
VEC_PATH = '../index/embeddings/frames_chinese_clip.f16.mmap'
VEC_SHAPE = (N, 512)

# Load FAISS index (this is smaller and safer)
INDEX = faiss.read_index('../index/faiss/ip_flat_chinese_clip.index')

# --- TensorFlow Hub Models (Global variables)
USE_MODEL = None
EFFICIENTNET_MODEL = None
TEXT_EMBEDDING_CACHE = {}

def load_tensorflow_hub_models():
    """Load TensorFlow Hub models"""
    global USE_MODEL, EFFICIENTNET_MODEL
    
    if not TF_AVAILABLE:
        print("⚠️  TensorFlow Hub not available")
        return False
    
    try:
        print("🔄 Loading TensorFlow Hub models...")
        
        # Universal Sentence Encoder Multilingual v3
        print("   Loading Universal Sentence Encoder...")
        USE_MODEL = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
        print("   ✅ USE Multilingual loaded")
        
        # EfficientNet V2 for visual features
        print("   Loading EfficientNet V2...")
        EFFICIENTNET_MODEL = hub.load("https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2")
        print("   ✅ EfficientNet V2 loaded")
        
        print("🎉 TensorFlow Hub models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load TF Hub models: {e}")
        return False

def encode_text_with_use(texts: List[str]) -> Optional[np.ndarray]:
    """Encode text using Universal Sentence Encoder"""
    global USE_MODEL, TEXT_EMBEDDING_CACHE
    
    if USE_MODEL is None:
        return None
    
    try:
        # Check cache
        cache_key = "|".join(texts)
        if cache_key in TEXT_EMBEDDING_CACHE:
            return TEXT_EMBEDDING_CACHE[cache_key]
        
        # Encode
        embeddings = USE_MODEL(texts)
        embeddings_np = embeddings.numpy()
        
        # Cache result
        TEXT_EMBEDDING_CACHE[cache_key] = embeddings_np
        
        return embeddings_np
    
    except Exception as e:
        print(f"❌ Error encoding with USE: {e}")
        return None

# --- Optional TF-IDF (per video)
try:
    with open('index/tfidf.pkl','rb') as f:
        VEC_TOK, X_TFIDF, VIDEO_IDS = pickle.load(f)
    HAS_TFIDF = True
except Exception:
    HAS_TFIDF = False

# --- Text encoder (CHINESE-CLIP)
import torch
from transformers import AutoProcessor, AutoModel
MODEL_ID = 'OFA-Sys/chinese-clip-vit-base-patch16'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PROC = AutoProcessor.from_pretrained(MODEL_ID)
MODEL = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
print(f"Chinese-CLIP model loaded on {DEVICE}")
print("🇻🇳 Vietnamese queries now optimized!")

def embed_text(q: str) -> np.ndarray:
    """
    Enhanced text embedding using TensorFlow Hub + Chinese-CLIP
    """
    # Try TensorFlow Hub first (better for multilingual)
    if USE_MODEL is not None:
        try:
            use_embedding = encode_text_with_use([q])
            if use_embedding is not None:
                # Normalize for similarity search
                embedding = use_embedding[0]
                return embedding / np.linalg.norm(embedding)
        except Exception as e:
            print(f"USE encoding failed, falling back to Chinese-CLIP: {e}")
    
    # Fallback to Chinese-CLIP
    with torch.no_grad():
        ins = PROC(text=[q], return_tensors='pt').to(DEVICE)
        out = MODEL.get_text_features(**ins)
        out = torch.nn.functional.normalize(out, dim=-1)
        return out[0].detach().cpu().numpy().astype('float32')

def analyze_query(query: str) -> dict:
    """Analyze query for enhanced processing"""
    analysis = {
        'original_query': query,
        'language': 'unknown',
        'topics': [],
        'keywords': []
    }
    
    # Language detection
    vietnamese_chars = len(re.findall(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', query.lower()))
    if vietnamese_chars > 0:
        analysis['language'] = 'vietnamese'
    elif re.search(r'[a-zA-Z]', query):
        analysis['language'] = 'english'
    
    # Topic detection
    topics_map = {
        'programming': ['react', 'javascript', 'js', 'code', 'coding', 'programming', 'lập trình'],
        'tutorial': ['tutorial', 'lesson', 'guide', 'hướng dẫn', 'học', 'learn'],
        'backend': ['backend', 'api', 'server', 'database'],
        'ecommerce': ['ecommerce', 'shop', 'store', 'commerce']
    }
    
    query_lower = query.lower()
    for topic, keywords in topics_map.items():
        for keyword in keywords:
            if keyword in query_lower:
                if topic not in analysis['topics']:
                    analysis['topics'].append(topic)
                analysis['keywords'].append(keyword)
    
    return analysis

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

class EnhancedSearchResponse(BaseModel):
    query: str
    query_analysis: dict
    tensorflow_hub_enabled: bool
    total_frames_searched: int
    search_time_ms: float
    results: List[FrameHit]
    system_info: dict

class VietnameseSearchResponse(BaseModel):
    original_query: str
    translated_query: str
    total_frames_searched: int
    results: List[FrameHit]
    translation_available: bool

class ModelSelectionRequest(BaseModel):
    user_intent: str
    max_memory_mb: int = 2000
    processing_priority: str = "balanced"  # lightweight, balanced, high_accuracy

class ModelSelectionResponse(BaseModel):
    recommendations: dict
    overlaps_detected: dict
    suggested_models: List[str]
    estimated_memory_usage: str

class VideoProcessingRequest(BaseModel):
    video_path: str
    query: str = ""
    selected_models: List[str]

class VideoProcessingResponse(BaseModel):
    video_path: str
    query: str
    processing_results: dict
    combined_features: dict
    processing_time: dict
    active_models: List[str]

app = FastAPI(title='Enhanced Text→Video Retrieval API with TensorFlow Hub')

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

@app.get('/enhanced_search', response_model=EnhancedSearchResponse)
def enhanced_search(q: str,
                   topk_frames: int = Query(50, ge=10, le=1000)):
    """
    Enhanced search using TensorFlow Hub + existing system
    """
    start_time = time.time()
    
    # Analyze query
    analysis = analyze_query(q)
    
    # Embed query với enhanced method
    qv = embed_text(q)[None, :]
    
    # Search using FAISS
    scores, idx = INDEX.search(qv, topk_frames)
    scores, idx = scores[0], idx[0]
    
    # Enhanced scoring with metadata
    enhanced_results = []
    
    for s, i in zip(scores, idx):
        if i >= len(META):
            continue
            
        row = META.iloc[i]
        
        # Base result
        result = FrameHit(
            frame_path=row['frame_path'],
            video_id=row['video_id'],
            timestamp=row['ts'],
            score=float(s),
            video_path=row.get('video_path', '')
        )
        
        # Enhanced scoring với metadata
        if ENHANCED_META and i < len(ENHANCED_META):
            meta = ENHANCED_META[i]
            
            # Text relevance boost
            text_boost = 0
            if analysis['keywords']:
                title_text = meta.get('tx', '').lower()
                for keyword in analysis['keywords']:
                    if keyword.lower() in title_text:
                        text_boost += 0.1
            
            # Early content boost
            if row['ts'] < 600:  # First 10 minutes
                text_boost += 0.05
            
            # Apply boost
            result.score = min(1.0, result.score + text_boost)
        
        enhanced_results.append(result)
    
    # Sort by enhanced score
    enhanced_results.sort(key=lambda x: x.score, reverse=True)
    
    search_time = (time.time() - start_time) * 1000
    
    return EnhancedSearchResponse(
        query=q,
        query_analysis=analysis,
        tensorflow_hub_enabled=USE_MODEL is not None,
        total_frames_searched=len(enhanced_results),
        search_time_ms=search_time,
        results=enhanced_results,
        system_info={
            'use_model_available': USE_MODEL is not None,
            'efficientnet_available': EFFICIENTNET_MODEL is not None,
            'enhanced_metadata_available': ENHANCED_META is not None,
            'chinese_clip_available': True
        }
    )

@app.post('/load_tensorflow_hub')
def load_tf_hub():
    """Load TensorFlow Hub models"""
    success = load_tensorflow_hub_models()
    return {
        'success': success,
        'message': 'TensorFlow Hub models loaded successfully' if success else 'Failed to load models',
        'models_available': {
            'use_model': USE_MODEL is not None,
            'efficientnet_model': EFFICIENTNET_MODEL is not None
        }
    }

@app.get('/status')
def get_status():
    """Get system status"""
    status = {
        'tensorflow_hub_available': TF_AVAILABLE,
        'use_model_loaded': USE_MODEL is not None,
        'efficientnet_model_loaded': EFFICIENTNET_MODEL is not None,
        'enhanced_metadata_available': ENHANCED_META is not None,
        'total_frames': len(META),
        'vietnamese_support': VIETNAMESE_SUPPORT,
        'enhanced_video_processor_available': ENHANCED_VIDEO_PROCESSOR is not None
    }
    
    # Add enhanced video processor status if available
    if ENHANCED_VIDEO_PROCESSOR:
        enhanced_status = ENHANCED_VIDEO_PROCESSOR.get_model_status()
        status['enhanced_models'] = enhanced_status
    
    return status

@app.post('/analyze_models', response_model=ModelSelectionResponse)
def analyze_model_requirements(request: ModelSelectionRequest):
    """Analyze user requirements and suggest optimal model combinations"""
    if not ENHANCED_VIDEO_PROCESSOR:
        return ModelSelectionResponse(
            recommendations={},
            overlaps_detected={},
            suggested_models=[],
            estimated_memory_usage="Enhanced Video Processor not available"
        )
    
    try:
        # Get recommendations from enhanced processor
        recommendations = ENHANCED_VIDEO_PROCESSOR.suggest_model_combinations(
            request.user_intent, 
            request.max_memory_mb
        )
        
        # Select models based on processing priority
        if request.processing_priority in recommendations:
            suggested_models = recommendations[request.processing_priority]
        else:
            suggested_models = recommendations.get('balanced', [])
        
        # Calculate estimated memory usage
        total_memory = 0
        memory_breakdown = []
        
        for model_name in suggested_models:
            if model_name in ENHANCED_VIDEO_PROCESSOR.model_configs:
                config = ENHANCED_VIDEO_PROCESSOR.model_configs[model_name]
                # Extract memory number (rough estimation)
                try:
                    memory_str = config.memory_usage.replace('~', '').replace('MB', '').replace('GB', '000')
                    memory_val = float(memory_str)
                    total_memory += memory_val
                    memory_breakdown.append(f"{config.name}: {config.memory_usage}")
                except:
                    memory_breakdown.append(f"{config.name}: {config.memory_usage}")
        
        estimated_memory = f"~{total_memory:.0f}MB ({', '.join(memory_breakdown)})"
        
        return ModelSelectionResponse(
            recommendations=recommendations,
            overlaps_detected=recommendations.get('overlaps_detected', {}),
            suggested_models=suggested_models,
            estimated_memory_usage=estimated_memory
        )
        
    except Exception as e:
        return ModelSelectionResponse(
            recommendations={},
            overlaps_detected={},
            suggested_models=[],
            estimated_memory_usage=f"Error analyzing requirements: {e}"
        )

@app.post('/load_selected_models')
def load_selected_models(model_names: List[str]):
    """Load user-selected models"""
    if not ENHANCED_VIDEO_PROCESSOR:
        return {
            'success': False,
            'message': 'Enhanced Video Processor not available',
            'results': {}
        }
    
    try:
        results = ENHANCED_VIDEO_PROCESSOR.load_selected_models(model_names)
        
        success_count = sum(1 for r in results.values() if r)
        total_count = len(results)
        
        return {
            'success': success_count > 0,
            'message': f'Loaded {success_count}/{total_count} models successfully',
            'results': results,
            'active_models': list(ENHANCED_VIDEO_PROCESSOR.active_models)
        }
        
    except Exception as e:
        return {
            'success': False,
            'message': f'Error loading models: {e}',
            'results': {}
        }

@app.post('/process_video', response_model=VideoProcessingResponse)
def process_video_enhanced(request: VideoProcessingRequest):
    """Process video with selected TensorFlow Hub models"""
    if not ENHANCED_VIDEO_PROCESSOR:
        return VideoProcessingResponse(
            video_path=request.video_path,
            query=request.query,
            processing_results={'error': 'Enhanced Video Processor not available'},
            combined_features={},
            processing_time={},
            active_models=[]
        )
    
    try:
        # Load selected models if not already loaded
        if request.selected_models:
            load_results = ENHANCED_VIDEO_PROCESSOR.load_selected_models(request.selected_models)
            print(f"Model loading results: {load_results}")
        
        # Process video
        results = ENHANCED_VIDEO_PROCESSOR.process_video_with_selected_models(
            request.video_path, 
            request.query
        )
        
        return VideoProcessingResponse(
            video_path=results['video_path'],
            query=results['query'],
            processing_results=results['processing_results'],
            combined_features=results.get('combined_features', {}),
            processing_time=results['processing_time'],
            active_models=list(ENHANCED_VIDEO_PROCESSOR.active_models)
        )
        
    except Exception as e:
        return VideoProcessingResponse(
            video_path=request.video_path,
            query=request.query,
            processing_results={'error': str(e)},
            combined_features={},
            processing_time={},
            active_models=[]
        )

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
    frame_results = _search_frames_advanced_internal(q, top_frames)
    
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

# --- Vietnamese Search Endpoints
@app.get('/search_vietnamese', response_model=VietnameseSearchResponse)
def search_vietnamese_frames(q: str, top_frames: int = Query(5, ge=1, le=100)):
    """
    🇻🇳 Tìm kiếm frame bằng tiếng Việt với Advanced Features
    
    Endpoint chuyên biệt cho query tiếng Việt:
    1. Tự động dịch tiếng Việt → tiếng Anh
    2. Sử dụng bản dịch để search với CLIP model
    3. Trả về kết quả với cả query gốc và bản dịch
    4. Hỗ trợ tất cả tính năng advanced của server
    
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
            'results': _search_frames_advanced_internal(q, top_frames)
        }
    
    try:
        # Translate Vietnamese to English
        original_query, translated_query = translate_vietnamese_query(q)
        
        # Use translated query for CLIP search
        search_query = translated_query if translated_query != original_query else original_query
        
        # Perform search
        results = _search_frames_advanced_internal(search_query, top_frames)
        
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
            'results': _search_frames_advanced_internal(q, top_frames)
        }

def _search_frames_advanced_internal(query: str, top_frames: int) -> List[dict]:
    """
    Internal function to perform advanced frame search
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
        print(f"Advanced frame search error: {e}")
        return []

@app.on_event("startup")
async def startup_event():
    """Initialize TensorFlow Hub models on startup"""
    print("🚀 Starting Enhanced Video Search API...")
    print(f"📊 System info:")
    print(f"   - Total frames: {len(META)}")
    print(f"   - Enhanced metadata: {'Available' if ENHANCED_META else 'Not available'}")
    print(f"   - Vietnamese support: {'Available' if VIETNAMESE_SUPPORT else 'Not available'}")
    print(f"   - TensorFlow Hub: {'Available' if TF_AVAILABLE else 'Not available'}")
    
    # Try to load TensorFlow Hub models
    if TF_AVAILABLE:
        print("🔄 Attempting to load TensorFlow Hub models...")
        success = load_tensorflow_hub_models()
        if success:
            print("✅ TensorFlow Hub models loaded successfully!")
        else:
            print("⚠️  TensorFlow Hub models failed to load, using Chinese-CLIP only")
    
    print("🎉 Enhanced Video Search API ready!")

if __name__ == "__main__":
    import uvicorn
    print("Starting Enhanced Video Search API with TensorFlow Hub integration...")
    uvicorn.run(app, host="0.0.0.0", port=8000)