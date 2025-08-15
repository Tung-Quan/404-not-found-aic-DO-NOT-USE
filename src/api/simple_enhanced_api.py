"""
🚀 SIMPLE ENHANCED API DEMO
===========================
FastAPI backend sử dụng hệ thống hiện tại với enhanced features
"""

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
import faiss
import json
import os
import time
import re
from typing import List, Optional

app = FastAPI(title="Enhanced Video Search API")

# Global variables
META = None
FAISS_INDEX = None
ENHANCED_META = None

class FrameResult(BaseModel):
    frame_path: str
    video_id: str
    timestamp: float
    score: float
    video_path: str

class SearchResponse(BaseModel):
    query: str
    query_analysis: dict
    total_frames_searched: int
    search_time_ms: float
    results: List[FrameResult]
    system_info: dict

def load_system():
    """Load search system components"""
    global META, FAISS_INDEX, ENHANCED_META
    
    try:
        print("🔄 Loading search system...")
        
        # Load metadata
        if os.path.exists('../index/meta.parquet'):
            META = pd.read_parquet('../index/meta.parquet')
            print(f"✅ Metadata: {len(META)} frames")
        else:
            print("❌ Metadata not found")
            return False
        
        # Load FAISS index
        if os.path.exists('../index/faiss/ip_flat_chinese_clip.index'):
            FAISS_INDEX = faiss.read_index('../index/faiss/ip_flat_chinese_clip.index')
            print("✅ FAISS index loaded")
        else:
            print("❌ FAISS index not found")
            return False
        
        # Load enhanced metadata
        if os.path.exists('../index/frames_meta.json'):
            with open('../index/frames_meta.json', 'r', encoding='utf-8') as f:
                ENHANCED_META = json.load(f)
            print(f"✅ Enhanced metadata: {len(ENHANCED_META)} records")
        else:
            print("⚠️  Enhanced metadata not found")
        
        print("🎉 System loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load system: {e}")
        return False

def analyze_query(query: str) -> dict:
    """Analyze query for enhanced processing"""
    analysis = {
        'original_query': query,
        'language': 'unknown',
        'topics': [],
        'keywords': [],
        'word_count': len(query.split())
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
        'ecommerce': ['ecommerce', 'shop', 'store', 'commerce'],
        'frontend': ['frontend', 'ui', 'css', 'html']
    }
    
    query_lower = query.lower()
    for topic, keywords in topics_map.items():
        for keyword in keywords:
            if keyword in query_lower:
                if topic not in analysis['topics']:
                    analysis['topics'].append(topic)
                analysis['keywords'].append(keyword)
    
    return analysis

def simple_text_similarity(query: str, text: str) -> float:
    """Simple text similarity using word overlap"""
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())
    
    if len(query_words) == 0:
        return 0.0
    
    common_words = query_words.intersection(text_words)
    return len(common_words) / len(query_words)

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Enhanced Video Search API",
        "endpoints": {
            "search": "/search?q=your_query",
            "status": "/status",
            "docs": "/docs"
        }
    }

@app.get("/status")
def get_status():
    """System status"""
    return {
        "system_loaded": META is not None,
        "total_frames": len(META) if META is not None else 0,
        "enhanced_metadata": ENHANCED_META is not None,
        "faiss_index": FAISS_INDEX is not None
    }

@app.get("/search", response_model=SearchResponse)
def search(q: str, topk: int = Query(10, ge=1, le=100)):
    """Enhanced search endpoint"""
    start_time = time.time()
    
    if META is None or FAISS_INDEX is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Search system not loaded"}
        )
    
    # Analyze query
    analysis = analyze_query(q)
    
    # Simple embedding simulation (since we don't have text encoder here)
    # In real implementation, this would use Chinese-CLIP or TensorFlow Hub
    results = []
    
    # Search through enhanced metadata if available
    if ENHANCED_META:
        scored_frames = []
        
        for i, meta in enumerate(ENHANCED_META):
            if i >= len(META):
                break
            
            # Text similarity
            title_text = meta.get('tx', '')
            text_score = simple_text_similarity(q, title_text)
            
            # Keyword boost
            keyword_boost = 0
            for keyword in analysis['keywords']:
                if keyword.lower() in title_text.lower():
                    keyword_boost += 0.2
            
            # Topic boost
            if analysis['topics']:
                for topic in analysis['topics']:
                    if topic.lower() in title_text.lower():
                        keyword_boost += 0.1
            
            final_score = text_score + keyword_boost
            
            if final_score > 0:
                scored_frames.append((i, final_score))
        
        # Sort by score
        scored_frames.sort(key=lambda x: x[1], reverse=True)
        
        # Take top results
        for frame_idx, score in scored_frames[:topk]:
            row = META.iloc[frame_idx]
            
            result = FrameResult(
                frame_path=row['frame_path'],
                video_id=row['video_id'],
                timestamp=float(row['ts']),
                score=float(score),
                video_path=f"videos/{row['video_id']}"
            )
            results.append(result)
    
    search_time = (time.time() - start_time) * 1000
    
    return SearchResponse(
        query=q,
        query_analysis=analysis,
        total_frames_searched=len(results),
        search_time_ms=search_time,
        results=results,
        system_info={
            "enhanced_metadata_used": ENHANCED_META is not None,
            "search_method": "text_similarity",
            "total_available_frames": len(META)
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Enhanced Video Search API...")
    
    # Load system
    if load_system():
        print("✅ System ready, starting server...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("❌ Failed to load system")
