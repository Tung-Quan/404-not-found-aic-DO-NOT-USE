"""
🎯 WEB INTERFACE FOR ENHANCED VIDEO SEARCH
==========================================
Flask web app để demo enhanced search system
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import faiss
import json
import os
import time
from typing import List, Dict, Tuple
import re

app = Flask(__name__)

class WebVideoSearch:
    """Web-based video search system"""
    
    def __init__(self):
        self.faiss_index = None
        self.metadata = None
        self.enhanced_metadata = None
        self.frame_embeddings = None
        self.is_loaded = False
        
    def load_system(self):
        """Load all components"""
        try:
            print("Loading system...")
            
            # Load metadata
            self.metadata = pd.read_parquet('index/meta.parquet')
            
            # Load enhanced metadata
            if os.path.exists('index/frames_meta.json'):
                with open('index/frames_meta.json', 'r', encoding='utf-8') as f:
                    self.enhanced_metadata = json.load(f)
            
            # Load FAISS index
            if os.path.exists('index/faiss/ip_flat_chinese_clip.index'):
                self.faiss_index = faiss.read_index('index/faiss/ip_flat_chinese_clip.index')
            elif os.path.exists('index/faiss/ip_flat.index'):
                self.faiss_index = faiss.read_index('index/faiss/ip_flat.index')
            
            # Load embeddings
            N = len(self.metadata)
            if os.path.exists('index/embeddings/frames_chinese_clip.f16.mmap'):
                self.frame_embeddings = np.memmap(
                    'index/embeddings/frames_chinese_clip.f16.mmap',
                    dtype='float16', mode='r', shape=(N, 512)
                ).astype('float32')
            
            self.is_loaded = True
            print(f"System loaded: {len(self.metadata)} frames")
            return True
            
        except Exception as e:
            print(f"Error loading: {e}")
            return False
    
    def analyze_query(self, query: str) -> Dict:
        """Analyze query"""
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
    
    def find_query_embedding(self, query: str) -> np.ndarray:
        """Find best embedding for query"""
        query_words = set(query.lower().split())
        best_score = 0
        best_embedding = None
        
        if self.enhanced_metadata:
            for i, meta in enumerate(self.enhanced_metadata):
                if i >= len(self.frame_embeddings):
                    break
                
                title_words = set(meta.get('tx', '').lower().split())
                label_words = set([label.lower() for label in meta.get('labels_obj', [])])
                all_words = title_words.union(label_words)
                
                if len(query_words) > 0:
                    common_words = query_words.intersection(all_words)
                    similarity = len(common_words) / len(query_words)
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_embedding = self.frame_embeddings[i].copy()
        
        if best_embedding is None:
            best_embedding = np.mean(self.frame_embeddings[:50], axis=0)
        
        return best_embedding
    
    def search_similar(self, query_embedding: np.ndarray, topk: int = 50) -> List[Tuple[int, float]]:
        """Search similar frames"""
        if self.faiss_index is None:
            return []
        
        query_embedding = query_embedding.copy()
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        
        scores, indices = self.faiss_index.search(query_embedding.reshape(1, -1), topk)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append((int(idx), float(score)))
        
        return results
    
    def score_text(self, query: str) -> np.ndarray:
        """Score text relevance"""
        scores = np.zeros(len(self.metadata))
        query_words = set(query.lower().split())
        
        if not self.enhanced_metadata:
            return scores
        
        for i, meta in enumerate(self.enhanced_metadata):
            if i >= len(scores):
                break
            
            title_words = set(meta.get('tx', '').lower().split())
            label_words = set([label.lower() for label in meta.get('labels_obj', [])])
            all_words = title_words.union(label_words)
            
            if len(query_words) > 0 and len(all_words) > 0:
                common_words = query_words.intersection(all_words)
                scores[i] = len(common_words) / len(query_words)
        
        return scores
    
    def search(self, query: str, topk: int = 10) -> Dict:
        """Enhanced search"""
        if not self.is_loaded:
            return {'error': 'System not loaded'}
        
        start_time = time.time()
        
        try:
            # Analyze query
            analysis = self.analyze_query(query)
            
            # Find query embedding
            query_embedding = self.find_query_embedding(query)
            
            # Visual similarity
            visual_hits = self.search_similar(query_embedding, topk * 2)
            visual_scores = np.zeros(len(self.metadata))
            for idx, score in visual_hits:
                if idx < len(visual_scores):
                    visual_scores[idx] = score
            
            # Text relevance
            text_scores = self.score_text(query)
            
            # Combine scores
            alpha, beta = 0.6, 0.4
            if len(analysis['topics']) > 0:
                alpha, beta = 0.4, 0.6  # Boost text for topical queries
            
            # Normalize
            if visual_scores.max() > 0:
                visual_scores = visual_scores / visual_scores.max()
            if text_scores.max() > 0:
                text_scores = text_scores / text_scores.max()
            
            final_scores = alpha * visual_scores + beta * text_scores
            
            # Get top results
            top_indices = np.argsort(-final_scores)[:topk]
            
            results = []
            for idx in top_indices:
                if final_scores[idx] > 0:
                    row = self.metadata.iloc[idx]
                    
                    result = {
                        'frame_id': int(idx),
                        'video': row['video_id'],
                        'timestamp': int(row['ts']),
                        'time_formatted': f"{row['ts'] // 60}:{row['ts'] % 60:02d}",
                        'frame_path': row['frame_path'],
                        'final_score': float(final_scores[idx]),
                        'visual_score': float(visual_scores[idx]),
                        'text_score': float(text_scores[idx])
                    }
                    
                    # Add metadata
                    if self.enhanced_metadata and idx < len(self.enhanced_metadata):
                        meta = self.enhanced_metadata[idx]
                        result['title'] = meta.get('tx', '')
                        result['labels'] = meta.get('labels_obj', [])
                    
                    results.append(result)
            
            search_time = (time.time() - start_time) * 1000
            
            return {
                'query': query,
                'analysis': analysis,
                'results': results,
                'search_time_ms': search_time,
                'total_results': len(results),
                'weights': {'alpha': alpha, 'beta': beta}
            }
            
        except Exception as e:
            return {'error': str(e)}

# Initialize search system
search_system = WebVideoSearch()

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/search')
def search():
    """Search endpoint"""
    query = request.args.get('q', '').strip()
    topk = int(request.args.get('topk', 10))
    
    if not query:
        return jsonify({'error': 'No query provided'})
    
    result = search_system.search(query, topk)
    return jsonify(result)

@app.route('/status')
def status():
    """System status"""
    return jsonify({
        'loaded': search_system.is_loaded,
        'frames': len(search_system.metadata) if search_system.metadata is not None else 0,
        'enhanced_metadata': len(search_system.enhanced_metadata) if search_system.enhanced_metadata else 0
    })

if __name__ == '__main__':
    print("🎯 Starting Enhanced Video Search Web App...")
    
    # Load system
    if search_system.load_system():
        print("✅ System ready!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load system")
