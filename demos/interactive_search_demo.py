"""
🎯 INTERACTIVE ENHANCED VIDEO SEARCH DEMO
==========================================
Real-time search với enhanced scoring system
Kết hợp Chinese-CLIP embeddings + metadata scoring
"""

import numpy as np
import pandas as pd
import faiss
import json
import os
import time
from typing import List, Dict, Tuple
import re

class InteractiveVideoSearch:
    """Interactive search system với enhanced features"""
    
    def __init__(self):
        self.faiss_index = None
        self.metadata = None
        self.enhanced_metadata = None
        self.frame_embeddings = None
        self.video_stats = {}
        
    def load_system(self):
        """Load system components"""
        print("🔄 Initializing Enhanced Video Search System...")
        
        try:
            # Load metadata
            print("📊 Loading metadata...")
            self.metadata = pd.read_parquet('index/meta.parquet')
            print(f"   ✅ Loaded {len(self.metadata)} frame records")
            
            # Load enhanced metadata
            print("📋 Loading enhanced metadata...")
            if os.path.exists('index/frames_meta.json'):
                with open('index/frames_meta.json', 'r', encoding='utf-8') as f:
                    self.enhanced_metadata = json.load(f)
                print(f"   ✅ Enhanced metadata: {len(self.enhanced_metadata)} frames")
            else:
                print("   ⚠️  No enhanced metadata found")
            
            # Load FAISS index
            print("🔍 Loading FAISS index...")
            if os.path.exists('index/faiss/ip_flat_chinese_clip.index'):
                self.faiss_index = faiss.read_index('index/faiss/ip_flat_chinese_clip.index')
                print("   ✅ Chinese-CLIP FAISS index loaded")
            elif os.path.exists('index/faiss/ip_flat.index'):
                self.faiss_index = faiss.read_index('index/faiss/ip_flat.index')
                print("   ✅ Original FAISS index loaded")
            else:
                print("   ❌ No FAISS index found!")
                return False
            
            # Load embeddings
            print("💾 Loading frame embeddings...")
            N = len(self.metadata)
            if os.path.exists('index/embeddings/frames_chinese_clip.f16.mmap'):
                self.frame_embeddings = np.memmap(
                    'index/embeddings/frames_chinese_clip.f16.mmap',
                    dtype='float16', mode='r', shape=(N, 512)
                ).astype('float32')
                print("   ✅ Chinese-CLIP embeddings loaded")
            else:
                print("   ❌ No embeddings found!")
                return False
            
            # Build video statistics
            self.build_video_stats()
            
            print("🎉 System loaded successfully!")
            print(f"📊 Stats: {len(self.video_stats)} videos, {N} frames")
            return True
            
        except Exception as e:
            print(f"❌ Error loading system: {e}")
            return False
    
    def build_video_stats(self):
        """Build video statistics for better search"""
        self.video_stats = {}
        
        for i, row in self.metadata.iterrows():
            video_id = row['video_id']
            if video_id not in self.video_stats:
                self.video_stats[video_id] = {
                    'frame_count': 0,
                    'max_timestamp': 0,
                    'frame_indices': []
                }
            
            self.video_stats[video_id]['frame_count'] += 1
            self.video_stats[video_id]['max_timestamp'] = max(
                self.video_stats[video_id]['max_timestamp'], 
                row['ts']
            )
            self.video_stats[video_id]['frame_indices'].append(i)
    
    def analyze_query(self, query: str) -> Dict:
        """Enhanced query analysis"""
        analysis = {
            'original_query': query,
            'cleaned_query': query.strip().lower(),
            'language': 'unknown',
            'topics': [],
            'keywords': [],
            'query_type': 'general'
        }
        
        # Language detection
        vietnamese_chars = len(re.findall(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', query.lower()))
        if vietnamese_chars > 0:
            analysis['language'] = 'vietnamese'
        elif re.search(r'[a-zA-Z]', query):
            analysis['language'] = 'english'
        
        # Topic detection
        topics_map = {
            'programming': ['react', 'javascript', 'js', 'code', 'coding', 'programming', 'lập trình', 'dev', 'development'],
            'tutorial': ['tutorial', 'lesson', 'guide', 'hướng dẫn', 'học', 'learn', 'how to'],
            'backend': ['backend', 'api', 'server', 'database', 'node'],
            'ecommerce': ['ecommerce', 'shop', 'store', 'commerce', 'shopping'],
            'frontend': ['frontend', 'ui', 'css', 'html', 'interface'],
            'deploy': ['deploy', 'deployment', 'aws', 'cloud', 'hosting']
        }
        
        query_lower = query.lower()
        
        for topic, keywords in topics_map.items():
            for keyword in keywords:
                if keyword in query_lower:
                    if topic not in analysis['topics']:
                        analysis['topics'].append(topic)
                    analysis['keywords'].append(keyword)
        
        # Query type
        if len(analysis['topics']) > 0:
            analysis['query_type'] = 'topical'
        elif any(word in query_lower for word in ['who', 'what', 'where', 'when', 'how']):
            analysis['query_type'] = 'question'
        
        return analysis
    
    def find_best_query_embedding(self, query: str) -> Tuple[np.ndarray, float]:
        """Find best matching embedding with confidence score"""
        query_words = set(query.lower().split())
        
        best_score = 0
        best_embedding = None
        candidates = []
        
        if self.enhanced_metadata:
            for i, meta in enumerate(self.enhanced_metadata):
                if i >= len(self.frame_embeddings):
                    break
                
                # Score calculation
                title_words = set(meta.get('tx', '').lower().split())
                label_words = set([label.lower() for label in meta.get('labels_obj', [])])
                all_words = title_words.union(label_words)
                
                if len(query_words) > 0 and len(all_words) > 0:
                    common_words = query_words.intersection(all_words)
                    overlap_score = len(common_words) / len(query_words)
                    jaccard_score = len(common_words) / len(query_words.union(all_words))
                    
                    final_score = 0.7 * overlap_score + 0.3 * jaccard_score
                    
                    if final_score > 0:
                        candidates.append((i, final_score, meta.get('tx', '')))
        
        # Select best candidate
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_idx, best_score, best_text = candidates[0]
            best_embedding = self.frame_embeddings[best_idx].copy()
            
            print(f"🎯 Best match: '{best_text[:50]}...' (score: {best_score:.3f})")
        else:
            # Fallback: use centroid of random sample
            sample_indices = np.random.choice(len(self.frame_embeddings), min(50, len(self.frame_embeddings)), replace=False)
            best_embedding = np.mean(self.frame_embeddings[sample_indices], axis=0)
            best_score = 0.1
            print("🔄 Using centroid embedding (fallback)")
        
        return best_embedding, best_score
    
    def search_similar_frames(self, query_embedding: np.ndarray, topk: int = 100) -> List[Tuple[int, float]]:
        """Search similar frames using FAISS"""
        if self.faiss_index is None:
            return []
        
        # Normalize
        query_embedding = query_embedding.copy()
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        
        # Search
        scores, indices = self.faiss_index.search(query_embedding.reshape(1, -1), topk)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append((int(idx), float(score)))
        
        return results
    
    def score_text_relevance(self, query: str) -> np.ndarray:
        """Enhanced text relevance scoring"""
        scores = np.zeros(len(self.metadata))
        query_words = set(query.lower().split())
        
        if not self.enhanced_metadata:
            return scores
        
        for i, meta in enumerate(self.enhanced_metadata):
            if i >= len(scores):
                break
            
            # Multiple text sources
            title_text = meta.get('tx', '').lower()
            labels = meta.get('labels_obj', [])
            
            # Word matching
            title_words = set(title_text.split())
            label_words = set([label.lower() for label in labels])
            all_words = title_words.union(label_words)
            
            if len(query_words) > 0 and len(all_words) > 0:
                # Multiple scoring methods
                overlap = len(query_words.intersection(all_words)) / len(query_words)
                jaccard = len(query_words.intersection(all_words)) / len(query_words.union(all_words))
                
                # Phrase matching bonus
                phrase_bonus = 0
                if len(query.split()) > 1:
                    if query.lower() in title_text:
                        phrase_bonus = 0.3
                
                scores[i] = 0.6 * overlap + 0.3 * jaccard + phrase_bonus
        
        return scores
    
    def enhanced_search(self, query: str, topk: int = 10, alpha: float = 0.6, beta: float = 0.4) -> Dict:
        """Enhanced multi-modal search"""
        start_time = time.time()
        
        try:
            # 1. Query analysis
            analysis = self.analyze_query(query)
            
            # 2. Find query embedding
            query_embedding, embedding_confidence = self.find_best_query_embedding(query)
            
            # 3. Visual similarity
            visual_hits = self.search_similar_frames(query_embedding, topk * 3)
            visual_scores = np.zeros(len(self.metadata))
            for idx, score in visual_hits:
                if idx < len(visual_scores):
                    visual_scores[idx] = score
            
            # 4. Text relevance
            text_scores = self.score_text_relevance(query)
            
            # 5. Dynamic weight adjustment
            if analysis['query_type'] == 'topical' and len(analysis['topics']) > 0:
                # Boost text scores for topical queries
                alpha, beta = 0.4, 0.6
            
            if embedding_confidence < 0.3:
                # Low confidence in visual embedding, boost text
                alpha, beta = 0.3, 0.7
            
            # 6. Score combination
            # Normalize
            if visual_scores.max() > 0:
                visual_scores = visual_scores / visual_scores.max()
            if text_scores.max() > 0:
                text_scores = text_scores / text_scores.max()
            
            final_scores = alpha * visual_scores + beta * text_scores
            
            # 7. Apply boosts
            for i, row in self.metadata.iterrows():
                video_id = row['video_id']
                timestamp = row['ts']
                
                # Topic relevance boost
                boost = 0
                for topic in analysis['topics']:
                    if topic.lower() in video_id.lower():
                        boost += 0.15
                
                # Early content boost (first 10 minutes)
                if timestamp < 600:
                    boost += 0.05
                
                # Apply boost
                final_scores[i] = min(1.0, final_scores[i] + boost)
            
            # 8. Get top results
            top_indices = np.argsort(-final_scores)[:topk]
            
            # 9. Format results with diversity
            results = []
            seen_videos = set()
            
            for idx in top_indices:
                if final_scores[idx] > 0:
                    row = self.metadata.iloc[idx]
                    video_id = row['video_id']
                    
                    # Add diversity: limit results per video
                    video_count = sum(1 for r in results if r['video'] == video_id)
                    if video_count >= 3:  # Max 3 results per video
                        continue
                    
                    # Enhanced result info
                    result = {
                        'frame_id': int(idx),
                        'video': video_id,
                        'timestamp': int(row['ts']),
                        'time_formatted': f"{row['ts'] // 60}:{row['ts'] % 60:02d}",
                        'frame_path': row['frame_path'],
                        'final_score': float(final_scores[idx]),
                        'visual_score': float(visual_scores[idx]),
                        'text_score': float(text_scores[idx]),
                        'weights': {'alpha': alpha, 'beta': beta}
                    }
                    
                    # Add enhanced metadata if available
                    if self.enhanced_metadata and idx < len(self.enhanced_metadata):
                        meta = self.enhanced_metadata[idx]
                        result['title'] = meta.get('tx', '')
                        result['labels'] = meta.get('labels_obj', [])
                    
                    results.append(result)
            
            search_time = (time.time() - start_time) * 1000
            
            return {
                'query': query,
                'query_analysis': analysis,
                'embedding_confidence': embedding_confidence,
                'results': results,
                'search_time_ms': search_time,
                'total_results': len(results),
                'system_info': {
                    'weights': {'alpha': alpha, 'beta': beta},
                    'videos_searched': len(self.video_stats),
                    'frames_searched': len(self.metadata)
                }
            }
            
        except Exception as e:
            return {
                'query': query,
                'error': str(e),
                'results': [],
                'search_time_ms': (time.time() - start_time) * 1000
            }
    
    def print_search_results(self, result: Dict):
        """Pretty print search results"""
        print(f"\n🔍 Query: '{result['query']}'")
        print(f"⏱️  Search time: {result['search_time_ms']:.1f}ms")
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return
        
        analysis = result['query_analysis']
        print(f"🌐 Language: {analysis['language']}")
        print(f"📋 Topics: {analysis['topics']}")
        print(f"🎯 Query type: {analysis['query_type']}")
        print(f"💪 Embedding confidence: {result['embedding_confidence']:.3f}")
        print(f"⚖️  Weights: Visual {result['system_info']['weights']['alpha']:.1f} | Text {result['system_info']['weights']['beta']:.1f}")
        
        print(f"\n📊 Found {result['total_results']} results:")
        print("-" * 80)
        
        for i, res in enumerate(result['results'], 1):
            print(f"{i:2d}. 📺 {res['video'][:55]}...")
            print(f"    ⏰ {res['time_formatted']} | 🎯 Score: {res['final_score']:.3f} | 👁️ Visual: {res['visual_score']:.3f} | 📝 Text: {res['text_score']:.3f}")
            
            if 'title' in res and res['title']:
                print(f"    📄 Title: {res['title'][:70]}...")
            
            if 'labels' in res and res['labels']:
                labels_str = ', '.join(res['labels'][:3])
                print(f"    🏷️  Labels: {labels_str}")
            
            print()

def interactive_demo():
    """Interactive search demo"""
    print("🎯 INTERACTIVE ENHANCED VIDEO SEARCH")
    print("=" * 50)
    
    # Initialize system
    search = InteractiveVideoSearch()
    
    if not search.load_system():
        print("❌ Failed to initialize system")
        return
    
    print("\n🎉 System ready! Enter your search queries.")
    print("💡 Try queries like:")
    print("   - 'React tutorial'")
    print("   - 'backend development'") 
    print("   - 'ecommerce application'")
    print("   - 'JavaScript coding'")
    print("   - Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        try:
            query = input("\n🔍 Search: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            # Perform search
            result = search.enhanced_search(query, topk=5)
            
            # Display results
            search.print_search_results(result)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    interactive_demo()
