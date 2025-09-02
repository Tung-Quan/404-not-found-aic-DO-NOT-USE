"""
🔄 Updated AI Search Engine - BLIP-2 Integration
Main search engine tích hợp BLIP-2 và hỗ trợ TensorFlow reranking
"""

import os
import numpy as np
import faiss
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Import new BLIP-2 manager
from blip2_core_manager import BLIP2CoreManager

# Import TensorFlow reranker (preserve existing)
try:
    from tensorflow_model_manager import TensorFlowModelManager
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("⚠️  TensorFlow model manager not available - reranking disabled")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedBLIP2SearchEngine:
    """
    🚀 Enhanced AI Search Engine với BLIP-2 + TensorFlow Reranking
    
    Architecture:
    1. BLIP-2 Primary Model: Vision-language understanding, complex query processing
    2. FAISS Vector Search: Fast similarity search
    3. TensorFlow Reranking: Secondary ranking với specialized models
    4. Complex Query Processor: Intent detection, entity extraction, temporal reasoning
    """
    
    def __init__(self, 
                 dataset_name: str = "mixed_collection",
                 blip2_model: str = "Salesforce/blip2-flan-t5-base",
                 enable_tensorflow_rerank: bool = True,
                 max_rerank_candidates: int = 100):
        
        self.dataset_name = dataset_name
        self.dataset_path = Path("datasets") / dataset_name
        self.index_path = Path("index")
        self.embeddings_path = Path("embeddings")
        
        # Model configurations
        self.blip2_model_name = blip2_model
        self.enable_tensorflow_rerank = enable_tensorflow_rerank and TENSORFLOW_AVAILABLE
        self.max_rerank_candidates = max_rerank_candidates
        
        # Initialize models
        self.blip2_manager = BLIP2CoreManager(
            model_name=blip2_model,
            device="auto",
            precision="fp16"
        )
        
        if self.enable_tensorflow_rerank:
            self.tf_manager = TensorFlowModelManager()
        else:
            self.tf_manager = None
        
        # Search infrastructure
        self.faiss_index = None
        self.image_paths = []
        self.metadata = {}
        
        # Performance tracking
        self.search_stats = {
            'total_searches': 0,
            'avg_blip2_time': 0.0,
            'avg_faiss_time': 0.0,
            'avg_rerank_time': 0.0,
            'cache_hits': 0
        }
        
        logger.info(f"🎯 EnhancedBLIP2SearchEngine initialized")
        logger.info(f"📁 Dataset: {dataset_name}")
        logger.info(f"🧠 BLIP-2 Model: {blip2_model}")
        logger.info(f"🔄 TensorFlow Rerank: {self.enable_tensorflow_rerank}")
    
    def setup_models(self) -> bool:
        """
        Setup all models với proper error handling
        """
        success = True
        
        # 1. Load BLIP-2
        logger.info("🚀 Loading BLIP-2 model...")
        if not self.blip2_manager.load_model():
            logger.error("❌ Failed to load BLIP-2 model")
            success = False
        
        # 2. Load TensorFlow models
        if self.enable_tensorflow_rerank:
            logger.info("🔄 Loading TensorFlow reranking models...")
            if not self.tf_manager.load_models():
                logger.warning("⚠️  TensorFlow models failed to load - continuing without reranking")
                self.enable_tensorflow_rerank = False
        
        return success
    
    def load_or_build_index(self, force_rebuild: bool = False) -> bool:
        """
        Load existing index hoặc build new index với BLIP-2
        """
        index_file = self.index_path / f"{self.dataset_name}_blip2.index"
        embeddings_file = self.embeddings_path / f"{self.dataset_name}_blip2_embeddings.npy"
        metadata_file = self.index_path / f"{self.dataset_name}_blip2_metadata.json"
        
        # Check if index exists và không force rebuild
        if not force_rebuild and all(f.exists() for f in [index_file, embeddings_file, metadata_file]):
            logger.info("📂 Loading existing BLIP-2 index...")
            return self._load_existing_index(index_file, embeddings_file, metadata_file)
        else:
            logger.info("🔨 Building new BLIP-2 index...")
            return self._build_new_index()
    
    def _load_existing_index(self, index_file: Path, embeddings_file: Path, metadata_file: Path) -> bool:
        """Load existing index files"""
        try:
            # Load FAISS index
            self.faiss_index = faiss.read_index(str(index_file))
            logger.info(f"✅ Loaded FAISS index: {self.faiss_index.ntotal} vectors")
            
            # Load embeddings
            embeddings = np.load(embeddings_file)
            logger.info(f"✅ Loaded embeddings: {embeddings.shape}")
            
            # Load metadata
            with open(metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            self.image_paths = self.metadata.get('image_paths', [])
            logger.info(f"✅ Loaded metadata: {len(self.image_paths)} images")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load existing index: {e}")
            return False
    
    def _build_new_index(self) -> bool:
        """Build new index với BLIP-2 embeddings"""
        try:
            # Find all images trong dataset
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(self.dataset_path.rglob(f"*{ext}"))
            
            if not image_files:
                logger.error(f"❌ No images found in {self.dataset_path}")
                return False
            
            logger.info(f"📸 Found {len(image_files)} images")
            self.image_paths = [str(img) for img in image_files]
            
            # Generate embeddings với BLIP-2
            logger.info("🧠 Generating BLIP-2 embeddings...")
            start_time = time.time()
            
            embeddings = self.blip2_manager.batch_encode_images(self.image_paths)
            
            embed_time = time.time() - start_time
            logger.info(f"⚡ Generated {len(embeddings)} embeddings in {embed_time:.2f}s")
            
            # Build FAISS index
            logger.info("🔍 Building FAISS index...")
            dimension = self.blip2_manager.get_embedding_dimension()
            
            # Use IVF index for large datasets
            if len(embeddings) > 1000:
                nlist = min(100, len(embeddings) // 10)
                quantizer = faiss.IndexFlatIP(dimension)
                self.faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                self.faiss_index.train(embeddings.astype(np.float32))
            else:
                self.faiss_index = faiss.IndexFlatIP(dimension)
            
            # Add embeddings to index
            self.faiss_index.add(embeddings.astype(np.float32))
            logger.info(f"✅ FAISS index built: {self.faiss_index.ntotal} vectors")
            
            # Save everything
            self._save_index(embeddings)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to build index: {e}")
            return False
    
    def _save_index(self, embeddings: np.ndarray):
        """Save index, embeddings, và metadata"""
        # Create directories
        self.index_path.mkdir(exist_ok=True)
        self.embeddings_path.mkdir(exist_ok=True)
        
        # Save FAISS index
        index_file = self.index_path / f"{self.dataset_name}_blip2.index"
        faiss.write_index(self.faiss_index, str(index_file))
        
        # Save embeddings
        embeddings_file = self.embeddings_path / f"{self.dataset_name}_blip2_embeddings.npy"
        np.save(embeddings_file, embeddings)
        
        # Save metadata
        metadata = {
            'dataset_name': self.dataset_name,
            'model_name': self.blip2_model_name,
            'image_paths': self.image_paths,
            'embedding_dimension': int(embeddings.shape[1]),
            'total_images': len(self.image_paths),
            'created_at': time.time()
        }
        
        metadata_file = self.index_path / f"{self.dataset_name}_blip2_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info("💾 Index saved successfully")
    
    def search(self, 
               query: str, 
               top_k: int = 20,
               use_reranking: bool = True,
               return_scores: bool = True) -> List[Dict]:
        """
        🔍 Main search function với 2-stage architecture:
        Stage 1: BLIP-2 similarity search
        Stage 2: TensorFlow reranking (optional)
        """
        start_time = time.time()
        self.search_stats['total_searches'] += 1
        
        # Stage 1: BLIP-2 Primary Search
        logger.info(f"🎯 Stage 1: BLIP-2 search for '{query}'")
        blip2_start = time.time()
        
        # Parse complex query
        parsed_query = self.blip2_manager.query_processor.parse_query(query)
        logger.info(f"📝 Query complexity: {parsed_query.get('complexity_score', 0):.2f}")
        
        # Encode query
        query_embedding = self.blip2_manager.encode_text(query)
        
        # FAISS search
        search_k = min(self.max_rerank_candidates, top_k * 5)  # Get more candidates for reranking
        similarities, indices = self.faiss_index.search(
            query_embedding.reshape(1, -1).astype(np.float32), 
            search_k
        )
        
        blip2_time = time.time() - blip2_start
        self.search_stats['avg_blip2_time'] = (
            self.search_stats['avg_blip2_time'] * (self.search_stats['total_searches'] - 1) + blip2_time
        ) / self.search_stats['total_searches']
        
        # Format Stage 1 results
        stage1_results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.image_paths):  # Valid index
                stage1_results.append({
                    'image_path': self.image_paths[idx],
                    'similarity': float(similarity),
                    'rank': i + 1,
                    'index': int(idx),
                    'stage': 'blip2'
                })
        
        # Stage 2: TensorFlow Reranking (optional)
        final_results = stage1_results
        
        if use_reranking and self.enable_tensorflow_rerank and len(stage1_results) > 1:
            logger.info(f"🔄 Stage 2: TensorFlow reranking {len(stage1_results)} candidates")
            rerank_start = time.time()
            
            try:
                # Prepare candidates for reranking
                candidate_paths = [r['image_path'] for r in stage1_results]
                
                # TensorFlow reranking
                reranked_results = self.tf_manager.rerank_results(
                    query=query,
                    image_paths=candidate_paths,
                    initial_scores=[r['similarity'] for r in stage1_results]
                )
                
                # Merge results
                final_results = []
                for i, rerank_item in enumerate(reranked_results[:top_k]):
                    original_item = stage1_results[rerank_item.get('original_index', i)]
                    final_results.append({
                        **original_item,
                        'tf_score': rerank_item.get('score', 0.0),
                        'final_score': rerank_item.get('score', original_item['similarity']),
                        'rank': i + 1,
                        'stage': 'reranked'
                    })
                
                rerank_time = time.time() - rerank_start
                self.search_stats['avg_rerank_time'] = (
                    self.search_stats['avg_rerank_time'] * (self.search_stats['total_searches'] - 1) + rerank_time
                ) / self.search_stats['total_searches']
                
                logger.info(f"✅ Reranking completed in {rerank_time:.2f}s")
                
            except Exception as e:
                logger.warning(f"⚠️  Reranking failed: {e}")
                final_results = stage1_results[:top_k]
        else:
            final_results = stage1_results[:top_k]
        
        # Add query analysis to results
        for result in final_results:
            result['parsed_query'] = parsed_query
            result['query_complexity'] = parsed_query.get('complexity_score', 0.0)
        
        total_time = time.time() - start_time
        logger.info(f"🎉 Search completed in {total_time:.2f}s")
        logger.info(f"📊 Results: {len(final_results)} items")
        
        return final_results
    
    def get_stats(self) -> Dict:
        """Get search engine statistics"""
        cache_stats = self.blip2_manager.get_cache_stats()
        
        return {
            'dataset_info': {
                'name': self.dataset_name,
                'total_images': len(self.image_paths),
                'index_type': 'BLIP-2 + FAISS'
            },
            'model_info': {
                'blip2_model': self.blip2_model_name,
                'tensorflow_rerank': self.enable_tensorflow_rerank,
                'embedding_dimension': self.blip2_manager.get_embedding_dimension()
            },
            'performance': self.search_stats,
            'cache': cache_stats
        }
    
    def clear_caches(self):
        """Clear all caches"""
        self.blip2_manager.clear_cache()
        if self.tf_manager:
            self.tf_manager.clear_cache()
        logger.info("🧹 All caches cleared")
    
    def switch_dataset(self, new_dataset: str) -> bool:
        """Switch to different dataset"""
        if new_dataset == self.dataset_name:
            return True
        
        logger.info(f"🔄 Switching dataset from {self.dataset_name} to {new_dataset}")
        
        self.dataset_name = new_dataset
        self.dataset_path = Path("datasets") / new_dataset
        
        # Reset index
        self.faiss_index = None
        self.image_paths = []
        self.metadata = {}
        
        # Load new index
        return self.load_or_build_index()
    
    def rebuild_index(self) -> bool:
        """Force rebuild index"""
        logger.info("🔨 Force rebuilding index...")
        return self.load_or_build_index(force_rebuild=True)


# Backwards compatibility wrapper
class AISearchEngine(EnhancedBLIP2SearchEngine):
    """Backwards compatibility wrapper for existing code"""
    pass


# Export classes
__all__ = ['EnhancedBLIP2SearchEngine', 'AISearchEngine']
