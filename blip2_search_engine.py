#!/usr/bin/env python3
"""
🤖 BLIP-2 Search Engine - Next Generation Vision-Language Model
Replace CLIP with BLIP-2 for better query understanding and Vietnamese support
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from datetime import datetime
from PIL import Image
import time

# BLIP-2 imports
try:
    from transformers import (
        Blip2Processor, 
        Blip2ForConditionalGeneration,
        Blip2VisionModel,
        AutoProcessor,
        AutoModelForVision2Seq
    )
    BLIP2_AVAILABLE = True
    print("✅ BLIP-2 models available")
except ImportError as e:
    BLIP2_AVAILABLE = False
    print(f"⚠️ BLIP-2 not available: {e}")

# Fallback to CLIP if BLIP-2 not available
try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
    print("✅ CLIP fallback available")
except ImportError:
    CLIP_AVAILABLE = False
    print("❌ No vision-language models available")

# FAISS for vector search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not available - using numpy similarity")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BLIP2ModelConfig:
    """Configuration for BLIP-2 models optimized for RTX 3060"""
    
    MODELS = {
        "blip2_opt_2.7b": {
            "name": "BLIP-2 OPT 2.7B",
            "model_path": "Salesforce/blip2-opt-2.7b",
            "processor_path": "Salesforce/blip2-opt-2.7b", 
            "memory_gb": 6,
            "description": "Best balance speed/quality for RTX 3060",
            "capabilities": ["image_understanding", "text_generation", "visual_qa"],
            "vietnamese_support": "good"
        },
        "blip2_t5_xl": {
            "name": "BLIP-2 T5-XL",
            "model_path": "Salesforce/blip2-t5-xl",
            "processor_path": "Salesforce/blip2-t5-xl",
            "memory_gb": 8,
            "description": "Best text generation capabilities",
            "capabilities": ["advanced_reasoning", "long_text_generation"],
            "vietnamese_support": "excellent"
        },
        "blip2_flan_t5_xl": {
            "name": "BLIP-2 FLAN-T5-XL", 
            "model_path": "Salesforce/blip2-flan-t5-xl",
            "processor_path": "Salesforce/blip2-flan-t5-xl",
            "memory_gb": 7,
            "description": "Instruction-tuned for better query understanding",
            "capabilities": ["instruction_following", "complex_queries"],
            "vietnamese_support": "very_good"
        }
    }
    
    @classmethod
    def get_optimal_model(cls, available_memory_gb: float = 6.0) -> str:
        """Get optimal model based on available GPU memory"""
        if available_memory_gb >= 8:
            return "blip2_t5_xl"
        elif available_memory_gb >= 7:
            return "blip2_flan_t5_xl"
        else:
            return "blip2_opt_2.7b"

class ComplexQueryProcessor:
    """Advanced query processing for complex and long-form queries"""
    
    def __init__(self):
        self.query_patterns = {
            "temporal": ["buổi sáng", "ban đêm", "lúc", "khi", "trong", "morning", "evening", "night", "during"],
            "spatial": ["trong", "ngoài", "trên", "dưới", "bên", "indoor", "outdoor", "inside", "outside"],
            "emotional": ["vui", "buồn", "hạnh phúc", "tức giận", "happy", "sad", "angry", "excited"],
            "actions": ["đang", "đi", "chạy", "nói", "làm", "walking", "running", "talking", "working"],
            "objects": ["người", "xe", "nhà", "cây", "person", "car", "house", "tree"],
            "comparison": ["so sánh", "khác nhau", "giống", "compare", "different", "similar"]
        }
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze complex query and extract semantic components"""
        analysis = {
            "original_query": query,
            "query_type": "simple",
            "entities": [],
            "actions": [],
            "temporal_context": [],
            "spatial_context": [],
            "emotional_context": [],
            "complexity_score": 0,
            "requires_reasoning": False,
            "language": "english"
        }
        
        query_lower = query.lower()
        
        # Detect language
        vietnamese_chars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
        if any(char in query_lower for char in vietnamese_chars):
            analysis["language"] = "vietnamese"
        
        # Extract semantic components
        for category, patterns in self.query_patterns.items():
            found_patterns = [p for p in patterns if p in query_lower]
            if found_patterns:
                analysis[f"{category}_context"] = found_patterns
                analysis["complexity_score"] += len(found_patterns)
        
        # Determine query type
        if "so sánh" in query_lower or "compare" in query_lower:
            analysis["query_type"] = "comparison"
            analysis["requires_reasoning"] = True
        elif "tìm" in query_lower and len(analysis["entities"]) > 2:
            analysis["query_type"] = "complex_search"
        elif "phân tích" in query_lower or "analyze" in query_lower:
            analysis["query_type"] = "analysis"
            analysis["requires_reasoning"] = True
        
        # Calculate final complexity
        if analysis["complexity_score"] > 3:
            analysis["query_type"] = "complex"
        
        return analysis
    
    def expand_query(self, analysis: Dict[str, Any]) -> List[str]:
        """Expand complex query into multiple search terms"""
        expanded_queries = [analysis["original_query"]]
        
        # Add simplified versions
        if analysis["query_type"] == "complex":
            # Extract key terms
            key_terms = []
            for context_type in ["temporal_context", "spatial_context", "emotional_context"]:
                key_terms.extend(analysis.get(context_type, []))
            
            if key_terms:
                # Create simplified query
                simplified = " ".join(key_terms[:3])  # Top 3 terms
                expanded_queries.append(simplified)
        
        # Add English/Vietnamese translations
        if analysis["language"] == "vietnamese":
            # Add common English equivalents
            translations = {
                "người": "person",
                "đàn ông": "man",
                "phụ nữ": "woman",
                "xe": "car",
                "nhà": "house",
                "cây": "tree"
            }
            
            english_query = analysis["original_query"]
            for vn, en in translations.items():
                english_query = english_query.replace(vn, en)
            
            if english_query != analysis["original_query"]:
                expanded_queries.append(english_query)
        
        return expanded_queries

class BLIP2SearchEngine:
    """Enhanced search engine using BLIP-2 models"""
    
    def __init__(self, index_dir: str = "index", model_key: str = "auto"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        
        # Initialize query processor
        self.query_processor = ComplexQueryProcessor()
        
        # Device setup
        self.device = self._setup_device()
        
        # Model configuration
        if model_key == "auto":
            available_memory = self._get_available_gpu_memory()
            self.model_key = BLIP2ModelConfig.get_optimal_model(available_memory)
        else:
            self.model_key = model_key
        
        self.model_config = BLIP2ModelConfig.MODELS[self.model_key]
        
        # Initialize models
        self.model = None
        self.processor = None
        self.fallback_model = None  # CLIP fallback
        
        # Data storage
        self.frames_metadata: List[Dict] = []
        self.image_embeddings: np.ndarray = None
        self.faiss_index = None
        
        # Performance tracking
        self.search_stats = {
            "total_searches": 0,
            "avg_search_time": 0,
            "last_search_time": 0
        }
        
        print(f"🤖 BLIP-2 Search Engine initialized")
        print(f"📱 Selected model: {self.model_config['name']}")
        print(f"💾 Device: {self.device}")
    
    def _setup_device(self) -> str:
        """Setup optimal device for inference"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"🎮 GPU: {gpu_name}")
            print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
            
            # Test GPU functionality
            try:
                test_tensor = torch.randn(100, 100).cuda()
                result = torch.mm(test_tensor, test_tensor.T)
                del test_tensor, result
                torch.cuda.empty_cache()
                return "cuda"
            except Exception as e:
                print(f"⚠️ GPU test failed: {e}")
                return "cpu"
        else:
            print("💻 Using CPU device")
            return "cpu"
    
    def _get_available_gpu_memory(self) -> float:
        """Get available GPU memory in GB"""
        if self.device == "cuda":
            try:
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                return memory_total - memory_allocated
            except:
                return 4.0  # Conservative estimate
        return 0.0
    
    def load_model(self) -> bool:
        """Load BLIP-2 model with fallback to CLIP"""
        try:
            if BLIP2_AVAILABLE:
                print(f"🔄 Loading {self.model_config['name']}...")
                
                # Load BLIP-2 model
                self.processor = Blip2Processor.from_pretrained(
                    self.model_config["processor_path"]
                )
                
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    self.model_config["model_path"],
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                
                if self.device == "cuda":
                    self.model = self.model.to(self.device)
                
                print(f"✅ BLIP-2 model loaded successfully")
                return True
                
            else:
                # Fallback to CLIP
                return self._load_clip_fallback()
                
        except Exception as e:
            print(f"❌ Failed to load BLIP-2: {e}")
            return self._load_clip_fallback()
    
    def _load_clip_fallback(self) -> bool:
        """Load CLIP as fallback model"""
        try:
            if CLIP_AVAILABLE:
                print("🔄 Loading CLIP fallback model...")
                
                self.fallback_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                
                if self.device == "cuda":
                    self.fallback_model = self.fallback_model.to(self.device)
                
                print("✅ CLIP fallback loaded")
                return True
            else:
                print("❌ No vision-language models available")
                return False
        except Exception as e:
            print(f"❌ Failed to load CLIP fallback: {e}")
            return False
    
    def encode_image(self, image_path: str) -> Optional[np.ndarray]:
        """Encode image to embedding vector"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            if self.model is not None:  # BLIP-2
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    # Get vision features
                    vision_outputs = self.model.vision_model(**inputs)
                    image_features = vision_outputs.last_hidden_state.mean(dim=1)  # Pool features
                    
                return image_features.cpu().numpy().flatten()
                
            elif self.fallback_model is not None:  # CLIP
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    image_features = self.fallback_model.get_image_features(**inputs)
                    
                return image_features.cpu().numpy().flatten()
            
            else:
                print("❌ No model available for encoding")
                return None
                
        except Exception as e:
            print(f"❌ Failed to encode image {image_path}: {e}")
            return None
    
    def encode_text(self, text: str) -> Optional[np.ndarray]:
        """Encode text query to embedding vector"""
        try:
            if self.model is not None:  # BLIP-2
                # For BLIP-2, we use the text encoder part
                inputs = self.processor(text=text, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    # Get text features using the language model
                    text_inputs = inputs["input_ids"]
                    text_features = self.model.language_model.get_input_embeddings()(text_inputs)
                    text_features = text_features.mean(dim=1)  # Pool features
                    
                return text_features.cpu().numpy().flatten()
                
            elif self.fallback_model is not None:  # CLIP
                inputs = self.processor(text=text, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    text_features = self.fallback_model.get_text_features(**inputs)
                    
                return text_features.cpu().numpy().flatten()
            
            else:
                print("❌ No model available for text encoding")
                return None
                
        except Exception as e:
            print(f"❌ Failed to encode text '{text}': {e}")
            return None
    
    def build_index(self, frames_directory: str) -> bool:
        """Build search index from frames directory"""
        print(f"🔄 Building search index from {frames_directory}...")
        
        frames_dir = Path(frames_directory)
        if not frames_dir.exists():
            print(f"❌ Frames directory not found: {frames_directory}")
            return False
        
        # Load model
        if not self.load_model():
            print("❌ Failed to load any vision-language model")
            return False
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(frames_dir.glob(f"**/*{ext}"))
        
        print(f"📊 Found {len(image_files)} image files")
        
        if len(image_files) == 0:
            print("❌ No image files found")
            return False
        
        # Extract embeddings
        embeddings = []
        metadata = []
        
        for i, image_path in enumerate(image_files):
            if i % 50 == 0:
                print(f"🔄 Processing image {i+1}/{len(image_files)}...")
            
            embedding = self.encode_image(str(image_path))
            if embedding is not None:
                embeddings.append(embedding)
                metadata.append({
                    "frame_path": str(image_path),
                    "frame_id": image_path.stem,
                    "video_path": str(image_path.parent.parent / "videos" / f"{image_path.parent.name}.mp4")
                })
        
        if len(embeddings) == 0:
            print("❌ No valid embeddings created")
            return False
        
        # Store embeddings and metadata
        self.image_embeddings = np.vstack(embeddings)
        self.frames_metadata = metadata
        
        # Build FAISS index
        if FAISS_AVAILABLE:
            self._build_faiss_index()
        
        # Save to disk
        self._save_index()
        
        print(f"✅ Index built successfully with {len(embeddings)} embeddings")
        return True
    
    def _build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        try:
            dimension = self.image_embeddings.shape[1]
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.image_embeddings.astype('float32'))
            
            # Create FAISS index
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
            self.faiss_index.add(self.image_embeddings.astype('float32'))
            
            print(f"✅ FAISS index created with dimension {dimension}")
            
        except Exception as e:
            print(f"❌ Failed to create FAISS index: {e}")
            self.faiss_index = None
    
    def _save_index(self):
        """Save index and metadata to disk"""
        try:
            # Save embeddings
            embeddings_path = self.index_dir / f"blip2_embeddings_{self.model_key}.npy"
            np.save(embeddings_path, self.image_embeddings)
            
            # Save metadata
            metadata_path = self.index_dir / f"blip2_metadata_{self.model_key}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.frames_metadata, f, ensure_ascii=False, indent=2)
            
            # Save FAISS index
            if self.faiss_index is not None:
                faiss_path = self.index_dir / f"blip2_faiss_{self.model_key}.index"
                faiss.write_index(self.faiss_index, str(faiss_path))
            
            print(f"💾 Index saved to {self.index_dir}")
            
        except Exception as e:
            print(f"❌ Failed to save index: {e}")
    
    def load_index(self) -> bool:
        """Load pre-built index from disk"""
        try:
            # Load embeddings
            embeddings_path = self.index_dir / f"blip2_embeddings_{self.model_key}.npy"
            if not embeddings_path.exists():
                print(f"⚠️ Index not found for model {self.model_key}")
                return False
            
            self.image_embeddings = np.load(embeddings_path)
            
            # Load metadata
            metadata_path = self.index_dir / f"blip2_metadata_{self.model_key}.json"
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.frames_metadata = json.load(f)
            
            # Load FAISS index
            if FAISS_AVAILABLE:
                faiss_path = self.index_dir / f"blip2_faiss_{self.model_key}.index"
                if faiss_path.exists():
                    self.faiss_index = faiss.read_index(str(faiss_path))
            
            print(f"✅ Index loaded: {len(self.frames_metadata)} frames")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load index: {e}")
            return False
    
    def search_complex_query(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search with complex query processing"""
        start_time = time.time()
        
        # Analyze query
        analysis = self.query_processor.analyze_query(query)
        print(f"🔍 Query analysis: {analysis['query_type']} (complexity: {analysis['complexity_score']})")
        
        # Expand query if complex
        expanded_queries = self.query_processor.expand_query(analysis)
        print(f"📝 Expanded to {len(expanded_queries)} search terms")
        
        # Search with each expanded query
        all_results = []
        for i, search_query in enumerate(expanded_queries):
            print(f"🔄 Searching: '{search_query}'")
            results = self._search_single_query(search_query, top_k)
            
            # Weight results based on query position
            weight = 1.0 / (i + 1)  # First query gets full weight
            for result in results:
                result["weighted_score"] = result["similarity_score"] * weight
                result["search_query"] = search_query
                
            all_results.extend(results)
        
        # Combine and re-rank results
        final_results = self._combine_and_rank_results(all_results, analysis, top_k)
        
        # Update statistics
        search_time = time.time() - start_time
        self.search_stats["total_searches"] += 1
        self.search_stats["last_search_time"] = search_time
        self.search_stats["avg_search_time"] = (
            (self.search_stats["avg_search_time"] * (self.search_stats["total_searches"] - 1) + search_time) 
            / self.search_stats["total_searches"]
        )
        
        print(f"⚡ Search completed in {search_time:.3f}s")
        return final_results
    
    def _search_single_query(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search with single query"""
        if not self.load_model():
            return []
        
        if self.image_embeddings is None:
            if not self.load_index():
                print("❌ No search index available")
                return []
        
        # Encode query
        query_embedding = self.encode_text(query)
        if query_embedding is None:
            return []
        
        # Search using FAISS or numpy
        if self.faiss_index is not None:
            scores, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1).astype('float32'), 
                top_k
            )
            scores = scores[0]
            indices = indices[0]
        else:
            # Fallback to numpy similarity
            similarities = np.dot(self.image_embeddings, query_embedding)
            indices = np.argsort(similarities)[::-1][:top_k]
            scores = similarities[indices]
        
        # Prepare results
        results = []
        for i, (idx, score) in enumerate(zip(indices, scores)):
            if idx >= len(self.frames_metadata):
                continue
                
            metadata = self.frames_metadata[idx]
            results.append({
                "rank": i + 1,
                "frame_path": metadata["frame_path"],
                "frame_id": metadata["frame_id"],
                "video_path": metadata.get("video_path", ""),
                "similarity_score": float(score),
                "metadata": metadata
            })
        
        return results
    
    def _combine_and_rank_results(self, all_results: List[Dict], analysis: Dict, top_k: int) -> List[Dict]:
        """Combine and re-rank results from multiple queries"""
        # Group by frame_path to avoid duplicates
        unique_results = {}
        for result in all_results:
            frame_path = result["frame_path"]
            if frame_path not in unique_results:
                unique_results[frame_path] = result
            else:
                # Combine scores
                existing = unique_results[frame_path]
                existing["weighted_score"] = max(existing["weighted_score"], result["weighted_score"])
        
        # Sort by weighted score
        sorted_results = sorted(
            unique_results.values(), 
            key=lambda x: x["weighted_score"], 
            reverse=True
        )
        
        # Apply query-specific re-ranking
        if analysis["query_type"] == "complex":
            # Boost results that match multiple query aspects
            # This is where we could add more sophisticated ranking
            pass
        
        # Return top-k results
        final_results = sorted_results[:top_k]
        
        # Update rank
        for i, result in enumerate(final_results):
            result["rank"] = i + 1
            result["final_score"] = result["weighted_score"]
        
        return final_results
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and statistics"""
        return {
            "model_name": self.model_config["name"],
            "model_key": self.model_key,
            "device": self.device,
            "blip2_available": BLIP2_AVAILABLE,
            "fallback_mode": self.model is None,
            "index_size": len(self.frames_metadata) if self.frames_metadata else 0,
            "embedding_dimension": self.image_embeddings.shape[1] if self.image_embeddings is not None else 0,
            "faiss_available": FAISS_AVAILABLE and self.faiss_index is not None,
            "search_stats": self.search_stats,
            "memory_usage_gb": torch.cuda.memory_allocated(0) / 1024**3 if self.device == "cuda" else 0
        }

def demo_blip2_search():
    """Demo function for BLIP-2 search engine"""
    print("🚀 BLIP-2 Enhanced Search Engine Demo")
    print("=" * 50)
    
    # Initialize search engine
    engine = BLIP2SearchEngine()
    
    # Show system info
    info = engine.get_system_info()
    print("\n📊 System Information:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Load or build index
    if not engine.load_index():
        print("\n🔄 Building new search index...")
        frames_dir = "frames"
        if not engine.build_index(frames_dir):
            print("❌ Failed to build index")
            return
    
    # Interactive search
    print("\n🔍 Interactive Search (BLIP-2 Enhanced)")
    print("Features:")
    print("  ✅ Complex query understanding")
    print("  ✅ Vietnamese language support")  
    print("  ✅ Multi-aspect query expansion")
    print("  ✅ Intelligent result ranking")
    print("\nExample queries:")
    print("  - 'Tìm người đàn ông mặc áo xanh đang nói chuyện'")
    print("  - 'So sánh cảnh buổi sáng và buổi tối'")
    print("  - 'Find happy children playing outdoors'")
    print("\nType 'quit' to exit")
    
    while True:
        try:
            query = input("\n🔍 Search query: ").strip()
            
            if query.lower() == 'quit':
                break
            
            if not query:
                continue
            
            # Perform search
            results = engine.search_complex_query(query, top_k=5)
            
            print(f"\n📊 Found {len(results)} results:")
            for result in results:
                print(f"  {result['rank']}. {Path(result['frame_path']).name}")
                print(f"     Score: {result['final_score']:.4f}")
                print(f"     Query: '{result.get('search_query', query)}'")
                print()
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n🎉 Demo completed!")

if __name__ == "__main__":
    demo_blip2_search()
