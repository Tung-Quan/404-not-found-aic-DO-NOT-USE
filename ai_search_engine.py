"""
Enhanced AI Video Search Engine with Hybrid Model Support
Supports both PyTorch and TensorFlow models with real-time switching
"""

import json
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from PIL import Image

# Import enhanced hybrid model manager
from enhanced_hybrid_manager import EnhancedHybridModelManager as HybridModelManager
from enhanced_hybrid_manager import ModelType, ModelBackend, ModelConfig

# Check for optional dependencies
try:
    import torch
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    # Disable GPU logging
    tf.get_logger().setLevel('ERROR')
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedAIVideoSearchEngine:
    """Enhanced AI search engine với hybrid model support"""
    
    def __init__(self, index_dir: str = "index", model_manager: Optional[HybridModelManager] = None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        
        # Initialize model manager
        if model_manager is not None:
            self.model_manager = model_manager
            print("✅ Using provided model manager")
        else:
            print("🔄 Initializing new model manager...")
            self.model_manager = HybridModelManager()
            print("✅ Model manager initialized")
        
        # Active models tracking
        self.active_models: Dict[str, str] = {
            "vision_language": None,  # For image-text matching
            "text_embedding": None,   # For text embeddings
            "object_detection": None, # For object detection
            "video_classification": None  # For video understanding
        }
        
        # Data storage
        self.frames_metadata: List[Dict] = []
        self.image_embeddings: Dict[str, np.ndarray] = {}  # Per model
        self.faiss_indexes: Dict[str, Any] = {}  # Per model FAISS indexes
        
        # Performance tracking
        self.performance_log: List[Dict[str, Any]] = []
        
        # GPU optimization settings
        self.use_mixed_precision = True
        self.batch_size = 32  # For batch processing
        self.max_gpu_memory_gb = 4  # Reserve some GPU memory
        
        # Device selection with GPU optimization
        self.device = self._setup_optimal_device()
        
        # Initialize GPU memory management
        if PYTORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                # Use 80% of GPU memory, leaving 20% for system
                torch.cuda.set_per_process_memory_fraction(0.8)
        
        print(f"🚀 Enhanced AI Video Search Engine initialized")
        print(f"💾 Device: {self.device}")
        print(f"🔧 Backends available: PyTorch={PYTORCH_AVAILABLE}, TensorFlow={TENSORFLOW_AVAILABLE}")
        print(f"⚡ Mixed precision: {self.use_mixed_precision}")
        
        # Load metadata
        self.load_metadata()
    
    def _setup_optimal_device(self) -> str:
        """Setup optimal compute device with GPU priority"""
        if PYTORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    if gpu_count > 0:
                        # Get GPU info
                        gpu_name = torch.cuda.get_device_name(0)
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        
                        print(f"🎮 GPU detected: {gpu_name}")
                        print(f"💾 GPU memory: {gpu_memory:.1f} GB")
                        print(f"🔧 CUDA version: {torch.version.cuda}")
                        
                        # Use GPU if it has enough memory (RTX 3060 has 6GB)
                        if gpu_memory >= 2.0:  # Lower threshold for more compatibility
                            # Test CUDA functionality
                            try:
                                test_tensor = torch.randn(100, 100).cuda()
                                result = torch.mm(test_tensor, test_tensor.T)
                                test_tensor.cpu()
                                result.cpu()
                                del test_tensor, result
                                torch.cuda.empty_cache()
                                print("🚀 CUDA functionality verified - GPU ready!")
                                return "cuda"
                            except Exception as e:
                                print(f"⚠️ CUDA test failed: {e}")
                                print("🔄 Falling back to CPU")
                        else:
                            print(f"⚠️ GPU memory too low ({gpu_memory:.1f}GB < 2.0GB required)")
                    else:
                        print("⚠️ No GPU devices found")
                else:
                    print("⚠️ CUDA not available in PyTorch")
            except Exception as e:
                print(f"❌ GPU initialization error: {e}")
        else:
            print("❌ PyTorch not available")
        
        print("💻 Using CPU device")
        return "cpu"
    
    def _log_performance(self, model_key: str, operation: str, duration: float):
        """Log performance data"""
        self.performance_log.append({
            "timestamp": time.time(),
            "model": model_key,
            "operation": operation,
            "duration": duration
        })
    
    def get_active_models(self) -> Dict[str, str]:
        """Get currently active models"""
        return {k: v for k, v in self.active_models.items() if v is not None}
    
    def set_active_model(self, task: str, model_key: str) -> bool:
        """Set active model for specific task"""
        if task not in self.active_models:
            print(f"❌ Unknown task: {task}")
            return False
        
        # Load model if not already loaded
        if not self.model_manager.load_model(model_key):
            print(f"❌ Failed to load model: {model_key}")
            return False
        
        # Verify model type compatibility
        config = self.model_manager.available_models[model_key]
        
        # Check capabilities if available
        if hasattr(config, 'capabilities'):
            if task == "vision_language" and "vision_language" not in config.capabilities:
                print(f"❌ Model {model_key} is not a vision-language model")
                return False
            elif task == "text_embedding" and "text_embedding" not in config.capabilities:
                print(f"❌ Model {model_key} is not a text embedding model")
                return False
        else:
            # Fallback to old method
            if task == "vision_language" and config.model_type != ModelType.VISION_LANGUAGE:
                print(f"❌ Model {model_key} is not a vision-language model")
                return False
            elif task == "text_embedding" and config.model_type != ModelType.TEXT_EMBEDDING:
                print(f"❌ Model {model_key} is not a text embedding model")
                return False
        
        # Set as active
        old_model = self.active_models[task]
        self.active_models[task] = model_key
        
        print(f"✅ Set {task} model to: {config.name}")
        
        # Optionally unload old model to save memory
        if old_model and old_model != model_key:
            self.model_manager.unload_model(old_model)
        
        return True
    
    def encode_image_with_model(self, image_path: Path, model_key: str) -> Optional[np.ndarray]:
        """Encode image using specific model with GPU optimization"""
        if model_key not in self.model_manager.loaded_models:
            print(f"❌ Model {model_key} not loaded")
            return None
        
        model_info = self.model_manager.loaded_models[model_key]
        config = model_info  # ModelConfig object itself
        
        try:
            image = Image.open(image_path).convert("RGB")
            
            start_time = time.time()
            
            if config.backend == ModelBackend.PYTORCH:
                # PyTorch model encoding with GPU optimization
                if "clip" in model_key.lower():
                    # CLIP encoding
                    inputs = model_info.processor(images=image, return_tensors="pt")
                    
                    # Move to GPU efficiently
                    if self.device == "cuda":
                        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
                    
                    # Use mixed precision if available and enabled
                    if self.use_mixed_precision and self.device == "cuda":
                        with torch.cuda.amp.autocast():
                            with torch.no_grad():
                                image_features = model_info.model.get_image_features(**inputs)
                                image_features = F.normalize(image_features, p=2, dim=1)
                    else:
                        with torch.no_grad():
                            image_features = model_info.model.get_image_features(**inputs)
                            image_features = F.normalize(image_features, p=2, dim=1)
                    
                    embedding = image_features.cpu().numpy().flatten()
                
                elif "blip" in model_key.lower():
                    # BLIP encoding (image features)
                    inputs = model_info.processor(image, return_tensors="pt")
                    
                    # Move to GPU efficiently
                    if self.device == "cuda":
                        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
                    
                    # Use mixed precision if available and enabled
                    if self.use_mixed_precision and self.device == "cuda":
                        with torch.cuda.amp.autocast():
                            with torch.no_grad():
                                vision_outputs = model_info.model.vision_model(**inputs)
                                image_features = vision_outputs.last_hidden_state.mean(dim=1)
                    else:
                        with torch.no_grad():
                            vision_outputs = model_info.model.vision_model(**inputs)
                            image_features = vision_outputs.last_hidden_state.mean(dim=1)
                    
                    embedding = image_features.cpu().numpy().flatten()
                
                else:
                    print(f"❌ Unknown PyTorch model type: {model_key}")
                    return None
            
            elif config.backend == ModelBackend.TENSORFLOW:
                # TensorFlow model encoding
                image_array = np.array(image)
                image_array = np.expand_dims(image_array, axis=0)
                
                if "clip" in model_key.lower():
                    # TensorFlow CLIP equivalent
                    # Preprocess image to expected format
                    image_resized = tf.image.resize(image_array, [224, 224])
                    image_normalized = tf.cast(image_resized, tf.float32) / 255.0
                    
                    # Get embedding
                    embedding = model_info.model(image_normalized)
                    embedding = embedding.numpy().flatten()
                
                elif "movinet" in model_key.lower():
                    # MoViNet requires video input, but we can use single frame
                    # Repeat frame to create video-like input
                    video_input = tf.repeat(tf.expand_dims(image_array, axis=1), 8, axis=1)
                    video_input = tf.cast(video_input, tf.float32) / 255.0
                    
                    embedding = model_info.model(video_input)
                    embedding = embedding.numpy().flatten()
                
                else:
                    # Generic TensorFlow model
                    image_preprocessed = tf.cast(image_array, tf.float32) / 255.0
                    embedding = model_info.model(image_preprocessed)
                    embedding = embedding.numpy().flatten()
            
            else:
                print(f"❌ Unknown backend: {config.backend}")
                return None
            
            # Log performance
            duration = time.time() - start_time
            self._log_performance(model_key, "image_encoding", duration)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error encoding image with {model_key}: {e}")
            return None
    
    def encode_images_batch(self, image_paths: List[Path], model_key: str) -> List[Optional[np.ndarray]]:
        """Encode multiple images in batch for GPU efficiency"""
        if model_key not in self.model_manager.loaded_models:
            print(f"❌ Model {model_key} not loaded")
            return [None] * len(image_paths)
        
        model_info = self.model_manager.loaded_models[model_key]
        config = model_info  # ModelConfig object itself
        
        if config.backend != ModelBackend.PYTORCH:
            # Fall back to individual processing for non-PyTorch models
            return [self.encode_image_with_model(path, model_key) for path in image_paths]
        
        results = []
        batch_size = min(self.batch_size, len(image_paths))
        
        try:
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                batch_images = []
                valid_indices = []
                
                # Load and validate images
                for j, path in enumerate(batch_paths):
                    try:
                        image = Image.open(path).convert("RGB")
                        batch_images.append(image)
                        valid_indices.append(i + j)
                    except Exception as e:
                        logger.error(f"Error loading image {path}: {e}")
                        continue
                
                if not batch_images:
                    results.extend([None] * len(batch_paths))
                    continue
                
                start_time = time.time()
                
                if "clip" in model_key.lower():
                    # CLIP batch encoding
                    inputs = model_info.processor(images=batch_images, return_tensors="pt", padding=True)
                    
                    if self.device == "cuda":
                        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
                    
                    if self.use_mixed_precision and self.device == "cuda":
                        with torch.cuda.amp.autocast():
                            with torch.no_grad():
                                image_features = model_info.model.get_image_features(**inputs)
                                image_features = F.normalize(image_features, p=2, dim=1)
                    else:
                        with torch.no_grad():
                            image_features = model_info.model.get_image_features(**inputs)
                            image_features = F.normalize(image_features, p=2, dim=1)
                    
                    # Convert to numpy
                    batch_embeddings = image_features.cpu().numpy()
                    
                    # Add to results
                    batch_results = [None] * len(batch_paths)
                    for k, embedding in enumerate(batch_embeddings):
                        if k < len(valid_indices):
                            idx = valid_indices[k] - i
                            batch_results[idx] = embedding
                    
                    results.extend(batch_results)
                
                else:
                    # Fall back to individual processing for other models
                    for path in batch_paths:
                        embedding = self.encode_image_with_model(path, model_key)
                        results.append(embedding)
                
                # Log batch performance
                duration = time.time() - start_time
                if batch_images:
                    avg_time = duration / len(batch_images)
                    self._log_performance(model_key, "batch_image_encoding", avg_time)
                
                # Clear GPU cache periodically
                if self.device == "cuda" and i % (batch_size * 4) == 0:
                    torch.cuda.empty_cache()
        
        except Exception as e:
            logger.error(f"Error in batch encoding: {e}")
            # Fall back to individual processing
            return [self.encode_image_with_model(path, model_key) for path in image_paths]
        
        return results
    
    def encode_text_with_model(self, text: str, model_key: str) -> Optional[np.ndarray]:
        """Encode text using specific model"""
        if model_key not in self.model_manager.loaded_models:
            print(f"❌ Model {model_key} not loaded")
            return None
        
        model_info = self.model_manager.loaded_models[model_key]
        config = model_info  # ModelConfig object itself
        
        try:
            start_time = time.time()
            
            if config.backend == ModelBackend.PYTORCH:
                # PyTorch model encoding
                if "clip" in model_key.lower():
                    # CLIP text encoding
                    inputs = model_info.processor(text=[text], return_tensors="pt", padding=True)
                    if PYTORCH_AVAILABLE and torch.cuda.is_available():
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        text_features = model_info.model.get_text_features(**inputs)
                        text_features = F.normalize(text_features, p=2, dim=1)
                    
                    embedding = text_features.cpu().numpy().flatten()
                
                elif "sentence" in model_key.lower():
                    # Sentence Transformers
                    if hasattr(model_info.model, 'encode'):
                        embedding = model_info.model.encode([text])[0]
                    else:
                        print(f"❌ Model {model_key} doesn't support text encoding")
                        return None
                
                else:
                    print(f"❌ Unknown PyTorch text model: {model_key}")
                    return None
            
            elif config.backend == ModelBackend.TENSORFLOW:
                # TensorFlow model encoding
                if "universal_sentence" in model_key.lower():
                    # Universal Sentence Encoder
                    embedding = model_info.model([text]).numpy()[0]
                
                elif "clip" in model_key.lower():
                    # TensorFlow CLIP text encoding
                    # This would need TensorFlow CLIP implementation
                    print(f"⚠️ TensorFlow CLIP text encoding not implemented")
                    return None
                
                else:
                    # Generic text model
                    embedding = model_info.model([text]).numpy()[0]
            
            else:
                print(f"❌ Unknown backend: {config.backend}")
                return None
            
            # Log performance
            duration = time.time() - start_time
            self._log_performance(model_key, "text_encoding", duration)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error encoding text with {model_key}: {e}")
            return None
    
    def load_metadata(self):
        """Load frames metadata, or auto-generate from all .jpg files in frames/ if missing/incomplete"""
        metadata_file = self.index_dir / "metadata.json"
        frames_dir = Path("frames")
        all_frame_files = []
        for root, dirs, files in os.walk(frames_dir):
            for file in files:
                if file.lower().endswith('.jpg'):
                    all_frame_files.append(str(Path(root) / file))
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                loaded_meta = json.load(f)
            # Only keep frames that exist on disk
            self.frames_metadata = [m for m in loaded_meta if Path(m['frame_path']).exists()]
            # Add any missing frames from disk
            existing_paths = set(m['frame_path'] for m in self.frames_metadata)
            for frame_path in all_frame_files:
                if frame_path not in existing_paths:
                    self.frames_metadata.append({'frame_path': frame_path})
            print(f"📊 Loaded {len(self.frames_metadata)} frame records (from metadata + disk)")
        else:
            # No metadata, generate from disk
            self.frames_metadata = [{'frame_path': frame_path} for frame_path in all_frame_files]
            print(f"📊 Auto-generated {len(self.frames_metadata)} frame records from disk")
    
    def initialize_default_models(self) -> bool:
        """Initialize default models for basic functionality"""
        if not self.model_manager:
            return False
        
        print("🔄 Initializing default models...")
        
        # Get recommendations for different use cases
        fast_models = self.model_manager.recommend_models("fast_search")
        
        success = False
        
        # Try to load vision-language model
        for task, model_key in fast_models.items():
            if task == "vision_language":
                if self.set_active_model("vision_language", model_key):
                    success = True
                    break
        
        # Try to load text embedding model
        for task, model_key in fast_models.items():
            if task == "text_embedding":
                if self.set_active_model("text_embedding", model_key):
                    success = True
                    break
        
        return success
    
    def build_embeddings_index(self, 
                              model_key: str = None, 
                              force_rebuild: bool = False) -> bool:
        """Build embeddings index với model cụ thể"""
        if not model_key:
            model_key = self.active_models.get("vision_language")
        
        if not model_key:
            print("❌ No active vision-language model")
            return False
        
        embeddings_dir = self.index_dir / "embeddings"
        embeddings_file = embeddings_dir / f"image_embeddings_{model_key}.npy"
        
        embeddings_dir.mkdir(exist_ok=True)
        
        # Check if already exists
        if not force_rebuild and embeddings_file.exists():
            print(f"📁 Loading existing embeddings for {model_key}...")
            try:
                self.image_embeddings[model_key] = np.load(embeddings_file)
                print(f"✅ Loaded {len(self.image_embeddings[model_key])} embeddings")
                return True
            except Exception as e:
                print(f"⚠️ Error loading embeddings: {e}")
        
        print(f"🔄 Building image embeddings with {model_key}...")
        print(f"🚀 Using batch processing (batch_size={self.batch_size}) for GPU optimization")
        
        # Collect all valid frame paths
        valid_frames = []
        valid_paths = []
        
        for frame_data in self.frames_metadata:
            frame_path = Path(frame_data['frame_path'])
            if frame_path.exists():
                valid_frames.append(frame_data)
                valid_paths.append(frame_path)
        
        if not valid_paths:
            print("❌ No valid frame paths found")
            return False
        
        print(f"📊 Processing {len(valid_paths)} valid frames...")
        
        # Process in batches for GPU efficiency
        all_embeddings = []
        batch_size = self.batch_size
        
        for i in range(0, len(valid_paths), batch_size):
            batch_paths = valid_paths[i:i + batch_size]
            progress = f"{i + len(batch_paths)}/{len(valid_paths)}"
            print(f"🔄 Processing batch {progress} frames...")
            
            # Use batch encoding for better GPU utilization
            batch_embeddings = self.encode_images_batch(batch_paths, model_key)
            
            # Filter successful embeddings
            for j, embedding in enumerate(batch_embeddings):
                if embedding is not None:
                    all_embeddings.append(embedding)
                else:
                    # Remove invalid frame from valid_frames
                    idx = i + j
                    if idx < len(valid_frames):
                        valid_frames[idx] = None
            
            # Clear GPU cache every few batches
            if self.device == "cuda" and i % (batch_size * 4) == 0:
                torch.cuda.empty_cache()
        
        # Filter out None frames
        successful_frames = [f for f in valid_frames if f is not None]
        
        if all_embeddings and len(all_embeddings) == len(successful_frames):
            self.image_embeddings[model_key] = np.array(all_embeddings)
            self.frames_metadata = successful_frames
            
            # Save embeddings
            np.save(embeddings_file, self.image_embeddings[model_key])
            print(f"✅ Built and saved {len(all_embeddings)} embeddings for {model_key}")
            print(f"⚡ GPU batch processing completed successfully")
            
            # Build FAISS index
            if FAISS_AVAILABLE:
                self.build_faiss_index(model_key)
            
            # Final GPU cleanup
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return True
        
        print("❌ No embeddings generated")
        return False
    
    def build_faiss_index(self, model_key: str):
        """Build FAISS index cho model cụ thể"""
        if not FAISS_AVAILABLE or model_key not in self.image_embeddings:
            return False
        
        try:
            print(f"🔄 Building FAISS index for {model_key}...")
            embeddings = self.image_embeddings[model_key]
            dimension = embeddings.shape[1]
            
            # Use IndexFlatIP for cosine similarity
            index = faiss.IndexFlatIP(dimension)
            
            # Normalize embeddings for cosine similarity
            embeddings_normalized = embeddings.copy()
            faiss.normalize_L2(embeddings_normalized)
            
            index.add(embeddings_normalized)
            self.faiss_indexes[model_key] = index
            
            # Save index
            faiss_dir = self.index_dir / "faiss"
            faiss_dir.mkdir(exist_ok=True)
            faiss.write_index(index, str(faiss_dir / f"image_index_{model_key}.faiss"))
            
            print(f"✅ FAISS index for {model_key} built with {index.ntotal} vectors")
            return True
            
        except Exception as e:
            print(f"❌ Error building FAISS index: {e}")
            return False
    
    def search_similar_frames(self, 
                            query: str, 
                            top_k: int = 5,
                            model_key: str = None,
                            use_faiss: bool = True) -> List[Dict[str, Any]]:
        """Search for frames similar to text query với model choice"""
        if not model_key:
            # Try vision-language model first, then text embedding
            model_key = (self.active_models.get("vision_language") or 
                        self.active_models.get("text_embedding"))
        
        if not model_key or model_key not in self.image_embeddings:
            print(f"❌ No embeddings available for model: {model_key}")
            return []
        
        # Encode query
        query_embedding = self.encode_text_with_model(query, model_key)
        if query_embedding is None:
            print(f"❌ Failed to encode query with {model_key}")
            return []
        
        start_time = time.time()
        
        if use_faiss and model_key in self.faiss_indexes:
            # Use FAISS for fast search
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            distances, indices = self.faiss_indexes[model_key].search(query_embedding, top_k)
            
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(self.frames_metadata):
                    frame_data = self.frames_metadata[idx].copy()
                    # Convert distance to similarity score (0=identical, higher=less similar)
                    # Use 1/(1+distance) to convert distance to similarity (0-1 range)
                    similarity = 1.0 / (1.0 + float(distance))
                    frame_data['similarity_score'] = similarity
                    frame_data['distance'] = float(distance)
                    frame_data['score'] = similarity  # For API compatibility
                    frame_data['model_used'] = model_key
                    results.append(frame_data)
        else:
            # Fallback to numpy similarity
            similarities = np.dot(self.image_embeddings[model_key], query_embedding)
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                frame_data = self.frames_metadata[idx].copy()
                similarity = float(similarities[idx])
                frame_data['similarity_score'] = similarity
                frame_data['score'] = similarity  # For API compatibility
                frame_data['model_used'] = model_key
                results.append(frame_data)
        
        # Log performance
        search_time = time.time() - start_time
        self._log_performance(model_key, "search", search_time)
        
        return results
    
    def compare_models_performance(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """So sánh performance của các models khác nhau"""
        results = {}
        
        # Test all loaded vision-language and text embedding models
        test_models = []
        for model_key in self.model_manager.loaded_models:
            config = self.model_manager.loaded_models[model_key]
            if config.config.model_type in [ModelType.VISION_LANGUAGE, ModelType.TEXT_EMBEDDING]:
                if model_key in self.image_embeddings:
                    test_models.append(model_key)
        
        for model_key in test_models:
            start_time = time.time()
            search_results = self.search_similar_frames(query, top_k, model_key)
            search_time = time.time() - start_time
            
            config = self.model_manager.loaded_models[model_key]
            results[model_key] = {
                "model_name": config.config.name,
                "backend": config.config.backend.value,
                "search_time": search_time,
                "results_count": len(search_results),
                "results": search_results,
                "avg_score": np.mean([r.get('similarity_score', 0) for r in search_results]) if search_results else 0
            }
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Lấy thống kê performance"""
        if not self.performance_log:
            return {}
        
        stats = {}
        for log_entry in self.performance_log:
            model = log_entry['model']
            operation = log_entry['operation']
            
            if model not in stats:
                stats[model] = {}
            if operation not in stats[model]:
                stats[model][operation] = []
            
            stats[model][operation].append(log_entry['duration'])
        
        # Calculate averages
        summary = {}
        for model, operations in stats.items():
            summary[model] = {}
            for operation, durations in operations.items():
                summary[model][operation] = {
                    "avg_time": np.mean(durations),
                    "min_time": np.min(durations),
                    "max_time": np.max(durations),
                    "count": len(durations)
                }
        
        return summary
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            "frames_count": len(self.frames_metadata),
            "active_models": self.get_active_models(),
            "available_models": len(self.model_manager.available_models) if self.model_manager else 0,
            "loaded_models": len(self.model_manager.loaded_models) if self.model_manager else 0,
            "embeddings_ready": list(self.image_embeddings.keys()),
            "faiss_indexes_ready": list(self.faiss_indexes.keys()),
            "backends_available": {
                "pytorch": PYTORCH_AVAILABLE,
                "tensorflow": TENSORFLOW_AVAILABLE,
                "faiss": FAISS_AVAILABLE
            },
            "device": self.device,
            "performance_logs": len(self.performance_log)
        }
        
        if self.model_manager:
            stats["system_info"] = self.model_manager.get_system_info()
        
        return stats
    
    def cleanup_unused_models(self):
        """Cleanup models không được sử dụng để tiết kiệm memory"""
        if not self.model_manager:
            return
        
        # Get currently active models
        active = set(self.active_models.values())
        active.discard(None)
        
        # Get loaded models
        loaded = set(self.model_manager.loaded_models.keys())
        
        # Find unused models
        unused = loaded - active
        
        if unused:
            print(f"🧹 Cleaning up {len(unused)} unused models...")
            for model_key in unused:
                self.model_manager.unload_model(model_key)
            
            # Clear GPU memory after cleanup
            if self.device == "cuda":
                torch.cuda.empty_cache()
                print(f"🚀 GPU memory cleared")
    
    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """Get current GPU memory usage information"""
        if not (PYTORCH_AVAILABLE and torch.cuda.is_available()):
            return {"gpu_available": False}
        
        try:
            device_id = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device_id).total_memory
            allocated_memory = torch.cuda.memory_allocated(device_id)
            cached_memory = torch.cuda.memory_reserved(device_id)
            
            return {
                "gpu_available": True,
                "device_name": torch.cuda.get_device_name(device_id),
                "total_memory_gb": total_memory / 1024**3,
                "allocated_memory_gb": allocated_memory / 1024**3,
                "cached_memory_gb": cached_memory / 1024**3,
                "free_memory_gb": (total_memory - allocated_memory) / 1024**3,
                "memory_utilization": (allocated_memory / total_memory) * 100
            }
        except Exception as e:
            logger.error(f"Error getting GPU memory info: {e}")
            return {"gpu_available": False, "error": str(e)}
    
    def optimize_gpu_memory(self):
        """Optimize GPU memory usage"""
        if self.device == "cuda" and PYTORCH_AVAILABLE:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Get memory info
            mem_info = self.get_gpu_memory_info()
            if mem_info.get("gpu_available"):
                utilization = mem_info.get("memory_utilization", 0)
                
                print(f"🚀 GPU Memory optimized:")
                print(f"   Utilization: {utilization:.1f}%")
                print(f"   Free: {mem_info.get('free_memory_gb', 0):.1f}GB")
                
                # If memory usage is high, suggest cleanup
                if utilization > 80:
                    print("⚠️ High GPU memory usage detected. Consider calling cleanup_unused_models()")
    
    def set_batch_size(self, batch_size: int):
        """Dynamically adjust batch size based on GPU memory"""
        if self.device == "cuda":
            mem_info = self.get_gpu_memory_info()
            if mem_info.get("gpu_available"):
                free_gb = mem_info.get("free_memory_gb", 0)
                
                # Adjust batch size based on available memory
                if free_gb < 1.0:  # Less than 1GB free
                    recommended_batch = max(4, batch_size // 4)
                elif free_gb < 2.0:  # Less than 2GB free
                    recommended_batch = max(8, batch_size // 2)
                else:
                    recommended_batch = batch_size
                
                self.batch_size = recommended_batch
                print(f"🔧 Batch size adjusted to {self.batch_size} (Free GPU memory: {free_gb:.1f}GB)")
            else:
                self.batch_size = batch_size
        else:
            self.batch_size = batch_size


def demo_enhanced_search():
    """Demo function for enhanced AI search với hybrid models"""
    print("🚀 Enhanced AI Video Search Engine Demo")
    print("=" * 50)
    
    # Initialize search engine
    search_engine = EnhancedAIVideoSearchEngine()
    
    # Show available models
    if search_engine.model_manager:
        print("\n📋 Available Models:")
        models = search_engine.model_manager.get_available_models()
        for model_key, config in models.items():
            print(f"  - {model_key}: {config.name} ({config.backend.value})")
    
    # Initialize default models
    if not search_engine.initialize_default_models():
        print("❌ Failed to initialize models")
        return
    
    print(f"\n🎯 Active Models:")
    for task, model_key in search_engine.get_active_models().items():
        if model_key:
            config = search_engine.model_manager.loaded_models[model_key]
            print(f"  {task}: {config.name} ({config.backend.value})")
    
    # Build embeddings if needed
    print("\n🔄 Checking embeddings...")
    vision_model = search_engine.active_models.get("vision_language")
    if vision_model and vision_model not in search_engine.image_embeddings:
        print(f"Building embeddings with {vision_model}...")
        search_engine.build_embeddings_index(vision_model)
    
    # Interactive search loop
    print("\n🔍 Interactive Search Commands:")
    print("  - Enter text to search")
    print("  - 'models' to see model options")
    print("  - 'use <model_key>' to switch models")
    print("  - 'compare' to compare model performance")
    print("  - 'stats' to show system statistics")
    print("  - 'quit' to exit")
    
    while True:
        try:
            query = input("\n> ").strip()
            
            if query.lower() == 'quit':
                break
            elif query.lower() == 'models':
                # Show model switching options
                print("\n🎯 Currently Active Models:")
                active_models = search_engine.get_active_models()
                for task, model_key in active_models.items():
                    if model_key:
                        config = search_engine.model_manager.loaded_models[model_key]
                        print(f"  {task}: {config.config.name} ({config.config.backend.value})")
                
                # Show alternative models
                available = search_engine.model_manager.get_available_models()
                print("\n📋 Available Vision-Language Models:")
                for model_key, config in available.get("vision_language", {}).items():
                    status = "✅ loaded" if model_key in search_engine.model_manager.loaded_models else "⭕ available"
                    print(f"  - {model_key}: {config.name} ({config.backend.value}) [{status}]")
                
                print("\n📋 Available Text Embedding Models:")
                for model_key, config in available.get("text_embedding", {}).items():
                    status = "✅ loaded" if model_key in search_engine.model_manager.loaded_models else "⭕ available"
                    print(f"  - {model_key}: {config.name} ({config.backend.value}) [{status}]")
                
                continue
            elif query.lower().startswith('use '):
                # Model switching command
                model_key = query[4:].strip()
                available_models = search_engine.model_manager.available_models
                if model_key in available_models:
                    config = available_models[model_key]
                    if config.model_type == ModelType.VISION_LANGUAGE:
                        success = search_engine.set_active_model("vision_language", model_key)
                        if success:
                            print(f"✅ Switched to {config.name}")
                            # Build embeddings if needed
                            if model_key not in search_engine.image_embeddings:
                                print("🔄 Building embeddings for new model...")
                                search_engine.build_embeddings_index(model_key)
                        else:
                            print(f"❌ Failed to switch to {model_key}")
                    elif config.model_type == ModelType.TEXT_EMBEDDING:
                        success = search_engine.set_active_model("text_embedding", model_key)
                        if success:
                            print(f"✅ Switched text model to {config.name}")
                        else:
                            print(f"❌ Failed to switch to {model_key}")
                    else:
                        print(f"❌ {model_key} is not a vision-language or text embedding model")
                else:
                    print(f"❌ Model '{model_key}' not found")
                continue
            elif query.lower() == 'compare':
                # Model comparison
                test_query = input("Enter query for comparison: ").strip()
                if test_query:
                    print("\n🔄 Comparing models...")
                    comparison = search_engine.compare_models_performance(test_query)
                    
                    print("\n📊 Model Comparison Results:")
                    for model_key, result in comparison.items():
                        print(f"\n{model_key} ({result['backend']}):")
                        print(f"  Search time: {result['search_time']:.3f}s")
                        print(f"  Avg score: {result['avg_score']:.3f}")
                        print(f"  Results: {result['results_count']}")
                        
                        # Show top result
                        if result['results']:
                            top_result = result['results'][0]
                            video_name = Path(top_result['video_path']).stem
                            print(f"  Top result: {video_name} @ {top_result['timestamp']}s")
                continue
            elif query.lower() == 'stats':
                # Show statistics
                stats = search_engine.get_stats()
                print("\n📊 System Statistics:")
                print(f"  📁 Frames: {stats['frames_count']}")
                print(f"  🤖 Active models: {len([m for m in stats['active_models'].values() if m])}")
                print(f"  📊 Embeddings ready: {len(stats['embeddings_ready'])}")
                print(f"  ⚡ FAISS indexes: {len(stats['faiss_indexes_ready'])}")
                print(f"  💾 Device: {stats['device']}")
                
                # Show backends
                backends = stats['backends_available']
                print(f"\n🔧 Backend Support:")
                print(f"  PyTorch: {'✅' if backends['pytorch'] else '❌'}")
                print(f"  TensorFlow: {'✅' if backends['tensorflow'] else '❌'}")
                print(f"  FAISS: {'✅' if backends['faiss'] else '❌'}")
                
                # Performance stats
                perf_stats = search_engine.get_performance_stats()
                if perf_stats:
                    print("\n⚡ Performance Stats:")
                    for model, operations in perf_stats.items():
                        print(f"  {model}:")
                        for op, stats_data in operations.items():
                            print(f"    {op}: {stats_data['avg_time']:.3f}s avg ({stats_data['count']} runs)")
                continue
            
            if not query:
                continue
            
            # Perform search
            print(f"\n🔍 Searching for: '{query}'")
            results = search_engine.search_similar_frames(query, top_k=5)
            
            if results:
                print(f"\n📋 Top {len(results)} results:")
                for i, result in enumerate(results, 1):
                    score = result.get('similarity_score', 0)
                    model_used = result.get('model_used', 'unknown')
                    video_name = Path(result['video_path']).stem
                    timestamp = result['timestamp']
                    
                    print(f"{i}. {video_name} at {timestamp}s")
                    print(f"   Score: {score:.3f} (Model: {model_used})")
                    print(f"   Frame: {result['frame_path']}")
            else:
                print("❌ No results found")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Cleanup
    search_engine.cleanup_unused_models()


if __name__ == "__main__":
    demo_enhanced_search()

# Aliases for backward compatibility
VideoSearchEngine = EnhancedAIVideoSearchEngine
