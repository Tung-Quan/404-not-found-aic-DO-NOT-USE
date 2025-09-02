"""
🚀 Simplified AI Search System - Single Model + OCR
===================================================
Hệ thống tìm kiếm AI đơn giản hóa chỉ sử dụng 1 mô hình chính với OCR tích hợp
- Model chính: CLIP (lightweight và stable)
- OCR: VietOCR cho text recognition 
- Architecture: Streamlined single-model approach
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any
import json
import faiss
import pickle
from datetime import datetime
import re

# OCR imports
try:
    import vietocr
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# CLIP imports
USE_TRANSFORMERS_CLIP = False  # Initialize default value
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    try:
        from transformers import CLIPProcessor, CLIPModel
        CLIP_AVAILABLE = True
        USE_TRANSFORMERS_CLIP = True
    except ImportError:
        CLIP_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplifiedSearchEngine:
    """
    🔍 Simplified AI Search Engine
    Sử dụng 1 mô hình chính (CLIP) + OCR cho tìm kiếm multimodal
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = "auto"):
        """
        Initialize với 1 mô hình chính
        
        Args:
            model_name: CLIP model variant (ViT-B/32, ViT-L/14)
            device: cuda, cpu, hoặc auto
        """
        self.device = self._setup_device(device)
        self.model_name = model_name
        
        # Load CLIP model
        self.clip_model = None
        self.clip_processor = None
        self._load_clip_model()
        
        # Load OCR model
        self.ocr_predictor = None
        self._load_ocr_model()
        
        # Embeddings storage
        self.embeddings_cache = {}
        self.index = None
        self.frame_paths = []
        self.metadata = []
        
        logger.info(f"✅ Simplified Search Engine initialized on {self.device}")
    
    def _setup_device(self, device: str) -> str:
        """Setup optimal device"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                logger.info("💻 Using CPU")
        return device
    
    def _load_clip_model(self):
        """Load CLIP model - single model approach"""
        try:
            if CLIP_AVAILABLE and not USE_TRANSFORMERS_CLIP:
                # OpenAI CLIP
                self.clip_model, self.clip_processor = clip.load(self.model_name, device=self.device)
                self.clip_model.eval()
                logger.info(f"✅ Loaded OpenAI CLIP: {self.model_name}")
            else:
                # HuggingFace CLIP
                model_name_hf = "openai/clip-vit-base-patch32"
                self.clip_model = CLIPModel.from_pretrained(model_name_hf).to(self.device)
                self.clip_processor = CLIPProcessor.from_pretrained(model_name_hf)
                self.clip_model.eval()
                logger.info(f"✅ Loaded HuggingFace CLIP: {model_name_hf}")
        except Exception as e:
            logger.error(f"❌ Failed to load CLIP: {e}")
            raise
    
    def _load_ocr_model(self):
        """Load VietOCR model"""
        if not OCR_AVAILABLE:
            logger.warning("⚠️ VietOCR not available. Install: pip install vietocr")
            return
            
        try:
            # VietOCR configuration
            config = Cfg.load_config_from_name('vgg_transformer')
            config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
            config['cnn']['pretrained'] = False
            config['device'] = self.device
            config['predictor']['beamsearch'] = False
            
            self.ocr_predictor = Predictor(config)
            logger.info("✅ VietOCR loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load VietOCR: {e}")
            self.ocr_predictor = None
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text using CLIP
        
        Args:
            text: Input text query
            
        Returns:
            Text embedding vector
        """
        try:
            with torch.no_grad():
                if USE_TRANSFORMERS_CLIP:  # HuggingFace CLIP
                    inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    text_features = self.clip_model.get_text_features(**inputs)
                else:  # OpenAI CLIP
                    text_tokens = clip.tokenize([text]).to(self.device)
                    text_features = self.clip_model.encode_text(text_tokens)
                
                # Normalize
                text_features = F.normalize(text_features, dim=-1)
                return text_features.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"❌ Text encoding failed: {e}")
            return np.zeros(512)
    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Encode image using CLIP
        
        Args:
            image: PIL Image
            
        Returns:
            Image embedding vector
        """
        try:
            with torch.no_grad():
                if USE_TRANSFORMERS_CLIP:  # HuggingFace CLIP
                    inputs = self.clip_processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    image_features = self.clip_model.get_image_features(**inputs)
                else:  # OpenAI CLIP
                    image_input = self.clip_processor(image).unsqueeze(0).to(self.device)
                    image_features = self.clip_model.encode_image(image_input)
                
                # Normalize
                image_features = F.normalize(image_features, dim=-1)
                return image_features.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"❌ Image encoding failed: {e}")
            return np.zeros(512)
    
    def extract_text_from_image(self, image: Image.Image) -> str:
        """
        Extract text from image using OCR
        
        Args:
            image: PIL Image
            
        Returns:
            Extracted text string
        """
        if not self.ocr_predictor:
            return ""
            
        try:
            # Convert PIL to numpy array
            img_array = np.array(image)
            
            # Extract text
            extracted_text = self.ocr_predictor.predict(img_array)
            
            # Clean text
            cleaned_text = re.sub(r'[^\w\s]', ' ', extracted_text)
            cleaned_text = ' '.join(cleaned_text.split())
            
            return cleaned_text
        except Exception as e:
            logger.error(f"❌ OCR extraction failed: {e}")
            return ""
    
    def process_image_multimodal(self, image_path: str) -> Dict[str, Any]:
        """
        Process image with both vision and OCR
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with visual embedding and extracted text
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Get visual embedding
            visual_embedding = self.encode_image(image)
            
            # Get text via OCR
            extracted_text = self.extract_text_from_image(image)
            
            # Get text embedding if OCR found text
            text_embedding = None
            if extracted_text.strip():
                text_embedding = self.encode_text(extracted_text)
            
            return {
                'visual_embedding': visual_embedding,
                'text_embedding': text_embedding,
                'extracted_text': extracted_text,
                'image_path': image_path
            }
        except Exception as e:
            logger.error(f"❌ Failed to process {image_path}: {e}")
            return {
                'visual_embedding': np.zeros(512),
                'text_embedding': None,
                'extracted_text': '',
                'image_path': image_path
            }
    
    def build_index(self, frames_directory: str, save_path: str = "index"):
        """
        Build search index from frames directory
        
        Args:
            frames_directory: Path to directory containing frame images
            save_path: Path to save index files
        """
        logger.info(f"🔨 Building index from {frames_directory}")
        
        # Find all image files
        frames_dir = Path(frames_directory)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(frames_dir.rglob(f"*{ext}"))
        
        logger.info(f"📁 Found {len(image_files)} images")
        
        # Process images
        visual_embeddings = []
        text_embeddings = []
        metadata = []
        
        for i, image_path in enumerate(image_files):
            if i % 100 == 0:
                logger.info(f"📊 Processing {i}/{len(image_files)}")
            
            result = self.process_image_multimodal(str(image_path))
            
            visual_embeddings.append(result['visual_embedding'])
            text_embeddings.append(result['text_embedding'])
            
            metadata.append({
                'path': str(image_path),
                'extracted_text': result['extracted_text'],
                'has_text': bool(result['extracted_text'].strip())
            })
        
        # Convert to numpy arrays
        visual_embeddings = np.array(visual_embeddings).astype('float32')
        
        # Build FAISS index for visual embeddings
        embedding_dim = visual_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(visual_embeddings)
        self.index.add(visual_embeddings)
        
        # Store metadata
        self.metadata = metadata
        self.frame_paths = [m['path'] for m in metadata]
        
        # Save index
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True)
        
        faiss.write_index(self.index, str(save_dir / "visual_index.faiss"))
        
        with open(save_dir / "metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Save text embeddings if any
        valid_text_embeddings = [emb for emb in text_embeddings if emb is not None]
        if valid_text_embeddings:
            text_embeddings_array = np.array(valid_text_embeddings).astype('float32')
            np.save(save_dir / "text_embeddings.npy", text_embeddings_array)
        
        logger.info(f"✅ Index built and saved to {save_path}")
        logger.info(f"📊 Visual embeddings: {len(visual_embeddings)}")
        logger.info(f"📝 Text embeddings: {len(valid_text_embeddings)}")
    
    def load_index(self, index_path: str):
        """
        Load pre-built index
        
        Args:
            index_path: Path to saved index directory
        """
        index_dir = Path(index_path)
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_dir / "visual_index.faiss"))
            
            # Load metadata
            with open(index_dir / "metadata.pkl", 'rb') as f:
                self.metadata = pickle.load(f)
            
            self.frame_paths = [m['path'] for m in self.metadata]
            
            logger.info(f"✅ Index loaded from {index_path}")
            logger.info(f"📊 Total frames: {len(self.metadata)}")
        except Exception as e:
            logger.error(f"❌ Failed to load index: {e}")
            raise
    
    def search(self, query: str, top_k: int = 20, search_mode: str = "hybrid") -> List[Dict[str, Any]]:
        """
        Search using text query
        
        Args:
            query: Text search query
            top_k: Number of results to return
            search_mode: "visual", "text", or "hybrid"
            
        Returns:
            List of search results with scores and metadata
        """
        if not self.index:
            logger.error("❌ Index not loaded. Call load_index() first.")
            return []
        
        # Encode query
        query_embedding = self.encode_text(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search visual index
        scores, indices = self.index.search(query_embedding, min(top_k * 2, len(self.metadata)))
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
                
            metadata = self.metadata[idx]
            result = {
                'path': metadata['path'],
                'score': float(score),
                'extracted_text': metadata['extracted_text'],
                'has_text': metadata['has_text'],
                'rank': len(results) + 1
            }
            
            # Text matching boost
            if search_mode in ["text", "hybrid"] and metadata['has_text']:
                query_lower = query.lower()
                text_lower = metadata['extracted_text'].lower()
                
                # Simple text similarity boost
                if query_lower in text_lower:
                    result['score'] += 0.2
                elif any(word in text_lower for word in query_lower.split()):
                    result['score'] += 0.1
            
            results.append(result)
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Update ranks
        for i, result in enumerate(results[:top_k]):
            result['rank'] = i + 1
        
        return results[:top_k]

# Global instance
simplified_engine = None

def get_engine() -> SimplifiedSearchEngine:
    """Get global engine instance"""
    global simplified_engine
    if simplified_engine is None:
        simplified_engine = SimplifiedSearchEngine()
    return simplified_engine

def initialize_engine(model_name: str = "ViT-B/32", device: str = "auto") -> SimplifiedSearchEngine:
    """Initialize global engine with specific config"""
    global simplified_engine
    simplified_engine = SimplifiedSearchEngine(model_name, device)
    return simplified_engine
