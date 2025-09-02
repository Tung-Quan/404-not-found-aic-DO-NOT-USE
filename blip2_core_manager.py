"""
🧠 BLIP-2 Core Manager - Primary Vision-Language Model
Thay thế CLIP làm model chính, hỗ trợ query phức tạp và cross-modal understanding.
"""

import torch
import torch.nn.functional as F
from transformers import (
    Blip2Processor, 
    Blip2ForConditionalGeneration,
    Blip2VisionModel,
    AutoTokenizer
)
from PIL import Image
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Union
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplexQueryProcessor:
    """
    🔍 Xử lý query phức tạp và dài dòng
    - Intent detection (search/compare/navigate/analyze)
    - Entity extraction (objects, colors, actions, emotions)
    - Temporal reasoning (before/after/during/when)
    - Multi-constraint fusion (AND/OR/NOT logic)
    """
    
    def __init__(self):
        self.intent_patterns = {
            'search': [
                r'(tìm|find|search|look for|show me)',
                r'(có|is there|are there|does)',
                r'(ở đâu|where|which)',
            ],
            'compare': [
                r'(so sánh|compare|khác nhau|difference|similar)',
                r'(giống|like|unlike|same as)',
                r'(hơn|better|worse|more|less)',
            ],
            'analyze': [
                r'(phân tích|analyze|explain|why|how)',
                r'(diễn ra|happen|occurring|going on)',
                r'(nguyên nhân|reason|cause|because)',
            ],
            'navigate': [
                r'(trước|before|sau|after)',
                r'(tiếp theo|next|previous|then)',
                r'(đầu|beginning|cuối|end)',
            ]
        }
        
        self.entity_patterns = {
            'objects': [
                r'(người|person|people|man|woman|child)',
                r'(xe|car|bike|vehicle|motorcycle)',
                r'(nhà|house|building|room|door)',
                r'(cây|tree|plant|flower|grass)',
                r'(động vật|animal|dog|cat|bird)',
            ],
            'colors': [
                r'(đỏ|red|xanh|blue|green|vàng|yellow)',
                r'(trắng|white|đen|black|nâu|brown)',
                r'(hồng|pink|tím|purple|cam|orange)',
            ],
            'emotions': [
                r'(vui|happy|buồn|sad|giận|angry)',
                r'(sợ|scared|ngạc nhiên|surprised)',
                r'(yêu|love|ghét|hate)',
            ],
            'actions': [
                r'(chạy|run|đi|walk|ngồi|sit)',
                r'(ăn|eat|uống|drink|nói|speak)',
                r'(cười|laugh|khóc|cry|hát|sing)',
            ]
        }
        
    def parse_query(self, query: str) -> Dict:
        """
        Parse complex query thành structured format
        """
        query_lower = query.lower()
        
        # 1. Intent Detection
        intent = self._detect_intent(query_lower)
        
        # 2. Entity Extraction
        entities = self._extract_entities(query_lower)
        
        # 3. Temporal Analysis
        temporal = self._analyze_temporal(query_lower)
        
        # 4. Constraint Logic (AND/OR/NOT)
        constraints = self._parse_constraints(query_lower)
        
        # 5. Query Decomposition for complex searches
        sub_queries = self._decompose_query(query_lower, intent)
        
        return {
            'original_query': query,
            'intent': intent,
            'entities': entities,
            'temporal': temporal,
            'constraints': constraints,
            'sub_queries': sub_queries,
            'complexity_score': self._calculate_complexity(query_lower)
        }
    
    def _detect_intent(self, query: str) -> str:
        """Phát hiện intent chính của query"""
        scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = sum(len(re.findall(pattern, query)) for pattern in patterns)
            scores[intent] = score
        
        return max(scores, key=scores.get) if any(scores.values()) else 'search'
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities từ query"""
        entities = {}
        for category, patterns in self.entity_patterns.items():
            found = []
            for pattern in patterns:
                matches = re.findall(pattern, query)
                found.extend(matches)
            if found:
                entities[category] = list(set(found))
        return entities
    
    def _analyze_temporal(self, query: str) -> Dict:
        """Phân tích temporal references"""
        temporal = {
            'has_temporal': False,
            'references': [],
            'sequence': None
        }
        
        # Temporal keywords
        temporal_patterns = {
            'before': r'(trước khi|before|earlier|previously)',
            'after': r'(sau khi|after|later|then|next)',
            'during': r'(trong khi|during|while|meanwhile)',
            'sequence': r'(đầu tiên|first|cuối cùng|last|tiếp theo|next)'
        }
        
        for ref_type, pattern in temporal_patterns.items():
            if re.search(pattern, query):
                temporal['has_temporal'] = True
                temporal['references'].append(ref_type)
        
        return temporal
    
    def _parse_constraints(self, query: str) -> Dict:
        """Parse logical constraints (AND/OR/NOT)"""
        constraints = {
            'type': 'simple',
            'operators': [],
            'negations': []
        }
        
        # Detect logical operators
        if re.search(r'(và|and|&|\+)', query):
            constraints['operators'].append('AND')
            constraints['type'] = 'complex'
        
        if re.search(r'(hoặc|or|\|)', query):
            constraints['operators'].append('OR')
            constraints['type'] = 'complex'
        
        # Detect negations
        negation_patterns = [
            r'(không|not|no|without)',
            r'(trừ|except|besides)',
            r'(ngoại trừ|excluding)'
        ]
        
        for pattern in negation_patterns:
            if re.search(pattern, query):
                constraints['negations'].append(pattern)
        
        return constraints
    
    def _decompose_query(self, query: str, intent: str) -> List[str]:
        """Chia query phức tạp thành sub-queries đơn giản"""
        sub_queries = []
        
        # Split by conjunctions
        parts = re.split(r'(và|and|or|hoặc)', query)
        
        if len(parts) > 1:
            for part in parts:
                part = part.strip()
                if part and part not in ['và', 'and', 'or', 'hoặc']:
                    sub_queries.append(part)
        else:
            sub_queries = [query]
        
        return sub_queries
    
    def _calculate_complexity(self, query: str) -> float:
        """Tính complexity score của query"""
        base_score = len(query.split()) / 10.0  # Word count factor
        
        # Add complexity for logical operators
        if re.search(r'(và|and|or|hoặc)', query):
            base_score += 0.3
        
        # Add for temporal references
        if re.search(r'(trước|sau|before|after)', query):
            base_score += 0.2
        
        # Add for negations
        if re.search(r'(không|not|without)', query):
            base_score += 0.2
        
        return min(base_score, 1.0)  # Cap at 1.0


class BLIP2CoreManager:
    """
    🎯 BLIP-2 Core Vision-Language Model Manager
    
    Primary model cho video search, thay thế CLIP.
    Features:
    - Text-to-Image search với Q-Former
    - Complex query understanding
    - Cross-modal reasoning
    - Compatible với TensorFlow reranking layer
    """
    
    def __init__(self, 
                 model_name: str = "Salesforce/blip2-flan-t5-base",
                 device: str = "auto",
                 precision: str = "fp16"):
        
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.precision = precision
        
        # Models và processors
        self.processor = None
        self.model = None
        self.vision_model = None
        self.tokenizer = None
        
        # Query processor
        self.query_processor = ComplexQueryProcessor()
        
        # Cache cho embeddings
        self.text_embedding_cache = {}
        self.image_embedding_cache = {}
        
        logger.info(f"🧠 BLIP2CoreManager initialized")
        logger.info(f"📱 Device: {self.device}")
        logger.info(f"🎯 Model: {model_name}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup device với fallback logic"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def load_model(self) -> bool:
        """
        Load BLIP-2 model với error handling và fallbacks
        """
        try:
            logger.info(f"🚀 Loading BLIP-2 model: {self.model_name}")
            
            # Load processor
            self.processor = Blip2Processor.from_pretrained(self.model_name)
            logger.info("✅ Processor loaded")
            
            # Load main model
            if self.precision == "fp16" and self.device.type == "cuda":
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    self.model_name
                ).to(self.device)
            
            logger.info("✅ Main model loaded")
            
            # Load vision encoder separately for embedding extraction
            self.vision_model = Blip2VisionModel.from_pretrained(self.model_name).to(self.device)
            logger.info("✅ Vision model loaded")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            logger.info("🎉 BLIP-2 model fully loaded!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load BLIP-2: {e}")
            return False
    
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text thành embeddings
        """
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        embeddings = []
        
        for txt in texts:
            # Check cache
            if txt in self.text_embedding_cache:
                embeddings.append(self.text_embedding_cache[txt])
                continue
            
            try:
                # Process query for complex understanding
                parsed = self.query_processor.parse_query(txt)
                
                # Use main query + sub-queries for richer representation
                all_texts = [txt] + parsed.get('sub_queries', [])
                
                # Encode với BLIP-2
                inputs = self.processor(
                    text=all_texts, 
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                with torch.no_grad():
                    # Get text embeddings from Q-Former
                    text_features = self.model.get_text_features(**inputs)
                    
                    # Average embeddings if multiple sub-queries
                    if len(all_texts) > 1:
                        embedding = text_features.mean(dim=0)
                    else:
                        embedding = text_features[0]
                    
                    # Normalize
                    embedding = F.normalize(embedding, p=2, dim=0)
                    embedding_np = embedding.cpu().numpy()
                
                # Cache result
                self.text_embedding_cache[txt] = embedding_np
                embeddings.append(embedding_np)
                
            except Exception as e:
                logger.error(f"❌ Error encoding text '{txt}': {e}")
                # Fallback: zero embedding
                embeddings.append(np.zeros(768))  # BLIP-2 dimension
        
        return np.array(embeddings)
    
    def encode_image(self, image_path: Union[str, Path, Image.Image]) -> np.ndarray:
        """
        Encode image thành embeddings
        """
        try:
            # Load image
            if isinstance(image_path, (str, Path)):
                image = Image.open(image_path).convert('RGB')
                cache_key = str(image_path)
            else:
                image = image_path
                cache_key = None
            
            # Check cache
            if cache_key and cache_key in self.image_embedding_cache:
                return self.image_embedding_cache[cache_key]
            
            # Process image
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                # Get image embeddings từ vision encoder
                image_features = self.vision_model(**inputs).last_hidden_state
                
                # Pool features (mean pooling)
                embedding = image_features.mean(dim=1).squeeze()
                
                # Normalize
                embedding = F.normalize(embedding, p=2, dim=0)
                embedding_np = embedding.cpu().numpy()
            
            # Cache if possible
            if cache_key:
                self.image_embedding_cache[cache_key] = embedding_np
            
            return embedding_np
            
        except Exception as e:
            logger.error(f"❌ Error encoding image: {e}")
            return np.zeros(768)  # Fallback
    
    def batch_encode_images(self, image_paths: List[Union[str, Path]]) -> np.ndarray:
        """
        Batch encode multiple images (efficient)
        """
        embeddings = []
        
        # Group by cached vs uncached
        uncached_images = []
        uncached_indices = []
        
        for i, path in enumerate(image_paths):
            cache_key = str(path)
            if cache_key in self.image_embedding_cache:
                embeddings.append(self.image_embedding_cache[cache_key])
            else:
                uncached_images.append(path)
                uncached_indices.append(i)
                embeddings.append(None)  # Placeholder
        
        # Batch process uncached images
        if uncached_images:
            try:
                # Load images
                images = []
                for path in uncached_images:
                    try:
                        img = Image.open(path).convert('RGB')
                        images.append(img)
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to load {path}: {e}")
                        images.append(Image.new('RGB', (224, 224)))  # Dummy image
                
                # Batch process
                inputs = self.processor(
                    images=images,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    batch_features = self.vision_model(**inputs).last_hidden_state
                    batch_embeddings = batch_features.mean(dim=1)
                    batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                    batch_embeddings_np = batch_embeddings.cpu().numpy()
                
                # Cache and assign results
                for i, (path, embedding) in enumerate(zip(uncached_images, batch_embeddings_np)):
                    cache_key = str(path)
                    self.image_embedding_cache[cache_key] = embedding
                    embeddings[uncached_indices[i]] = embedding
                    
            except Exception as e:
                logger.error(f"❌ Batch encoding failed: {e}")
                # Fallback to individual encoding
                for i, path in enumerate(uncached_images):
                    embedding = self.encode_image(path)
                    embeddings[uncached_indices[i]] = embedding
        
        return np.array([emb for emb in embeddings if emb is not None])
    
    def search_similar(self, 
                      query: str, 
                      image_embeddings: np.ndarray, 
                      top_k: int = 20) -> List[Dict]:
        """
        Search similar images dựa trên query
        Returns: List of {similarity, index, parsed_query} sorted by similarity
        """
        # Parse complex query
        parsed_query = self.query_processor.parse_query(query)
        
        # Encode query
        query_embedding = self.encode_text(query)
        
        # Calculate similarities
        similarities = np.dot(image_embeddings, query_embedding.T).flatten()
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'index': int(idx),
                'similarity': float(similarities[idx]),
                'parsed_query': parsed_query,
                'complexity_score': parsed_query.get('complexity_score', 0.0)
            })
        
        return results
    
    def get_embedding_dimension(self) -> int:
        """Return embedding dimension"""
        return 768  # BLIP-2 standard dimension
    
    def clear_cache(self):
        """Clear embedding caches"""
        self.text_embedding_cache.clear()
        self.image_embedding_cache.clear()
        logger.info("🧹 Embedding caches cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'text_cache_size': len(self.text_embedding_cache),
            'image_cache_size': len(self.image_embedding_cache),
            'total_cached_items': len(self.text_embedding_cache) + len(self.image_embedding_cache)
        }


# Export main class
__all__ = ['BLIP2CoreManager', 'ComplexQueryProcessor']
