#!/usr/bin/env python3
"""
Enhanced Hybrid AI Model Manager - Complete Version
Quản lý AI models với RTX 3060 GPU optimization
Tích hợp AI Agents và TensorFlow models
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from datetime import datetime
from enum import Enum

# Import AI Agent và TensorFlow managers
try:
    from ai_agent_manager import AIAgentManager, ai_agent_manager
    AI_AGENTS_AVAILABLE = True
except ImportError:
    AI_AGENTS_AVAILABLE = False
    ai_agent_manager = None

try:
    from tensorflow_model_manager import TensorFlowModelManager, tensorflow_model_manager
    TENSORFLOW_MODELS_AVAILABLE = True
except ImportError:
    TENSORFLOW_MODELS_AVAILABLE = False
    tensorflow_model_manager = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelBackend(Enum):
    """Các backend model khả dụng"""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    AUTO = "auto"

class ModelType(Enum):
    """Các loại model"""
    VISION_LANGUAGE = "vision_language"  # CLIP
    IMAGE_CAPTIONING = "image_captioning"  # BLIP
    TEXT_EMBEDDING = "text_embedding"  # Basic embeddings

# PyTorch models - Focus on working components
PYTORCH_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
try:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    PYTORCH_AVAILABLE = True
    print("✅ PyTorch backend available")
    
    # Import only essential transformers components
    try:
        from transformers import (
            CLIPProcessor, CLIPModel,
            BlipProcessor, BlipForConditionalGeneration,
            AutoTokenizer, AutoModel
        )
        TRANSFORMERS_AVAILABLE = True
        print("✅ Transformers (core) available")
    except ImportError as e:
        print(f"⚠️ Transformers not available: {e}")
        TRANSFORMERS_AVAILABLE = False
    except Exception as e:
        print(f"⚠️ Transformers error: {e}")
        TRANSFORMERS_AVAILABLE = False

except ImportError as e:
    print(f"⚠️ PyTorch backend not available: {e}")

# TensorFlow models - Optional
TENSORFLOW_AVAILABLE = False
TENSORFLOW_HUB_AVAILABLE = False
try:
    import tensorflow as tf
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    
    try:
        import tensorflow_hub as hub
        TENSORFLOW_HUB_AVAILABLE = True
        print("✅ TensorFlow Hub available")
    except ImportError:
        print("⚠️ TensorFlow Hub not available")
        TENSORFLOW_HUB_AVAILABLE = False
    
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow backend available")
except ImportError:
    print("⚠️ TensorFlow backend not available")
except Exception as e:
    print(f"⚠️ TensorFlow initialization error: {e}")
    TENSORFLOW_AVAILABLE = False

# FAISS for vector similarity
FAISS_AVAILABLE = False
try:
    import faiss
    FAISS_AVAILABLE = True
    print("✅ FAISS available")
except ImportError:
    print("⚠️ FAISS not available")

class ModelConfig:
    """Configuration for a model"""
    def __init__(self, name: str, model_type: ModelType, backend: ModelBackend, 
                 model_path: str, description: str = "", 
                 performance: Dict[str, str] = None, requirements: List[str] = None,
                 capabilities: List[str] = None):
        self.name = name
        self.model_type = model_type
        self.backend = backend
        self.model_path = model_path
        self.description = description
        self.performance = performance or {}
        self.requirements = requirements or []
        self.capabilities = capabilities or [model_type.value]  # Default to primary type
        
        # Runtime properties
        self.loaded = False
        self.model = None
        self.processor = None
        self.device = None

class EnhancedHybridModelManager:
    """Enhanced model manager với GPU optimization cho RTX 3060"""
    
    def __init__(self, device: str = "auto"):
        print(f"🤖 Enhanced Hybrid Model Manager starting...")
        
        # Device selection
        self.device = self._setup_device(device)
        print(f"💾 Using device: {self.device}")
        
        # Initialize external managers
        self.ai_agent_manager = None
        self.tensorflow_manager = None
        
        # Initialize AI Agent Manager if available
        if AI_AGENTS_AVAILABLE and ai_agent_manager:
            try:
                self.ai_agent_manager = ai_agent_manager
                print(f"🤖 AI Agent Manager loaded")
            except Exception as e:
                print(f"⚠️ AI Agent Manager failed to load: {e}")
                self.ai_agent_manager = None
        else:
            print(f"⚠️ AI Agent Manager not available")
        
        # Initialize TensorFlow Manager if available  
        if TENSORFLOW_MODELS_AVAILABLE and tensorflow_model_manager:
            try:
                self.tensorflow_manager = tensorflow_model_manager
                print(f"🔧 TensorFlow Manager loaded")
            except Exception as e:
                print(f"⚠️ TensorFlow Manager failed to load: {e}")
                self.tensorflow_manager = None
        else:
            print(f"⚠️ TensorFlow Manager not available")
        
        # Model storage
        self.available_models: Dict[str, ModelConfig] = {}
        self.loaded_models: Dict[str, ModelConfig] = {}
        self.active_models: Dict[str, str] = {
            "vision_language": None,
            "image_captioning": None,
            "text_embedding": None
        }
        
        # GPU optimization settings
        self.use_mixed_precision = True if self.device == "cuda" else False
        self.batch_size = 16  # Conservative for RTX 3060
        
        # Initialize model configurations
        self._setup_model_configs()
        
        print(f"🚀 Enhanced manager ready with {len(self.available_models)} models")
        print(f"⚡ Mixed precision: {self.use_mixed_precision}")
    
    def _setup_device(self, device: str = "auto") -> str:
        """Setup optimal device"""
        if device == "auto":
            if PYTORCH_AVAILABLE and torch.cuda.is_available():
                # Test GPU functionality
                try:
                    test_tensor = torch.randn(100, 100).cuda()
                    result = torch.mm(test_tensor, test_tensor.T)
                    del test_tensor, result
                    torch.cuda.empty_cache()
                    
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    
                    print(f"🎮 GPU detected: {gpu_name}")
                    print(f"💾 GPU memory: {gpu_memory:.1f} GB")
                    
                    return "cuda"
                except Exception as e:
                    print(f"⚠️ GPU test failed: {e}")
                    return "cpu"
            else:
                print("💻 Using CPU device")
                return "cpu"
        else:
            return device
    
    def _setup_model_configs(self):
        """Setup available model configurations"""
        
        # CLIP Models (Vision-Language)
        if TRANSFORMERS_AVAILABLE:
            self.available_models["clip_vit_base"] = ModelConfig(
                name="CLIP ViT Base",
                model_type=ModelType.VISION_LANGUAGE,
                backend=ModelBackend.PYTORCH,
                model_path="openai/clip-vit-base-patch32",
                description="OpenAI CLIP model cho vision-language understanding",
                performance={"speed": "fast", "accuracy": "good", "memory": "medium"},
                requirements=["torch", "transformers", "pillow"],
                capabilities=["vision_language", "text_embedding"]  # Dual capability
            )
            
            self.available_models["clip_vit_large"] = ModelConfig(
                name="CLIP ViT Large",
                model_type=ModelType.VISION_LANGUAGE,
                backend=ModelBackend.PYTORCH,
                model_path="openai/clip-vit-large-patch14",
                description="CLIP Large model cho độ chính xác cao hơn",
                performance={"speed": "medium", "accuracy": "excellent", "memory": "high"},
                requirements=["torch", "transformers", "pillow"],
                capabilities=["vision_language", "text_embedding"]  # Dual capability
            )
            
            # BLIP Models (Image Captioning)
            self.available_models["blip_base"] = ModelConfig(
                name="BLIP Base",
                model_type=ModelType.IMAGE_CAPTIONING,
                backend=ModelBackend.PYTORCH,
                model_path="Salesforce/blip-image-captioning-base",
                description="Salesforce BLIP cho image captioning",
                performance={"speed": "medium", "accuracy": "good", "memory": "medium"},
                requirements=["torch", "transformers", "pillow"]
            )
        
        # TensorFlow Hub models (if available)
        if TENSORFLOW_HUB_AVAILABLE:
            self.available_models["use_v4"] = ModelConfig(
                name="Universal Sentence Encoder v4",
                model_type=ModelType.TEXT_EMBEDDING,
                backend=ModelBackend.TENSORFLOW,
                model_path="https://tfhub.dev/google/universal-sentence-encoder/4",
                description="Google Universal Sentence Encoder",
                performance={"speed": "fast", "accuracy": "good", "memory": "medium"},
                requirements=["tensorflow", "tensorflow-hub"]
            )
    
    def recommend_models(self, use_case: str = "general") -> Dict[str, str]:
        """Gợi ý models phù hợp cho use case"""
        recommendations = {
            "fast_search": {
                "vision_language": "clip_vit_base",
                "image_captioning": "blip_base",
                "text_embedding": "clip_vit_base"
            },
            "high_quality": {
                "vision_language": "clip_vit_large", 
                "image_captioning": "blip_base",
                "text_embedding": "clip_vit_large"
            },
            "general": {
                "vision_language": "clip_vit_base",
                "image_captioning": "blip_base",
                "text_embedding": "clip_vit_base"
            }
        }
        
        return recommendations.get(use_case, recommendations["general"])
    
    def get_available_models(self, model_type: Optional[ModelType] = None) -> Dict[str, ModelConfig]:
        """Lấy danh sách models khả dụng"""
        if model_type is None:
            return self.available_models
        
        return {
            key: config for key, config in self.available_models.items()
            if config.model_type == model_type
        }
    
    def load_model(self, model_key: str) -> bool:
        """Load một model với GPU optimization"""
        if model_key not in self.available_models:
            print(f"❌ Model '{model_key}' not found")
            return False
        
        config = self.available_models[model_key]
        
        if config.loaded:
            print(f"✅ Model '{config.name}' already loaded")
            return True
        
        print(f"🔄 Loading {config.name} ({config.backend.value})...")
        
        try:
            if config.backend == ModelBackend.PYTORCH:
                success = self._load_pytorch_model(config)
            elif config.backend == ModelBackend.TENSORFLOW:
                success = self._load_tensorflow_model(config)
            else:
                print(f"❌ Unknown backend: {config.backend}")
                return False
            
            if success:
                config.loaded = True
                config.device = self.device
                self.loaded_models[model_key] = config
                print(f"✅ {config.name} loaded successfully on {self.device}")
                return True
            else:
                print(f"❌ Failed to load {config.name}")
                return False
                
        except Exception as e:
            print(f"❌ Error loading {config.name}: {e}")
            return False
    
    def _load_pytorch_model(self, config: ModelConfig) -> bool:
        """Load PyTorch model với GPU optimization"""
        if not PYTORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            print(f"❌ PyTorch/Transformers not available for {config.name}")
            return False
        
        try:
            if config.model_type == ModelType.VISION_LANGUAGE:
                # CLIP models
                config.model = CLIPModel.from_pretrained(config.model_path)
                config.processor = CLIPProcessor.from_pretrained(config.model_path)
                
                # Move to device
                config.model.to(self.device)
                config.model.eval()
                
                # Enable mixed precision if using GPU
                if self.device == "cuda" and self.use_mixed_precision:
                    config.model = config.model.half()
                
            elif config.model_type == ModelType.IMAGE_CAPTIONING:
                # BLIP models
                config.model = BlipForConditionalGeneration.from_pretrained(config.model_path)
                config.processor = BlipProcessor.from_pretrained(config.model_path)
                
                # Move to device
                config.model.to(self.device)
                config.model.eval()
                
                # Enable mixed precision if using GPU
                if self.device == "cuda" and self.use_mixed_precision:
                    config.model = config.model.half()
            
            return True
            
        except Exception as e:
            print(f"PyTorch loading error for {config.name}: {e}")
            return False
    
    def _load_tensorflow_model(self, config: ModelConfig) -> bool:
        """Load TensorFlow model"""
        if not TENSORFLOW_AVAILABLE or not TENSORFLOW_HUB_AVAILABLE:
            print(f"❌ TensorFlow not available for {config.name}")
            return False
        
        try:
            config.model = hub.load(config.model_path)
            print(f"✅ TensorFlow model loaded: {config.model_path}")
            return True
            
        except Exception as e:
            print(f"TensorFlow loading error for {config.name}: {e}")
            return False
    
    def encode_image(self, image_path: str, model_key: str = None) -> Optional[np.ndarray]:
        """Encode image using loaded model"""
        if model_key is None:
            model_key = self.active_models.get("vision_language")
        
        if not model_key or model_key not in self.loaded_models:
            print(f"❌ No loaded model for image encoding")
            return None
        
        config = self.loaded_models[model_key]
        
        try:
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            
            if config.model_type == ModelType.VISION_LANGUAGE:
                # CLIP encoding
                inputs = config.processor(images=image, return_tensors="pt")
                
                # Move to device
                for key in inputs:
                    inputs[key] = inputs[key].to(self.device)
                
                with torch.no_grad():
                    if self.use_mixed_precision and self.device == "cuda":
                        with torch.autocast(device_type='cuda'):
                            image_features = config.model.get_image_features(**inputs)
                    else:
                        image_features = config.model.get_image_features(**inputs)
                
                # Normalize features
                image_features = F.normalize(image_features, p=2, dim=1)
                
                return image_features.cpu().numpy()
            
        except Exception as e:
            print(f"❌ Error encoding image: {e}")
            return None
    
    def encode_text(self, text: str, model_key: str = None) -> Optional[np.ndarray]:
        """Encode text using loaded model"""
        if model_key is None:
            model_key = self.active_models.get("vision_language")
        
        if not model_key or model_key not in self.loaded_models:
            print(f"❌ No loaded model for text encoding")
            return None
        
        config = self.loaded_models[model_key]
        
        try:
            if config.model_type == ModelType.VISION_LANGUAGE:
                # CLIP text encoding
                inputs = config.processor(text=[text], return_tensors="pt", padding=True)
                
                # Move to device
                for key in inputs:
                    inputs[key] = inputs[key].to(self.device)
                
                with torch.no_grad():
                    if self.use_mixed_precision and self.device == "cuda":
                        with torch.autocast(device_type='cuda'):
                            text_features = config.model.get_text_features(**inputs)
                    else:
                        text_features = config.model.get_text_features(**inputs)
                
                # Normalize features
                text_features = F.normalize(text_features, p=2, dim=1)
                
                return text_features.cpu().numpy()
                
        except Exception as e:
            print(f"❌ Error encoding text: {e}")
            return None
    
    def set_active_model(self, model_type: str, model_key: str) -> bool:
        """Set active model for a specific type"""
        if model_key not in self.loaded_models:
            print(f"❌ Model {model_key} not loaded")
            return False
        
        self.active_models[model_type] = model_key
        config = self.loaded_models[model_key]
        print(f"✅ Set {model_type} model to: {config.name}")
        return True
    
    def get_model_info(self, model_key: str) -> Dict[str, Any]:
        """Lấy thông tin chi tiết về model"""
        if model_key not in self.available_models:
            return {}
        
        config = self.available_models[model_key]
        
        return {
            "name": config.name,
            "type": config.model_type.value,
            "backend": config.backend.value,
            "description": config.description,
            "performance": config.performance,
            "requirements": config.requirements,
            "loaded": config.loaded,
            "device": config.device if config.loaded else None,
            "available": self._check_model_availability(config)
        }
    
    def _check_model_availability(self, config: ModelConfig) -> bool:
        """Kiểm tra model có thể load được không"""
        if config.backend == ModelBackend.PYTORCH:
            if not PYTORCH_AVAILABLE:
                return False
            
            # Check specific requirements
            if config.model_type in [ModelType.VISION_LANGUAGE, ModelType.IMAGE_CAPTIONING]:
                return TRANSFORMERS_AVAILABLE
            return True
            
        elif config.backend == ModelBackend.TENSORFLOW:
            return TENSORFLOW_AVAILABLE and TENSORFLOW_HUB_AVAILABLE
        
        return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Lấy thông tin hệ thống"""
        gpu_info = {}
        if PYTORCH_AVAILABLE and torch.cuda.is_available():
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                "cuda_version": torch.version.cuda
            }
        
        return {
            "backends": {
                "pytorch": PYTORCH_AVAILABLE,
                "tensorflow": TENSORFLOW_AVAILABLE,
                "transformers": TRANSFORMERS_AVAILABLE,
                "tensorflow_hub": TENSORFLOW_HUB_AVAILABLE,
                "faiss": FAISS_AVAILABLE
            },
            "device": self.device,
            "gpu_info": gpu_info,
            "mixed_precision": self.use_mixed_precision,
            "available_models": len(self.available_models),
            "loaded_models": len(self.loaded_models)
        }
    
    def unload_model(self, model_key: str):
        """Unload một model để giải phóng memory"""
        if model_key in self.loaded_models:
            config = self.loaded_models[model_key]
            
            # Clear GPU memory
            if config.model is not None:
                del config.model
            if config.processor is not None:
                del config.processor
            
            config.model = None
            config.processor = None
            config.loaded = False
            config.device = None
            
            del self.loaded_models[model_key]
            
            # Clear GPU cache
            if PYTORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"🗑️ Unloaded {config.name}")
    
    def clear_gpu_memory(self):
        """Clear GPU memory"""
        if PYTORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU memory cleared")
    
    def get_enhanced_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information including AI agents and TensorFlow"""
        base_info = self.get_system_info()
        
        # Add AI agents info
        if self.ai_agent_manager:
            base_info["ai_agents"] = self.get_ai_agents_status()
        else:
            base_info["ai_agents"] = {"available": False, "reason": "AI Agent Manager not loaded"}
        
        # Add TensorFlow models info
        if self.tensorflow_manager:
            base_info["tensorflow_models"] = self.get_tensorflow_models_status()
        else:
            base_info["tensorflow_models"] = {"available": False, "reason": "TensorFlow Manager not loaded"}
        
        # Add availability flags
        base_info["enhanced_features"] = {
            "ai_agents_available": AI_AGENTS_AVAILABLE,
            "tensorflow_models_available": TENSORFLOW_MODELS_AVAILABLE,
            "total_capabilities": len(self.available_models) + \
                               (len(self.tensorflow_manager.configs) if self.tensorflow_manager else 0)
        }
        
        return base_info
    
    def get_ai_agents_status(self) -> Dict[str, Any]:
        """Get AI agents status"""
        if not self.ai_agent_manager:
            return {"available": False, "reason": "AI Agent Manager not loaded"}
        
        return {
            "available": True,
            "agents": self.ai_agent_manager.get_available_agents(),
            "api_keys_configured": {
                "openai": bool(self.ai_agent_manager.api_keys.get("openai")),
                "anthropic": bool(self.ai_agent_manager.api_keys.get("anthropic"))
            }
        }
    
    def get_tensorflow_models_status(self) -> Dict[str, Any]:
        """Get TensorFlow models status"""
        if not self.tensorflow_manager:
            return {"available": False, "reason": "TensorFlow Manager not loaded"}
        
        return {
            "available": True,
            "models": self.tensorflow_manager.get_available_models(),
            "memory_usage": self.tensorflow_manager.get_memory_usage()
        }

# Alias for backward compatibility
HybridModelManager = EnhancedHybridModelManager

if __name__ == "__main__":
    # Test the enhanced manager
    print("🧪 Testing Enhanced Hybrid Model Manager...")
    print("=" * 50)
    
    manager = EnhancedHybridModelManager()
    
    # Show system info
    print("\n📊 System Info:")
    system_info = manager.get_system_info()
    for key, value in system_info.items():
        print(f"   {key}: {value}")
    
    # Show available models
    print(f"\n📋 Available Models ({len(manager.available_models)}):")
    for key, config in manager.available_models.items():
        status = "✅" if manager._check_model_availability(config) else "❌"
        print(f"   {status} {key}: {config.name}")
    
    print("=" * 50)
    print("✅ Enhanced manager test completed")

    # ===== NEW ENHANCED FEATURES =====
    
    def _setup_ai_agents(self):
        """Setup AI agents if available"""
        if not self.ai_agent_manager:
            return
        
        print("🤖 Setting up AI agents...")
        
        # Initialize key agents
        agents_to_init = ["gpt4_vision", "gpt4_text", "claude3", "blip_local"]
        
        for agent_name in agents_to_init:
            if self.ai_agent_manager.initialize_agent(agent_name):
                print(f"   ✅ {agent_name} initialized")
            else:
                print(f"   ⚠️ {agent_name} not available")
    
    def analyze_frame_with_agent(self, image_path: str, agent_name: str = "gpt4_vision") -> Optional[str]:
        """Analyze frame using AI agent"""
        if not self.ai_agent_manager:
            return None
        
        return self.ai_agent_manager.analyze_frame(image_path, agent_name)
    
    def generate_search_query_with_agent(self, user_query: str, agent_name: str = "gpt4_text") -> Optional[str]:
        """Generate optimized search query using AI agent"""
        if not self.ai_agent_manager:
            return user_query
        
        optimized = self.ai_agent_manager.generate_search_query(user_query, agent_name)
        return optimized if optimized else user_query
    
    def summarize_results_with_agent(self, search_results: List[Dict], agent_name: str = "gpt4_text") -> Optional[str]:
        """Summarize search results using AI agent"""
        if not self.ai_agent_manager:
            return None
        
        return self.ai_agent_manager.summarize_results(search_results, agent_name)
    
    def extract_tensorflow_features(self, image_path: str, model_key: str = "mobilenet_v2") -> Optional[np.ndarray]:
        """Extract features using TensorFlow models"""
        if not self.tensorflow_manager:
            return None
        
        return self.tensorflow_manager.extract_image_features(image_path, model_key)
    
    def encode_text_tensorflow(self, text: str, model_key: str = "universal_sentence_encoder") -> Optional[np.ndarray]:
        """Encode text using TensorFlow models"""
        if not self.tensorflow_manager:
            return None
        
        return self.tensorflow_manager.encode_text(text, model_key)
    
    def detect_objects_tensorflow(self, image_path: str, model_key: str = "ssd_mobilenet") -> Optional[Dict]:
        """Detect objects using TensorFlow models"""
        if not self.tensorflow_manager:
            return None
        
        return self.tensorflow_manager.detect_objects(image_path, model_key)
    
    def get_ai_agents_status(self) -> Dict[str, Any]:
        """Get AI agents status"""
        if not self.ai_agent_manager:
            return {"available": False, "reason": "AI Agent Manager not loaded"}
        
        return {
            "available": True,
            "agents": self.ai_agent_manager.get_available_agents(),
            "api_keys_configured": {
                "openai": bool(self.ai_agent_manager.api_keys.get("openai")),
                "anthropic": bool(self.ai_agent_manager.api_keys.get("anthropic"))
            }
        }
    
    def get_tensorflow_models_status(self) -> Dict[str, Any]:
        """Get TensorFlow models status"""
        if not self.tensorflow_manager:
            return {"available": False, "reason": "TensorFlow Manager not loaded"}
        
        return {
            "available": True,
            "models": self.tensorflow_manager.get_available_models(),
            "memory_usage": self.tensorflow_manager.get_memory_usage()
        }
    
    def get_enhanced_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information including AI agents and TensorFlow"""
        base_info = self.get_system_info()
        
        # Add AI agents info
        base_info["ai_agents"] = self.get_ai_agents_status()
        
        # Add TensorFlow models info
        base_info["tensorflow_models"] = self.get_tensorflow_models_status()
        
        # Add availability flags
        base_info["enhanced_features"] = {
            "ai_agents_available": AI_AGENTS_AVAILABLE,
            "tensorflow_models_available": TENSORFLOW_MODELS_AVAILABLE,
            "total_capabilities": len(self.available_models) + \
                               (len(self.tensorflow_manager.configs) if self.tensorflow_manager else 0)
        }
        
        return base_info

def test_enhanced_manager():
    """Test enhanced hybrid manager với đầy đủ features"""
    print("🧪 Testing Enhanced Hybrid Model Manager with AI Agents & TensorFlow...")
    print("=" * 70)
    
    manager = EnhancedHybridModelManager()
    
    # Show enhanced system info
    print("\n📊 Enhanced System Info:")
    system_info = manager.get_enhanced_system_info()
    for key, value in system_info.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for subkey, subvalue in value.items():
                print(f"     {subkey}: {subvalue}")
        else:
            print(f"   {key}: {value}")
    
    # Show available models
    print(f"\n📋 Available Models ({len(manager.available_models)}):")
    for key, config in manager.available_models.items():
        status = "✅" if manager._check_model_availability(config) else "❌"
        print(f"   {status} {key}: {config.name}")
    
    # Test AI agents if available
    if manager.ai_agent_manager:
        print(f"\n🤖 AI Agents Status:")
        agents_status = manager.get_ai_agents_status()
        if agents_status["available"]:
            for name, info in agents_status["agents"].items():
                status = "✅" if info["initialized"] else "⚠️"
                print(f"   {status} {name}: {info['name']} ({info['provider']})")
    
    # Test TensorFlow models if available
    if manager.tensorflow_manager:
        print(f"\n🔧 TensorFlow Models Status:")
        tf_status = manager.get_tensorflow_models_status()
        if tf_status["available"]:
            for name, info in tf_status["models"].items():
                status = "✅" if info["loaded"] else "⚠️"
                print(f"   {status} {name}: {info['name']} ({info['type']})")
    
    print("=" * 70)
    print("✅ Enhanced manager test completed")

# Aliases for backward compatibility
EnhancedHybridManager = EnhancedHybridModelManager
