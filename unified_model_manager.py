#!/usr/bin/env python3
"""
🔄 UNIFIED MODEL MANAGER - Simplified & Consolidated
Replaces Enhanced Hybrid Manager, TensorFlow Manager, and AI Agent Manager
Single source of truth for all AI models
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
from enum import Enum

# Core model imports with fallbacks
class ModelBackend(Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow" 
    LOCAL = "local"

class ModelType(Enum):
    VISION_LANGUAGE = "vision_language"
    TEXT_EMBEDDING = "text_embedding"
    IMAGE_CAPTIONING = "image_captioning"
    OBJECT_DETECTION = "object_detection"

# Import check and availability
AVAILABILITY = {
    "pytorch": False,
    "transformers": False,
    "tensorflow": False,
    "blip2": False,
    "clip": False
}

# PyTorch and Transformers
try:
    import torch
    AVAILABILITY["pytorch"] = True
    
    try:
        from transformers import (
            # BLIP-2 models
            Blip2Processor, Blip2ForConditionalGeneration,
            # CLIP models  
            CLIPProcessor, CLIPModel,
            # Other models
            AutoProcessor, AutoModel, AutoTokenizer
        )
        AVAILABILITY["transformers"] = True
        AVAILABILITY["blip2"] = True
        AVAILABILITY["clip"] = True
        print("✅ PyTorch + Transformers + BLIP-2 + CLIP available")
    except ImportError as e:
        print(f"⚠️ Some transformers models not available: {e}")
        
except ImportError:
    print("❌ PyTorch not available")

# TensorFlow (optional)
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    AVAILABILITY["tensorflow"] = True
    print("✅ TensorFlow available")
except ImportError:
    print("⚠️ TensorFlow not available")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelConfig:
    """Unified model configuration"""
    
    def __init__(self, name: str, model_type: ModelType, backend: ModelBackend,
                 model_path: str, description: str = "", memory_gb: float = 2.0,
                 capabilities: List[str] = None, multilingual: bool = False):
        self.name = name
        self.model_type = model_type
        self.backend = backend
        self.model_path = model_path
        self.description = description
        self.memory_gb = memory_gb
        self.capabilities = capabilities or [model_type.value]
        self.multilingual = multilingual
        
        # Runtime properties
        self.loaded = False
        self.model = None
        self.processor = None
        self.device = None
        self.load_time = None

class UnifiedModelManager:
    """Unified manager for all AI models - replaces 3 separate managers"""
    
    def __init__(self, device: str = "auto"):
        print("🚀 Initializing Unified Model Manager...")
        
        # Device setup
        self.device = self._setup_device(device)
        
        # Model storage
        self.available_models: Dict[str, ModelConfig] = {}
        self.loaded_models: Dict[str, ModelConfig] = {}
        self.active_models: Dict[str, str] = {
            "primary_vision_language": None,
            "fallback_vision_language": None,
            "text_embedding": None,
            "image_captioning": None
        }
        
        # Performance settings
        self.batch_size = 16 if self.device == "cuda" else 4
        self.use_mixed_precision = self.device == "cuda"
        
        # Setup model configurations
        self._setup_model_configs()
        
        print(f"✅ Unified Manager ready")
        print(f"📱 Device: {self.device}")
        print(f"🎯 Available models: {len(self.available_models)}")
    
    def _setup_device(self, device: str) -> str:
        """Setup optimal device"""
        if device == "auto":
            if AVAILABILITY["pytorch"] and torch.cuda.is_available():
                try:
                    # Test GPU
                    test_tensor = torch.randn(100, 100).cuda()
                    result = torch.mm(test_tensor, test_tensor.T)
                    del test_tensor, result
                    torch.cuda.empty_cache()
                    
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    
                    print(f"🎮 GPU: {gpu_name}")
                    print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
                    
                    return "cuda"
                except Exception as e:
                    print(f"⚠️ GPU test failed: {e}")
                    return "cpu"
            else:
                print("💻 Using CPU device")
                return "cpu"
        return device
    
    def _setup_model_configs(self):
        """Setup all available model configurations"""
        
        # BLIP-2 Models (Primary - best for complex queries)
        if AVAILABILITY["blip2"]:
            self.available_models["blip2_opt_2.7b"] = ModelConfig(
                name="BLIP-2 OPT 2.7B",
                model_type=ModelType.VISION_LANGUAGE,
                backend=ModelBackend.PYTORCH,
                model_path="Salesforce/blip2-opt-2.7b",
                description="Advanced vision-language model with reasoning",
                memory_gb=6.0,
                capabilities=["vision_language", "image_captioning", "visual_qa"],
                multilingual=True
            )
            
            self.available_models["blip2_t5_xl"] = ModelConfig(
                name="BLIP-2 T5-XL",
                model_type=ModelType.VISION_LANGUAGE,
                backend=ModelBackend.PYTORCH,
                model_path="Salesforce/blip2-t5-xl", 
                description="BLIP-2 with T5 for better text generation",
                memory_gb=8.0,
                capabilities=["vision_language", "text_generation", "reasoning"],
                multilingual=True
            )
        
        # CLIP Models (Fallback - reliable and fast)
        if AVAILABILITY["clip"]:
            self.available_models["clip_vit_base"] = ModelConfig(
                name="CLIP ViT Base",
                model_type=ModelType.VISION_LANGUAGE,
                backend=ModelBackend.PYTORCH,
                model_path="openai/clip-vit-base-patch32",
                description="OpenAI CLIP - reliable vision-language model",
                memory_gb=2.0,
                capabilities=["vision_language", "text_embedding"],
                multilingual=False
            )
            
            self.available_models["clip_vit_large"] = ModelConfig(
                name="CLIP ViT Large", 
                model_type=ModelType.VISION_LANGUAGE,
                backend=ModelBackend.PYTORCH,
                model_path="openai/clip-vit-large-patch14",
                description="CLIP Large - higher accuracy",
                memory_gb=4.0,
                capabilities=["vision_language", "text_embedding"],
                multilingual=False
            )
        
        # Specialized models
        if AVAILABILITY["transformers"]:
            # Image captioning
            self.available_models["blip_captioning"] = ModelConfig(
                name="BLIP Image Captioning",
                model_type=ModelType.IMAGE_CAPTIONING,
                backend=ModelBackend.PYTORCH,
                model_path="Salesforce/blip-image-captioning-base",
                description="Specialized image captioning model",
                memory_gb=3.0,
                capabilities=["image_captioning"],
                multilingual=False
            )
        
        # TensorFlow models (if available)
        if AVAILABILITY["tensorflow"]:
            self.available_models["universal_sentence_encoder"] = ModelConfig(
                name="Universal Sentence Encoder",
                model_type=ModelType.TEXT_EMBEDDING,
                backend=ModelBackend.TENSORFLOW,
                model_path="https://tfhub.dev/google/universal-sentence-encoder/4",
                description="Google text embedding model",
                memory_gb=1.0,
                capabilities=["text_embedding"],
                multilingual=True
            )
    
    def get_recommended_models(self, use_case: str = "general") -> Dict[str, str]:
        """Get recommended model configuration for use case"""
        recommendations = {
            "high_performance": {
                "primary_vision_language": "blip2_opt_2.7b",
                "fallback_vision_language": "clip_vit_base",
                "text_embedding": "universal_sentence_encoder",
                "image_captioning": "blip_captioning"
            },
            "memory_efficient": {
                "primary_vision_language": "clip_vit_base",
                "fallback_vision_language": "clip_vit_base", 
                "text_embedding": "clip_vit_base",
                "image_captioning": "blip_captioning"
            },
            "general": {
                "primary_vision_language": "blip2_opt_2.7b" if AVAILABILITY["blip2"] else "clip_vit_base",
                "fallback_vision_language": "clip_vit_base",
                "text_embedding": "clip_vit_base",
                "image_captioning": "blip_captioning"
            }
        }
        
        return recommendations.get(use_case, recommendations["general"])
    
    def load_model(self, model_key: str) -> bool:
        """Load a specific model"""
        if model_key not in self.available_models:
            print(f"❌ Model {model_key} not available")
            return False
        
        if model_key in self.loaded_models:
            print(f"✅ Model {model_key} already loaded")
            return True
        
        config = self.available_models[model_key]
        
        # Check memory requirements
        if self.device == "cuda":
            available_memory = self._get_available_gpu_memory()
            if config.memory_gb > available_memory:
                print(f"⚠️ Model {model_key} requires {config.memory_gb}GB, only {available_memory:.1f}GB available")
                if config.memory_gb > available_memory * 1.5:  # Hard limit
                    print(f"❌ Not enough memory for {model_key}")
                    return False
        
        try:
            start_time = datetime.now()
            print(f"🔄 Loading {config.name}...")
            
            if config.backend == ModelBackend.PYTORCH:
                success = self._load_pytorch_model(model_key, config)
            elif config.backend == ModelBackend.TENSORFLOW:
                success = self._load_tensorflow_model(model_key, config)
            else:
                print(f"❌ Unsupported backend: {config.backend}")
                return False
            
            if success:
                config.loaded = True
                config.device = self.device
                config.load_time = (datetime.now() - start_time).total_seconds()
                self.loaded_models[model_key] = config
                print(f"✅ {config.name} loaded in {config.load_time:.1f}s")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Failed to load {model_key}: {e}")
            return False
    
    def _load_pytorch_model(self, model_key: str, config: ModelConfig) -> bool:
        """Load PyTorch-based model"""
        try:
            if "blip2" in model_key:
                # BLIP-2 models
                config.processor = Blip2Processor.from_pretrained(config.model_path)
                config.model = Blip2ForConditionalGeneration.from_pretrained(
                    config.model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" and self.use_mixed_precision else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
            
            elif "clip" in model_key:
                # CLIP models
                config.processor = CLIPProcessor.from_pretrained(config.model_path)
                config.model = CLIPModel.from_pretrained(config.model_path)
            
            elif "blip" in model_key and "captioning" in model_key:
                # BLIP captioning
                from transformers import BlipProcessor, BlipForConditionalGeneration
                config.processor = BlipProcessor.from_pretrained(config.model_path)
                config.model = BlipForConditionalGeneration.from_pretrained(config.model_path)
            
            else:
                # Generic transformers model
                config.processor = AutoProcessor.from_pretrained(config.model_path)
                config.model = AutoModel.from_pretrained(config.model_path)
            
            # Move to device
            if self.device == "cuda" and hasattr(config.model, 'to'):
                config.model = config.model.to(self.device)
            
            return True
            
        except Exception as e:
            print(f"❌ PyTorch model loading failed: {e}")
            return False
    
    def _load_tensorflow_model(self, model_key: str, config: ModelConfig) -> bool:
        """Load TensorFlow-based model"""
        try:
            if AVAILABILITY["tensorflow"]:
                import tensorflow_hub as hub
                config.model = hub.load(config.model_path)
                config.processor = None  # TensorFlow Hub models often don't need separate processors
                return True
            else:
                print("❌ TensorFlow not available")
                return False
        except Exception as e:
            print(f"❌ TensorFlow model loading failed: {e}")
            return False
    
    def set_active_model(self, role: str, model_key: str) -> bool:
        """Set a model as active for specific role"""
        if role not in self.active_models:
            print(f"❌ Unknown role: {role}")
            return False
        
        if not self.load_model(model_key):
            return False
        
        old_model = self.active_models[role]
        self.active_models[role] = model_key
        
        print(f"✅ Set {role} to {self.loaded_models[model_key].name}")
        
        # Optionally unload old model to save memory
        if old_model and old_model != model_key and self._should_unload_model(old_model):
            self.unload_model(old_model)
        
        return True
    
    def _should_unload_model(self, model_key: str) -> bool:
        """Check if model should be unloaded to save memory"""
        if self.device == "cpu":
            return False  # CPU memory is more abundant
        
        # Check if model is used in other roles
        for role, active_model in self.active_models.items():
            if active_model == model_key:
                return False  # Still in use
        
        return True
    
    def unload_model(self, model_key: str) -> bool:
        """Unload a model to free memory"""
        if model_key not in self.loaded_models:
            return True
        
        try:
            config = self.loaded_models[model_key]
            
            # Clear GPU memory
            if hasattr(config.model, 'cpu'):
                config.model.cpu()
            del config.model
            del config.processor
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            config.loaded = False
            config.model = None
            config.processor = None
            
            del self.loaded_models[model_key]
            
            print(f"🗑️ Unloaded {config.name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to unload {model_key}: {e}")
            return False
    
    def get_active_models(self) -> Dict[str, Optional[str]]:
        """Get currently active models"""
        return {
            role: model_key for role, model_key in self.active_models.items()
            if model_key and model_key in self.loaded_models
        }
    
    def _get_available_gpu_memory(self) -> float:
        """Get available GPU memory in GB"""
        if self.device == "cuda" and AVAILABILITY["pytorch"]:
            try:
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                return memory_total - memory_allocated
            except:
                return 4.0
        return 0.0
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        gpu_memory = 0
        if self.device == "cuda" and AVAILABILITY["pytorch"]:
            gpu_memory = torch.cuda.memory_allocated(0) / 1024**3
        
        return {
            "device": self.device,
            "gpu_memory_used_gb": gpu_memory,
            "available_models": len(self.available_models),
            "loaded_models": len(self.loaded_models),
            "active_models": self.get_active_models(),
            "availability": AVAILABILITY,
            "batch_size": self.batch_size,
            "mixed_precision": self.use_mixed_precision,
            "models_info": {
                key: {
                    "name": config.name,
                    "type": config.model_type.value,
                    "backend": config.backend.value,
                    "loaded": config.loaded,
                    "memory_gb": config.memory_gb,
                    "multilingual": config.multilingual
                }
                for key, config in self.available_models.items()
            }
        }
    
    def auto_setup(self, use_case: str = "general") -> bool:
        """Automatically setup recommended models for use case"""
        print(f"🤖 Auto-setting up models for '{use_case}' use case...")
        
        recommendations = self.get_recommended_models(use_case)
        success_count = 0
        
        for role, model_key in recommendations.items():
            if model_key and model_key in self.available_models:
                if self.set_active_model(role, model_key):
                    success_count += 1
                else:
                    print(f"⚠️ Failed to setup {role}: {model_key}")
            else:
                print(f"⚠️ Recommended model not available for {role}: {model_key}")
        
        print(f"✅ Setup complete: {success_count}/{len(recommendations)} models loaded")
        return success_count > 0
    
    def benchmark_models(self, test_text: str = "a person walking") -> Dict[str, Dict[str, float]]:
        """Benchmark loaded vision-language models"""
        results = {}
        
        for model_key, config in self.loaded_models.items():
            if ModelType.VISION_LANGUAGE in [config.model_type] or "vision_language" in config.capabilities:
                try:
                    start_time = time.time()
                    
                    # Test text encoding
                    if hasattr(config.processor, 'tokenizer') or "clip" in model_key:
                        if "clip" in model_key:
                            inputs = config.processor(text=[test_text], return_tensors="pt", padding=True)
                            if self.device == "cuda":
                                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                            with torch.no_grad():
                                text_features = config.model.get_text_features(**inputs)
                        else:
                            # BLIP-2 or other models
                            inputs = config.processor(text=test_text, return_tensors="pt")
                            if self.device == "cuda":
                                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                            with torch.no_grad():
                                outputs = config.model.generate(**inputs, max_length=20)
                    
                    encoding_time = time.time() - start_time
                    
                    results[model_key] = {
                        "encoding_time": encoding_time,
                        "memory_gb": config.memory_gb,
                        "status": "success"
                    }
                    
                except Exception as e:
                    results[model_key] = {
                        "encoding_time": -1,
                        "memory_gb": config.memory_gb,
                        "status": f"failed: {e}"
                    }
        
        return results

def demo_unified_manager():
    """Demo function for unified model manager"""
    print("🚀 Unified Model Manager Demo")
    print("=" * 50)
    
    # Initialize manager
    manager = UnifiedModelManager()
    
    # Show system info
    info = manager.get_system_info()
    print("\n📊 System Information:")
    for key, value in info.items():
        if key != "models_info":  # Skip detailed model info for brevity
            print(f"   {key}: {value}")
    
    # Show available models
    print(f"\n📋 Available Models ({len(manager.available_models)}):")
    for key, config in manager.available_models.items():
        status = "✅ loaded" if config.loaded else "⭕ available"
        multilingual = "🌍" if config.multilingual else "🇺🇸"
        print(f"   {status} {multilingual} {key}: {config.name} ({config.memory_gb}GB)")
    
    # Auto-setup
    print("\n🤖 Auto-setting up models...")
    manager.auto_setup("general")
    
    # Show active models
    active = manager.get_active_models()
    print(f"\n🎯 Active Models ({len(active)}):")
    for role, model_key in active.items():
        if model_key:
            config = manager.loaded_models[model_key]
            print(f"   {role}: {config.name}")
    
    # Benchmark if models loaded
    if active:
        print("\n⚡ Benchmarking models...")
        benchmark = manager.benchmark_models()
        for model_key, results in benchmark.items():
            status = results["status"]
            time_str = f"{results['encoding_time']:.3f}s" if results['encoding_time'] > 0 else "failed"
            print(f"   {model_key}: {time_str} ({status})")
    
    print("\n✅ Demo completed!")

if __name__ == "__main__":
    demo_unified_manager()
