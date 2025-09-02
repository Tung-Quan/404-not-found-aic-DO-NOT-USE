"""
TensorFlow Models Manager - Tích hợp đầy đủ TensorFlow Hub models
Hỗ trợ các models từ TensorFlow Hub cho computer vision và NLP
"""
import os
import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from pathlib import Path
import json

# TensorFlow imports
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    TENSORFLOW_AVAILABLE = True
    
    # Configure TensorFlow
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # GPU configuration
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"🎮 TensorFlow GPU enabled: {len(physical_devices)} device(s)")
    else:
        print("💻 TensorFlow using CPU")
        
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available")

# Additional dependencies
try:
    from PIL import Image
    import cv2
    IMAGE_PROCESSING_AVAILABLE = True
except ImportError:
    IMAGE_PROCESSING_AVAILABLE = False

class TensorFlowModelConfig:
    """Configuration cho TensorFlow model"""
    def __init__(self, name: str, url: str, model_type: str, 
                 description: str = "", input_shape: Optional[Tuple] = None,
                 preprocessing: Optional[str] = None, capabilities: List[str] = None):
        self.name = name
        self.url = url
        self.model_type = model_type  # 'image_embedding', 'text_embedding', 'classification', 'detection'
        self.description = description
        self.input_shape = input_shape
        self.preprocessing = preprocessing
        self.capabilities = capabilities or []

class TensorFlowModelManager:
    """Manager cho TensorFlow Hub models"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models: Dict[str, Any] = {}
        self.configs: Dict[str, TensorFlowModelConfig] = {}
        self.device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
        
        # Setup default models
        self._setup_tensorflow_models()
        
        print(f"🚀 TensorFlow Model Manager initialized")
        print(f"💾 Using device: {self.device}")
    
    def _setup_tensorflow_models(self):
        """Setup TensorFlow Hub models"""
        
        # Image Feature Extraction Models
        self.configs["mobilenet_v2"] = TensorFlowModelConfig(
            name="MobileNet V2",
            url="https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
            model_type="image_embedding",
            description="Lightweight image feature extraction",
            input_shape=(224, 224, 3),
            preprocessing="mobilenet",
            capabilities=["image_embedding", "feature_extraction"]
        )
        
        self.configs["inception_v3"] = TensorFlowModelConfig(
            name="Inception V3",
            url="https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4",
            model_type="image_embedding", 
            description="High-quality image feature extraction",
            input_shape=(299, 299, 3),
            preprocessing="inception",
            capabilities=["image_embedding", "feature_extraction"]
        )
        
        self.configs["resnet50"] = TensorFlowModelConfig(
            name="ResNet-50",
            url="https://tfhub.dev/tensorflow/resnet_50/feature_vector/1",
            model_type="image_embedding",
            description="ResNet-50 for robust image features",
            input_shape=(224, 224, 3),
            preprocessing="imagenet",
            capabilities=["image_embedding", "feature_extraction"]
        )
        
        self.configs["efficientnet_b0"] = TensorFlowModelConfig(
            name="EfficientNet B0",
            url="https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1",
            model_type="image_embedding",
            description="Efficient image feature extraction",
            input_shape=(224, 224, 3),
            preprocessing="efficientnet",
            capabilities=["image_embedding", "feature_extraction"]
        )
        
        # Text Embedding Models
        self.configs["universal_sentence_encoder"] = TensorFlowModelConfig(
            name="Universal Sentence Encoder",
            url="https://tfhub.dev/google/universal-sentence-encoder/4",
            model_type="text_embedding",
            description="Universal text embedding for semantic similarity",
            capabilities=["text_embedding", "semantic_search"]
        )
        
        self.configs["use_multilingual"] = TensorFlowModelConfig(
            name="Universal Sentence Encoder Multilingual",
            url="https://tfhub.dev/google/universal-sentence-encoder-multilingual/3",
            model_type="text_embedding",
            description="Multilingual text embeddings",
            capabilities=["text_embedding", "multilingual", "semantic_search"]
        )
        
        self.configs["use_large"] = TensorFlowModelConfig(
            name="Universal Sentence Encoder Large",
            url="https://tfhub.dev/google/universal-sentence-encoder-large/5",
            model_type="text_embedding",
            description="Large text embeddings for high accuracy",
            capabilities=["text_embedding", "semantic_search", "high_quality"]
        )
        
        # Object Detection Models
        self.configs["ssd_mobilenet"] = TensorFlowModelConfig(
            name="SSD MobileNet",
            url="https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2",
            model_type="object_detection",
            description="Fast object detection",
            input_shape=(320, 320, 3),
            capabilities=["object_detection", "real_time"]
        )
        
        self.configs["faster_rcnn"] = TensorFlowModelConfig(
            name="Faster R-CNN",
            url="https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1",
            model_type="object_detection",
            description="High-accuracy object detection",
            input_shape=(640, 640, 3),
            capabilities=["object_detection", "high_accuracy"]
        )
        
        # Image Classification Models
        self.configs["imagenet_mobilenet"] = TensorFlowModelConfig(
            name="ImageNet MobileNet",
            url="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4",
            model_type="classification",
            description="Image classification with ImageNet classes",
            input_shape=(224, 224, 3),
            preprocessing="mobilenet",
            capabilities=["classification", "imagenet"]
        )
        
        # Advanced Vision Models
        self.configs["bit_resnet50"] = TensorFlowModelConfig(
            name="BiT ResNet-50",
            url="https://tfhub.dev/google/bit/s-r50x1/1",
            model_type="image_embedding",
            description="Big Transfer ResNet for fine-tuning",
            input_shape=(224, 224, 3),
            capabilities=["image_embedding", "transfer_learning"]
        )
    
    def load_model(self, model_key: str) -> bool:
        """Load TensorFlow model"""
        if not TENSORFLOW_AVAILABLE:
            self.logger.error("TensorFlow not available")
            return False
        
        if model_key not in self.configs:
            self.logger.error(f"Model {model_key} not configured")
            return False
        
        if model_key in self.models:
            self.logger.info(f"Model {model_key} already loaded")
            return True
        
        config = self.configs[model_key]
        
        try:
            self.logger.info(f"🔄 Loading {config.name} from TensorFlow Hub...")
            
            with tf.device(self.device):
                model = hub.load(config.url)
                self.models[model_key] = model
            
            self.logger.info(f"✅ {config.name} loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load {config.name}: {e}")
            return False
    
    def preprocess_image(self, image_path: str, config: TensorFlowModelConfig) -> Optional[Any]:
        """Preprocess image for TensorFlow model"""
        if not IMAGE_PROCESSING_AVAILABLE or not TENSORFLOW_AVAILABLE:
            return None
        
        try:
            # Load image
            if isinstance(image_path, str):
                image = Image.open(image_path).convert('RGB')
            else:
                image = image_path
            
            # Resize to model input shape
            if config.input_shape:
                target_size = config.input_shape[:2]
                image = image.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            image_array = np.array(image).astype(np.float32)
            
            # Normalize based on preprocessing type
            if config.preprocessing == "mobilenet":
                image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
            elif config.preprocessing == "inception":
                image_array = tf.keras.applications.inception_v3.preprocess_input(image_array)
            elif config.preprocessing == "imagenet":
                image_array = tf.keras.applications.imagenet_utils.preprocess_input(image_array)
            elif config.preprocessing == "efficientnet":
                image_array = tf.keras.applications.efficientnet.preprocess_input(image_array)
            else:
                # Default normalization to [0, 1]
                image_array = image_array / 255.0
            
            # Add batch dimension
            return tf.expand_dims(image_array, 0)
            
        except Exception as e:
            self.logger.error(f"Image preprocessing failed: {e}")
            return None
    
    def extract_image_features(self, image_path: str, model_key: str = "mobilenet_v2") -> Optional[np.ndarray]:
        """Extract features from image"""
        if model_key not in self.models:
            if not self.load_model(model_key):
                return None
        
        config = self.configs[model_key]
        model = self.models[model_key]
        
        # Preprocess image
        processed_image = self.preprocess_image(image_path, config)
        if processed_image is None:
            return None
        
        try:
            with tf.device(self.device):
                features = model(processed_image)
                
            # Convert to numpy
            if isinstance(features, dict):
                # Some models return dict, get the main feature vector
                features = features.get('default', features.get('feature_vector', features))
            
            return features.numpy().flatten()
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return None
    
    def encode_text(self, text: str, model_key: str = "universal_sentence_encoder") -> Optional[np.ndarray]:
        """Encode text to embedding"""
        if model_key not in self.models:
            if not self.load_model(model_key):
                return None
        
        model = self.models[model_key]
        
        try:
            with tf.device(self.device):
                embeddings = model([text])
                
            return embeddings.numpy()[0]
            
        except Exception as e:
            self.logger.error(f"Text encoding failed: {e}")
            return None
    
    def batch_extract_features(self, image_paths: List[str], model_key: str = "mobilenet_v2", 
                             batch_size: int = 32) -> List[Optional[np.ndarray]]:
        """Extract features from multiple images in batches"""
        if model_key not in self.models:
            if not self.load_model(model_key):
                return [None] * len(image_paths)
        
        config = self.configs[model_key]
        model = self.models[model_key]
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            # Preprocess batch
            for path in batch_paths:
                processed = self.preprocess_image(path, config)
                if processed is not None:
                    batch_images.append(processed[0])  # Remove batch dimension
                else:
                    batch_images.append(None)
            
            # Filter out None values for processing
            valid_images = [img for img in batch_images if img is not None]
            valid_indices = [i for i, img in enumerate(batch_images) if img is not None]
            
            if valid_images:
                try:
                    # Stack valid images
                    batch_tensor = tf.stack(valid_images)
                    
                    with tf.device(self.device):
                        batch_features = model(batch_tensor)
                    
                    # Convert to numpy
                    if isinstance(batch_features, dict):
                        batch_features = batch_features.get('default', batch_features.get('feature_vector', batch_features))
                    
                    batch_numpy = batch_features.numpy()
                    
                    # Map back to original positions
                    batch_results = [None] * len(batch_images)
                    for idx, valid_idx in enumerate(valid_indices):
                        batch_results[valid_idx] = batch_numpy[idx]
                    
                    results.extend(batch_results)
                    
                except Exception as e:
                    self.logger.error(f"Batch processing failed: {e}")
                    results.extend([None] * len(batch_images))
            else:
                results.extend([None] * len(batch_images))
        
        return results
    
    def detect_objects(self, image_path: str, model_key: str = "ssd_mobilenet") -> Optional[Dict]:
        """Detect objects in image"""
        if model_key not in self.models:
            if not self.load_model(model_key):
                return None
        
        config = self.configs[model_key]
        model = self.models[model_key]
        
        # Preprocess image
        processed_image = self.preprocess_image(image_path, config)
        if processed_image is None:
            return None
        
        try:
            with tf.device(self.device):
                detections = model(processed_image)
            
            # Process detection results
            result = {}
            if isinstance(detections, dict):
                for key, value in detections.items():
                    result[key] = value.numpy()
            else:
                result['detections'] = detections.numpy()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Object detection failed: {e}")
            return None
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get available TensorFlow models"""
        return {
            key: {
                "name": config.name,
                "type": config.model_type,
                "description": config.description,
                "capabilities": config.capabilities,
                "loaded": key in self.models,
                "url": config.url
            }
            for key, config in self.configs.items()
        }
    
    def get_model_info(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about specific model"""
        if model_key not in self.configs:
            return None
        
        config = self.configs[model_key]
        return {
            "name": config.name,
            "url": config.url,
            "type": config.model_type,
            "description": config.description,
            "input_shape": config.input_shape,
            "preprocessing": config.preprocessing,
            "capabilities": config.capabilities,
            "loaded": model_key in self.models
        }
    
    def unload_model(self, model_key: str) -> bool:
        """Unload model to free memory"""
        if model_key in self.models:
            del self.models[model_key]
            tf.keras.backend.clear_session()  # Clear TensorFlow session
            self.logger.info(f"🗑️ Unloaded model: {model_key}")
            return True
        return False
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get GPU memory usage info"""
        memory_info = {"loaded_models": len(self.models)}
        
        if tf.config.list_physical_devices('GPU'):
            try:
                # Get GPU memory info
                gpu_devices = tf.config.experimental.list_physical_devices('GPU')
                memory_info["gpu_devices"] = len(gpu_devices)
                memory_info["device"] = self.device
            except Exception as e:
                memory_info["error"] = str(e)
        else:
            memory_info["device"] = "CPU"
        
        return memory_info

# Global instance
tensorflow_model_manager = TensorFlowModelManager() if TENSORFLOW_AVAILABLE else None

def test_tensorflow_models():
    """Test TensorFlow models functionality"""
    print("🧪 Testing TensorFlow Model Manager...")
    
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow not available")
        return
    
    manager = TensorFlowModelManager()
    
    # Show available models
    models = manager.get_available_models()
    print(f"📋 Available TensorFlow models: {len(models)}")
    
    for key, info in models.items():
        status = "✅" if info["loaded"] else "⚠️"
        print(f"  {status} {info['name']} ({info['type']}) - {info['capabilities']}")
    
    # Test loading a lightweight model
    print("\n🔄 Testing model loading...")
    success = manager.load_model("mobilenet_v2")
    if success:
        print("✅ Model loading test passed")
        
        # Test memory info
        memory = manager.get_memory_usage()
        print(f"💾 Memory info: {memory}")
    
    print("✅ TensorFlow Model Manager test completed")

if __name__ == "__main__":
    test_tensorflow_models()

tensorflow_model_manager = TensorFlowModelManager()