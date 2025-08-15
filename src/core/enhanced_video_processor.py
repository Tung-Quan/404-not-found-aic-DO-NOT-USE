"""
🎥 ENHANCED VIDEO PROCESSING WITH TENSORFLOW HUB
===============================================
Advanced video processing với multiple TensorFlow Hub models
Tích hợp intelligent model selection và video analysis
"""

import os
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import json
import logging
from dataclasses import dataclass
from enum import Enum

# Try to import TensorFlow Hub models
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow_text
    import cv2
    TF_AVAILABLE = True
    print("✅ TensorFlow Hub available for video processing")
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow Hub not available")

class ModelType(Enum):
    """Available model types"""
    TEXT_ENCODER = "text_encoder"
    VIDEO_CLASSIFIER = "video_classifier"
    ACTION_RECOGNITION = "action_recognition"
    OBJECT_DETECTION = "object_detection"
    SCENE_UNDERSTANDING = "scene_understanding"

@dataclass
class ModelInfo:
    """Model information structure"""
    name: str
    url: str
    model_type: ModelType
    description: str
    input_shape: tuple
    output_shape: tuple
    memory_usage: str
    processing_speed: str
    overlap_models: List[str] = None
    complementary_models: List[str] = None

class TensorFlowHubVideoManager:
    """
    Advanced TensorFlow Hub model manager for video processing
    """
    
    def __init__(self):
        self.models = {}
        self.model_configs = self._load_model_configurations()
        self.active_models = set()
        self.user_preferences = {}
        self.performance_cache = {}
        
    def _load_model_configurations(self) -> Dict[str, ModelInfo]:
        """Load all available TensorFlow Hub model configurations"""
        return {
            # Text and Language Models
            "use_multilingual": ModelInfo(
                name="Universal Sentence Encoder Multilingual",
                url="https://tfhub.dev/google/universal-sentence-encoder-multilingual/3",
                model_type=ModelType.TEXT_ENCODER,
                description="Multilingual text encoding supporting Vietnamese",
                input_shape=("variable_text",),
                output_shape=(512,),
                memory_usage="~500MB",
                processing_speed="Fast",
                overlap_models=["use_v4", "chinese_clip"],
                complementary_models=["efficientnet", "movinet"]
            ),
            
            "use_v4": ModelInfo(
                name="Universal Sentence Encoder v4",
                url="https://tfhub.dev/google/universal-sentence-encoder/4",
                model_type=ModelType.TEXT_ENCODER,
                description="Latest English text encoder",
                input_shape=("variable_text",),
                output_shape=(512,),
                memory_usage="~300MB",
                processing_speed="Very Fast",
                overlap_models=["use_multilingual"],
                complementary_models=["efficientnet", "movinet"]
            ),
            
            # Video and Action Recognition Models
            "movinet_a0": ModelInfo(
                name="MoViNet A0 Action Recognition",
                url="https://tfhub.dev/tensorflow/movinet/a0/base/kinetics-600/classification/3",
                model_type=ModelType.ACTION_RECOGNITION,
                description="Lightweight action recognition in videos",
                input_shape=(None, None, None, 3),
                output_shape=(600,),
                memory_usage="~100MB",
                processing_speed="Fast",
                overlap_models=["movinet_a2"],
                complementary_models=["use_multilingual", "efficientnet"]
            ),
            
            "movinet_a2": ModelInfo(
                name="MoViNet A2 Action Recognition",
                url="https://tfhub.dev/tensorflow/movinet/a2/base/kinetics-600/classification/3",
                model_type=ModelType.ACTION_RECOGNITION,
                description="High-accuracy action recognition",
                input_shape=(None, None, None, 3),
                output_shape=(600,),
                memory_usage="~300MB",
                processing_speed="Medium",
                overlap_models=["movinet_a0"],
                complementary_models=["use_multilingual", "efficientnet"]
            ),
            
            # Visual Feature Models
            "efficientnet_v2_b0": ModelInfo(
                name="EfficientNet V2 B0",
                url="https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2",
                model_type=ModelType.SCENE_UNDERSTANDING,
                description="Lightweight visual features",
                input_shape=(224, 224, 3),
                output_shape=(1280,),
                memory_usage="~50MB",
                processing_speed="Very Fast",
                overlap_models=["efficientnet_v2_b3", "chinese_clip"],
                complementary_models=["use_multilingual", "movinet_a0"]
            ),
            
            "efficientnet_v2_b3": ModelInfo(
                name="EfficientNet V2 B3",
                url="https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b3/feature_vector/2",
                model_type=ModelType.SCENE_UNDERSTANDING,
                description="High-quality visual features",
                input_shape=(300, 300, 3),
                output_shape=(1536,),
                memory_usage="~200MB",
                processing_speed="Medium",
                overlap_models=["efficientnet_v2_b0"],
                complementary_models=["use_multilingual", "movinet_a2"]
            ),
            
            # Object Detection Models
            "ssd_mobilenet": ModelInfo(
                name="SSD MobileNet Object Detection",
                url="https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2",
                model_type=ModelType.OBJECT_DETECTION,
                description="Real-time object detection",
                input_shape=(None, None, 3),
                output_shape=("variable",),
                memory_usage="~100MB",
                processing_speed="Fast",
                overlap_models=["faster_rcnn"],
                complementary_models=["use_multilingual", "efficientnet_v2_b0"]
            ),
            
            "faster_rcnn": ModelInfo(
                name="Faster R-CNN Object Detection",
                url="https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1",
                model_type=ModelType.OBJECT_DETECTION,
                description="High-accuracy object detection",
                input_shape=(640, 640, 3),
                output_shape=("variable",),
                memory_usage="~500MB",
                processing_speed="Slow",
                overlap_models=["ssd_mobilenet"],
                complementary_models=["use_multilingual", "efficientnet_v2_b3"]
            )
        }
    
    def detect_overlapping_models(self, requested_models: List[str]) -> Dict[str, List[str]]:
        """Detect overlapping functionality between requested models"""
        overlaps = {}
        
        for model1 in requested_models:
            if model1 not in self.model_configs:
                continue
                
            overlapping = []
            for model2 in requested_models:
                if model1 != model2 and model2 in self.model_configs:
                    config1 = self.model_configs[model1]
                    config2 = self.model_configs[model2]
                    
                    # Check if models have overlapping functionality
                    if (config1.model_type == config2.model_type or
                        model2 in (config1.overlap_models or []) or
                        model1 in (config2.overlap_models or [])):
                        overlapping.append(model2)
            
            if overlapping:
                overlaps[model1] = overlapping
        
        return overlaps
    
    def suggest_model_combinations(self, user_intent: str, max_memory_mb: int = 2000) -> Dict[str, any]:
        """Suggest optimal model combinations based on user intent and constraints"""
        
        # Analyze user intent
        intent_keywords = {
            'text': ['text', 'query', 'search', 'language', 'vietnamese', 'english'],
            'action': ['action', 'movement', 'activity', 'behavior', 'motion'],
            'objects': ['object', 'thing', 'item', 'detection', 'recognize'],
            'scene': ['scene', 'visual', 'image', 'frame', 'appearance'],
            'video': ['video', 'temporal', 'sequence', 'timeline']
        }
        
        intent_scores = {}
        user_lower = user_intent.lower()
        
        for category, keywords in intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in user_lower)
            intent_scores[category] = score
        
        # Suggest models based on intent
        recommendations = {
            'lightweight': [],
            'balanced': [],
            'high_accuracy': [],
            'overlaps_detected': {},
            'complementary_pairs': []
        }
        
        # Base recommendations
        if intent_scores['text'] > 0:
            recommendations['lightweight'].append('use_v4')
            recommendations['balanced'].append('use_multilingual')
            recommendations['high_accuracy'].append('use_multilingual')
        
        if intent_scores['action'] > 0:
            recommendations['lightweight'].append('movinet_a0')
            recommendations['balanced'].append('movinet_a0')
            recommendations['high_accuracy'].append('movinet_a2')
        
        if intent_scores['scene'] > 0 or intent_scores['video'] > 0:
            recommendations['lightweight'].append('efficientnet_v2_b0')
            recommendations['balanced'].append('efficientnet_v2_b0')
            recommendations['high_accuracy'].append('efficientnet_v2_b3')
        
        if intent_scores['objects'] > 0:
            recommendations['lightweight'].append('ssd_mobilenet')
            recommendations['balanced'].append('ssd_mobilenet')
            recommendations['high_accuracy'].append('faster_rcnn')
        
        # Check for overlaps and suggest alternatives
        for config_type in ['lightweight', 'balanced', 'high_accuracy']:
            models = recommendations[config_type]
            overlaps = self.detect_overlapping_models(models)
            
            if overlaps:
                recommendations['overlaps_detected'][config_type] = overlaps
                
                # Suggest alternatives for overlapping models
                for model, overlapping_with in overlaps.items():
                    print(f"⚠️  Model overlap detected: {model} overlaps with {overlapping_with}")
        
        return recommendations
    
    def create_interactive_model_selector(self, recommendations: Dict) -> List[str]:
        """Create interactive selection interface for overlapping models"""
        selected_models = []
        
        print("\n🤖 INTELLIGENT MODEL SELECTION")
        print("=" * 50)
        
        # Show recommendations
        for config_type, models in recommendations.items():
            if config_type not in ['overlaps_detected', 'complementary_pairs'] and models:
                print(f"\n📊 {config_type.upper()} Configuration:")
                for i, model in enumerate(models, 1):
                    if model in self.model_configs:
                        config = self.model_configs[model]
                        print(f"   {i}. {config.name}")
                        print(f"      Memory: {config.memory_usage} | Speed: {config.processing_speed}")
                        print(f"      {config.description}")
        
        # Handle overlaps
        if 'overlaps_detected' in recommendations and recommendations['overlaps_detected']:
            print("\n⚠️  OVERLAPPING FUNCTIONALITY DETECTED")
            print("Please choose between overlapping models:")
            
            for config_type, overlaps in recommendations['overlaps_detected'].items():
                print(f"\n🔧 {config_type.upper()} Configuration Conflicts:")
                
                for model, overlapping_with in overlaps.items():
                    print(f"\n   Model: {self.model_configs[model].name}")
                    print(f"   Overlaps with: {[self.model_configs[m].name for m in overlapping_with]}")
                    
                    choice = input(f"   Use {model}? (y/n/skip): ").lower().strip()
                    if choice == 'y':
                        selected_models.append(model)
                        # Remove overlapping models from consideration
                        for overlap_model in overlapping_with:
                            if overlap_model in selected_models:
                                selected_models.remove(overlap_model)
                    elif choice == 'skip':
                        continue
        else:
            # No overlaps, ask user to select configuration
            print(f"\n🎯 Select Configuration:")
            print("   1. Lightweight (Fast, Low Memory)")
            print("   2. Balanced (Good Performance/Memory)")  
            print("   3. High Accuracy (Best Quality, High Memory)")
            
            choice = input("Enter choice (1-3): ").strip()
            config_map = {'1': 'lightweight', '2': 'balanced', '3': 'high_accuracy'}
            
            if choice in config_map:
                selected_models = recommendations[config_map[choice]]
        
        return selected_models
    
    def load_selected_models(self, model_names: List[str]) -> Dict[str, bool]:
        """Load selected models with progress tracking"""
        results = {}
        
        print(f"\n🔄 Loading {len(model_names)} selected models...")
        
        for i, model_name in enumerate(model_names, 1):
            if model_name not in self.model_configs:
                print(f"❌ Unknown model: {model_name}")
                results[model_name] = False
                continue
            
            config = self.model_configs[model_name]
            print(f"\n[{i}/{len(model_names)}] Loading {config.name}...")
            print(f"   URL: {config.url}")
            print(f"   Expected memory: {config.memory_usage}")
            
            try:
                start_time = time.time()
                model = hub.load(config.url)
                load_time = time.time() - start_time
                
                self.models[model_name] = model
                self.active_models.add(model_name)
                results[model_name] = True
                
                print(f"   ✅ Loaded in {load_time:.1f}s")
                
            except Exception as e:
                print(f"   ❌ Failed to load: {e}")
                results[model_name] = False
        
        return results
    
    def get_model_status(self) -> Dict[str, any]:
        """Get status of all models"""
        status = {
            'tensorflow_hub_available': TF_AVAILABLE,
            'total_models_available': len(self.model_configs),
            'active_models': list(self.active_models),
            'model_details': {}
        }
        
        for model_name, is_loaded in [(name, name in self.active_models) for name in self.model_configs.keys()]:
            config = self.model_configs[model_name]
            status['model_details'][model_name] = {
                'loaded': is_loaded,
                'name': config.name,
                'type': config.model_type.value,
                'memory_usage': config.memory_usage,
                'processing_speed': config.processing_speed,
                'description': config.description
            }
        
        return status
    
    def process_video_with_selected_models(self, video_path: str, query: str = "") -> Dict[str, any]:
        """Process video using selected models"""
        if not self.active_models:
            return {'error': 'No models loaded'}
        
        results = {
            'video_path': video_path,
            'query': query,
            'processing_results': {},
            'combined_features': {},
            'processing_time': {}
        }
        
        print(f"\n🎥 Processing video: {os.path.basename(video_path)}")
        print(f"📝 Query: {query}")
        print(f"🔧 Active models: {list(self.active_models)}")
        
        # Process with each active model
        for model_name in self.active_models:
            if model_name not in self.models:
                continue
                
            config = self.model_configs[model_name]
            model = self.models[model_name]
            
            print(f"\n🔄 Processing with {config.name}...")
            start_time = time.time()
            
            try:
                if config.model_type == ModelType.TEXT_ENCODER and query:
                    # Process text query
                    embeddings = model([query])
                    results['processing_results'][model_name] = {
                        'type': 'text_embedding',
                        'shape': embeddings.shape,
                        'embedding': embeddings.numpy().tolist()
                    }
                    
                elif config.model_type in [ModelType.ACTION_RECOGNITION, ModelType.SCENE_UNDERSTANDING]:
                    # Process video frames
                    frame_results = self._process_video_frames(video_path, model, config)
                    results['processing_results'][model_name] = frame_results
                    
                elif config.model_type == ModelType.OBJECT_DETECTION:
                    # Process for object detection
                    detection_results = self._process_video_detection(video_path, model, config)
                    results['processing_results'][model_name] = detection_results
                
                processing_time = time.time() - start_time
                results['processing_time'][model_name] = processing_time
                print(f"   ✅ Completed in {processing_time:.1f}s")
                
            except Exception as e:
                print(f"   ❌ Processing failed: {e}")
                results['processing_results'][model_name] = {'error': str(e)}
        
        return results
    
    def _process_video_frames(self, video_path: str, model, config: ModelInfo, max_frames: int = 10):
        """Process video frames with visual models"""
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames evenly
            frame_indices = np.linspace(0, frame_count-1, min(max_frames, frame_count), dtype=int)
            
            frame_features = []
            processed_frames = []
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Preprocess frame according to model requirements
                    processed_frame = self._preprocess_frame(frame, config)
                    
                    # Get features
                    features = model(processed_frame)
                    frame_features.append(features.numpy())
                    processed_frames.append(frame_idx)
            
            cap.release()
            
            return {
                'type': 'video_features',
                'processed_frames': processed_frames,
                'features_shape': [f.shape for f in frame_features],
                'avg_features': np.mean(frame_features, axis=0).tolist() if frame_features else []
            }
            
        except Exception as e:
            return {'error': f'Video processing failed: {e}'}
    
    def _process_video_detection(self, video_path: str, model, config: ModelInfo, max_frames: int = 5):
        """Process video for object detection"""
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample fewer frames for detection (more computationally expensive)
            frame_indices = np.linspace(0, frame_count-1, min(max_frames, frame_count), dtype=int)
            
            all_detections = []
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Preprocess frame for detection
                    processed_frame = self._preprocess_frame(frame, config)
                    
                    # Run detection
                    detections = model(processed_frame)
                    
                    # Process detection results (simplified)
                    frame_detections = {
                        'frame_idx': frame_idx,
                        'num_detections': len(detections) if hasattr(detections, '__len__') else 'unknown'
                    }
                    all_detections.append(frame_detections)
            
            cap.release()
            
            return {
                'type': 'object_detection',
                'detections': all_detections,
                'total_frames_processed': len(frame_indices)
            }
            
        except Exception as e:
            return {'error': f'Detection processing failed: {e}'}
    
    def _preprocess_frame(self, frame, config: ModelInfo):
        """Preprocess frame according to model requirements"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize according to model input requirements
        if hasattr(config.input_shape, '__len__') and len(config.input_shape) >= 2:
            if config.input_shape[0] and config.input_shape[1]:
                height, width = config.input_shape[:2]
                frame_rgb = cv2.resize(frame_rgb, (width, height))
        
        # Convert to tensor and normalize
        frame_tensor = tf.convert_to_tensor(frame_rgb, dtype=tf.float32)
        frame_tensor = tf.cast(frame_tensor, tf.float32) / 255.0
        frame_tensor = tf.expand_dims(frame_tensor, 0)  # Add batch dimension
        
        return frame_tensor

def main():
    """Main function for interactive model selection and video processing"""
    print("🎥 ENHANCED TENSORFLOW HUB VIDEO PROCESSING")
    print("=" * 60)
    
    if not TF_AVAILABLE:
        print("❌ TensorFlow Hub not available. Please install:")
        print("   pip install tensorflow tensorflow-hub tensorflow-text opencv-python")
        return
    
    manager = TensorFlowHubVideoManager()
    
    # Get user intent
    user_intent = input("\n📝 Describe what you want to do with videos: ")
    
    # Get memory constraints
    try:
        max_memory = int(input("💾 Maximum memory usage (MB, default 2000): ") or "2000")
    except ValueError:
        max_memory = 2000
    
    # Generate recommendations
    recommendations = manager.suggest_model_combinations(user_intent, max_memory)
    
    # Interactive model selection
    selected_models = manager.create_interactive_model_selector(recommendations)
    
    if not selected_models:
        print("❌ No models selected. Exiting.")
        return
    
    # Load selected models
    load_results = manager.load_selected_models(selected_models)
    
    # Show status
    print("\n📊 MODEL STATUS:")
    status = manager.get_model_status()
    for model_name, details in status['model_details'].items():
        if details['loaded']:
            print(f"   ✅ {details['name']} - {details['type']}")
    
    # Optional video processing demo
    demo_choice = input("\n🎬 Process a demo video? (y/n): ").lower().strip()
    if demo_choice == 'y':
        video_path = input("📁 Enter video path (or press Enter for default): ").strip()
        if not video_path:
            video_path = "videos/sample.mp4"  # Default path
        
        query = input("🔍 Enter search query (optional): ").strip()
        
        if os.path.exists(video_path):
            results = manager.process_video_with_selected_models(video_path, query)
            print(f"\n🎉 Processing completed! Results: {len(results['processing_results'])} models")
        else:
            print(f"❌ Video file not found: {video_path}")

if __name__ == "__main__":
    main()
