#!/usr/bin/env python3
"""
AI Search Engine - Lite Version
Phiên bản đơn giản chỉ sử dụng dependencies có sẵn
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time
import numpy as np
from PIL import Image
import cv2

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AISearchEngineLite:
    """
    AI Search Engine phiên bản lite
    Chỉ sử dụng OpenCV, PIL, numpy - không cần GPU hay AI models phức tạp
    """
    
    def __init__(self, frames_dir: str = "./frames"):
        self.frames_dir = Path(frames_dir)
        self.index_file = self.frames_dir / "search_index_lite.json"
        self.search_index = {}
        
        # Ensure frames directory exists
        self.frames_dir.mkdir(exist_ok=True)
        
        # Load existing index
        self._load_index()
        
        logger.info(f"✅ AI Search Engine Lite initialized")
        logger.info(f"📁 Frames directory: {self.frames_dir}")
        logger.info(f"📊 Indexed frames: {len(self.search_index)}")
    
    def _load_index(self):
        """Load search index từ file"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.search_index = json.load(f)
                logger.info(f"📥 Loaded index with {len(self.search_index)} entries")
            except Exception as e:
                logger.warning(f"⚠️ Could not load index: {e}")
                self.search_index = {}
        else:
            self.search_index = {}
    
    def _save_index(self):
        """Save search index to file"""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.search_index, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Saved index with {len(self.search_index)} entries")
        except Exception as e:
            logger.error(f"❌ Could not save index: {e}")
    
    def extract_image_features(self, image_path: str) -> Dict[str, Any]:
        """
        Extract basic image features using OpenCV and PIL
        Không cần AI models phức tạp
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return {}
            
            # Basic features
            height, width, channels = img.shape
            
            # Color histogram
            hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
            hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])
            
            # Average colors
            avg_color = np.mean(img, axis=(0, 1))
            
            # Brightness and contrast
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Edge density (canny edges)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (width * height)
            
            # Dominant colors (simplified k-means)
            data = img.reshape((-1, 3))
            data = np.float32(data)
            
            # Simple dominant color calculation
            dominant_color = np.mean(data, axis=0)
            
            features = {
                "width": int(width),
                "height": int(height),
                "channels": int(channels),
                "brightness": float(brightness),
                "contrast": float(contrast),
                "edge_density": float(edge_density),
                "avg_color_bgr": [float(x) for x in avg_color],
                "dominant_color_bgr": [float(x) for x in dominant_color],
                "histogram_b": hist_b.flatten()[:50].tolist(),  # Simplified histogram
                "histogram_g": hist_g.flatten()[:50].tolist(),
                "histogram_r": hist_r.flatten()[:50].tolist(),
            }
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Error extracting features from {image_path}: {e}")
            return {}
    
    def calculate_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two feature sets
        Sử dụng các metrics đơn giản
        """
        if not features1 or not features2:
            return 0.0
        
        try:
            similarity_scores = []
            
            # Brightness similarity
            if "brightness" in features1 and "brightness" in features2:
                brightness_diff = abs(features1["brightness"] - features2["brightness"])
                brightness_sim = max(0, 1 - brightness_diff / 255)
                similarity_scores.append(brightness_sim * 0.2)
            
            # Contrast similarity
            if "contrast" in features1 and "contrast" in features2:
                contrast_diff = abs(features1["contrast"] - features2["contrast"])
                contrast_sim = max(0, 1 - contrast_diff / 100)
                similarity_scores.append(contrast_sim * 0.1)
            
            # Color similarity
            if "avg_color_bgr" in features1 and "avg_color_bgr" in features2:
                color1 = np.array(features1["avg_color_bgr"])
                color2 = np.array(features2["avg_color_bgr"])
                color_diff = np.linalg.norm(color1 - color2)
                color_sim = max(0, 1 - color_diff / (255 * np.sqrt(3)))
                similarity_scores.append(color_sim * 0.3)
            
            # Histogram similarity
            if all(k in features1 and k in features2 for k in ["histogram_r", "histogram_g", "histogram_b"]):
                hist_similarities = []
                for channel in ["histogram_r", "histogram_g", "histogram_b"]:
                    hist1 = np.array(features1[channel])
                    hist2 = np.array(features2[channel])
                    
                    # Correlation coefficient
                    correlation = np.corrcoef(hist1, hist2)[0, 1]
                    if not np.isnan(correlation):
                        hist_similarities.append(correlation)
                
                if hist_similarities:
                    avg_hist_sim = np.mean(hist_similarities)
                    similarity_scores.append(avg_hist_sim * 0.4)
            
            # Overall similarity
            if similarity_scores:
                return float(np.mean(similarity_scores))
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"❌ Error calculating similarity: {e}")
            return 0.0
    
    def index_frames(self, video_folder: str = None):
        """
        Index all frames in a video folder
        """
        if video_folder is None:
            video_folders = [d for d in self.frames_dir.iterdir() if d.is_dir()]
        else:
            video_folders = [self.frames_dir / video_folder]
        
        for folder in video_folders:
            if not folder.is_dir():
                continue
                
            logger.info(f"🔍 Indexing frames in {folder.name}")
            
            frame_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
            
            for frame_file in frame_files:
                if str(frame_file) not in self.search_index:
                    logger.info(f"📸 Processing {frame_file.name}")
                    features = self.extract_image_features(str(frame_file))
                    
                    if features:
                        self.search_index[str(frame_file)] = {
                            "features": features,
                            "video_folder": folder.name,
                            "frame_name": frame_file.name,
                            "indexed_at": time.time()
                        }
        
        self._save_index()
        logger.info(f"✅ Indexing completed. Total frames: {len(self.search_index)}")
    
    def search_similar_frames(self, query_image: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar frames using image similarity
        """
        if not os.path.exists(query_image):
            logger.error(f"❌ Query image not found: {query_image}")
            return []
        
        logger.info(f"🔍 Searching for frames similar to {query_image}")
        
        # Extract features from query image
        query_features = self.extract_image_features(query_image)
        if not query_features:
            logger.error("❌ Could not extract features from query image")
            return []
        
        # Calculate similarities
        similarities = []
        for frame_path, frame_data in self.search_index.items():
            if "features" in frame_data:
                similarity = self.calculate_similarity(query_features, frame_data["features"])
                similarities.append({
                    "frame_path": frame_path,
                    "similarity": similarity,
                    "video_folder": frame_data.get("video_folder", "unknown"),
                    "frame_name": frame_data.get("frame_name", "unknown"),
                    "features": frame_data["features"]
                })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top K
        results = similarities[:top_k]
        
        logger.info(f"✅ Found {len(results)} similar frames")
        for i, result in enumerate(results[:5]):
            logger.info(f"  {i+1}. {result['frame_name']} (similarity: {result['similarity']:.3f})")
        
        return results
    
    def search_by_color(self, target_color_bgr: List[int], tolerance: int = 50, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search frames by dominant color
        """
        logger.info(f"🎨 Searching frames by color: {target_color_bgr}")
        
        target_color = np.array(target_color_bgr)
        matches = []
        
        for frame_path, frame_data in self.search_index.items():
            if "features" in frame_data and "dominant_color_bgr" in frame_data["features"]:
                frame_color = np.array(frame_data["features"]["dominant_color_bgr"])
                color_distance = np.linalg.norm(target_color - frame_color)
                
                if color_distance <= tolerance:
                    similarity = max(0, 1 - color_distance / (tolerance * 2))
                    matches.append({
                        "frame_path": frame_path,
                        "similarity": similarity,
                        "color_distance": float(color_distance),
                        "video_folder": frame_data.get("video_folder", "unknown"),
                        "frame_name": frame_data.get("frame_name", "unknown"),
                        "dominant_color": frame_data["features"]["dominant_color_bgr"]
                    })
        
        # Sort by similarity
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        
        results = matches[:top_k]
        logger.info(f"✅ Found {len(results)} frames matching color")
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        if not self.search_index:
            return {
                "total_frames": 0,
                "total_videos": 0,
                "avg_brightness": 0,
                "avg_contrast": 0
            }
        
        brightnesses = []
        contrasts = []
        video_folders = set()
        
        for frame_data in self.search_index.values():
            if "features" in frame_data:
                features = frame_data["features"]
                if "brightness" in features:
                    brightnesses.append(features["brightness"])
                if "contrast" in features:
                    contrasts.append(features["contrast"])
            
            if "video_folder" in frame_data:
                video_folders.add(frame_data["video_folder"])
        
        return {
            "total_frames": len(self.search_index),
            "total_videos": len(video_folders),
            "avg_brightness": float(np.mean(brightnesses)) if brightnesses else 0,
            "avg_contrast": float(np.mean(contrasts)) if contrasts else 0,
            "video_folders": list(video_folders)
        }

def test_lite_search_engine():
    """Test function for lite search engine"""
    print("🧪 Testing AI Search Engine Lite...")
    print("=" * 50)
    
    # Initialize
    engine = AISearchEngineLite()
    
    # Get stats
    stats = engine.get_stats()
    print(f"📊 Stats: {stats}")
    
    # Check if we have frames to index
    frames_dir = Path("./frames")
    if frames_dir.exists():
        video_folders = [d for d in frames_dir.iterdir() if d.is_dir()]
        if video_folders:
            print(f"📁 Found {len(video_folders)} video folders")
            
            # Index a small sample
            sample_folder = video_folders[0]
            print(f"🔍 Testing indexing with: {sample_folder.name}")
            engine.index_frames(sample_folder.name)
            
            # Test color search
            print("🎨 Testing color search...")
            results = engine.search_by_color([128, 128, 128], tolerance=100, top_k=5)
            print(f"Found {len(results)} frames with similar color")
            
        else:
            print("⚠️ No video folders found in frames directory")
    else:
        print("⚠️ Frames directory not found")
    
    print("=" * 50)
    print("✅ Lite search engine test completed")

if __name__ == "__main__":
    test_lite_search_engine()
