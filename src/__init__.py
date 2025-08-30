"""
Enhanced Video Search System - Source Package
"""

__version__ = "2.0.0"
__author__ = "Enhanced Video Search Team"
__description__ = "Advanced AI-powered video search with TensorFlow Hub integration"

# Package imports
from .core.enhanced_video_processor import TensorFlowHubVideoManager, ModelType
from .api.app import app as enhanced_api
from .ui.enhanced_web_interface import main as enhanced_web_ui

__all__ = [
    "TensorFlowHubVideoManager",
    "ModelType", 
    "enhanced_api",
    "enhanced_web_ui"
]
