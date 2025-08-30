#!/usr/bin/env python3
"""
Environment Setup Script
Sets up proper environment variables to avoid TensorFlow warnings
"""

import os
import sys

def setup_environment():
    """Setup environment variables for better TensorFlow performance"""
    print("Setting up environment variables...")
    
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations warnings
    
    # UTF-8 encoding
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # Suppress Protobuf warnings
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    
    # TensorFlow memory management
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # Suppress deprecation warnings
    os.environ['TF_SUPPRESS_DEPRECATION_WARNINGS'] = '1'
    
    print("[OK] Environment variables set:")
    print(f"   TF_CPP_MIN_LOG_LEVEL: {os.environ.get('TF_CPP_MIN_LOG_LEVEL')}")
    print(f"   TF_ENABLE_ONEDNN_OPTS: {os.environ.get('TF_ENABLE_ONEDNN_OPTS')}")
    print(f"   PYTHONIOENCODING: {os.environ.get('PYTHONIOENCODING')}")
    
    return True

if __name__ == "__main__":
    setup_environment()
