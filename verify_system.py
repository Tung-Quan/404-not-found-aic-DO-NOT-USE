#!/usr/bin/env python3
"""
🔍 System Verification Test
Kiểm tra tất cả các components chính của AI Video Search System
"""

import sys
import os

def test_imports():
    """Test all critical imports"""
    print("🔄 Testing imports...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__}")
    except ImportError as e:
        print(f"❌ TensorFlow: {e}")
        return False
        
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"   🎮 GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
        
    try:
        import sentence_transformers
        print(f"✅ sentence-transformers {sentence_transformers.__version__}")
    except ImportError as e:
        print(f"❌ sentence-transformers: {e}")
        return False
        
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy: {e}")
        return False
    
    return True

def test_core_modules():
    """Test core system modules"""
    print("\n🔄 Testing core modules...")
    
    # Test basic imports
    try:
        from ai_search_engine import VideoSearchEngine
        print("✅ VideoSearchEngine")
    except Exception as e:
        print(f"❌ VideoSearchEngine: {e}")
        
    try:
        from tensorflow_model_manager import TensorFlowModelManager
        print("✅ TensorFlowModelManager")
    except Exception as e:
        print(f"❌ TensorFlowModelManager: {e}")
        
    try:
        from enhanced_hybrid_manager import EnhancedHybridManager
        print("✅ EnhancedHybridManager")
    except Exception as e:
        print(f"❌ EnhancedHybridManager: {e}")

def main():
    """Run all tests"""
    print("🎯 AI Video Search System - Verification Test")
    print("=" * 60)
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ Critical imports failed!")
        sys.exit(1)
    
    # Test 2: Core modules
    test_core_modules()
    
    print("\n" + "=" * 60)
    print("🎉 System verification completed!")
    print("✅ Ready to run AI Video Search System")
    print("\nNext steps:")
    print("1. Run: python setup.py (if needed)")
    print("2. Start API: python api/app.py")
    print("3. Or use: python main_launcher.py")

if __name__ == "__main__":
    main()
