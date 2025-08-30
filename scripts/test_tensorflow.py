#!/usr/bin/env python3
"""
Simple TensorFlow Test Script
"""

import os
import sys

# Setup environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONIOENCODING'] = 'utf-8'

print("Testing TensorFlow imports...")
print("=" * 40)

try:
    print("1. Testing basic TensorFlow...")
    import tensorflow as tf
    print(f"[OK] TensorFlow {tf.__version__} imported successfully")
    
    print("\n2. Testing TensorFlow Hub...")
    import tensorflow_hub as hub
    print("[OK] TensorFlow Hub imported successfully")
    
    print("\n3. Testing TensorFlow Text (optional)...")
    try:
        import tensorflow_text
        print("[OK] TensorFlow Text imported successfully")
    except ImportError:
        print("[INFO] TensorFlow Text not available (optional)")
    
    print("\n4. Testing basic model loading...")
    try:
        # Test a simple operation
        x = tf.constant([1, 2, 3])
        print(f"[OK] Basic TensorFlow operation: {x}")
        
        print("\n[SUCCESS] All TensorFlow tests passed!")
        
    except Exception as e:
        print(f"[ERROR] TensorFlow operation failed: {e}")
        
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print("\nTry running: python scripts/fix_tensorflow.py")
    
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 40)
print("Test completed.")
