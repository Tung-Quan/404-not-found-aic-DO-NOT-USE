#!/usr/bin/env python3
"""
System Status Checker for Enhanced Video Search System
"""
import os
import sys

def check_system_status():
    print("System Status Check")
    print("=" * 20)
    print()
    
    # Check virtual environment
    if 'VIRTUAL_ENV' in os.environ:
        print("[OK] Virtual environment: Active")
        print(f"  Path: {os.environ['VIRTUAL_ENV']}")
    else:
        print("[WARNING] Virtual environment: Not active")
    
    print()
    
    # Check metadata
    try:
        import pandas as pd
        meta = pd.read_parquet('index/meta.parquet')
        print(f"[OK] Metadata: {len(meta)} frames loaded")
    except Exception as e:
        print(f"[WARNING] Metadata: Not found ({str(e)[:50]}...)")
    
    # Check FAISS index
    try:
        import faiss
        idx = faiss.read_index('index/faiss/ip_flat_chinese_clip.index')
        print(f"✓ FAISS index: {idx.ntotal} vectors loaded")
    except Exception as e:
        print(f"✗ FAISS index: Not found ({str(e)[:50]}...)")
    
    # Check enhanced metadata
    try:
        import json
        with open('index/frames_meta.json') as f:
            meta = json.load(f)
        print(f"✓ Enhanced metadata: {len(meta)} records")
    except Exception as e:
        print(f"✗ Enhanced metadata: Not found ({str(e)[:50]}...)")
    
    # Check TensorFlow Hub
    try:
        import tensorflow_hub as hub
        print("✓ TensorFlow Hub: Available")
    except Exception as e:
        print(f"✗ TensorFlow Hub: Not installed ({str(e)[:50]}...)")
    
    # Check TensorFlow Text (optional)
    try:
        import tensorflow_text as tf_text
        print("✓ TensorFlow Text: Available")
    except Exception as e:
        print(f"⚠️  TensorFlow Text: Not installed (optional) ({str(e)[:50]}...)")
    
    # Check core dependencies
    print()
    print("=== Core Dependencies ===")
    dependencies = [
        ('fastapi', 'FastAPI', 'pip install fastapi uvicorn'),
        ('streamlit', 'Streamlit', 'pip install streamlit'),
        ('numpy', 'NumPy', 'pip install numpy'),
        ('cv2', 'OpenCV', 'pip install opencv-python'),
        ('PIL', 'Pillow', 'pip install Pillow'),
    ]
    
    missing_deps = []
    for module, name, install_cmd in dependencies:
        try:
            __import__(module)
            print(f"✓ {name}: Available")
        except ImportError:
            print(f"✗ {name}: Not installed")
            missing_deps.append((name, install_cmd))
    
    # Show installation instructions if there are missing dependencies
    if missing_deps:
        print()
        print("=== Installation Instructions ===")
        print("To install missing dependencies, run:")
        print()
        for name, cmd in missing_deps:
            print(f"  {cmd}")
        print()
        print("Or install all at once:")
        all_packages = []
        for _, cmd in missing_deps:
            package = cmd.replace('pip install ', '')
            all_packages.append(package)
        print(f"  pip install {' '.join(all_packages)}")
        print()
        print("For TensorFlow Hub (required for enhanced features):")
        print("  pip install tensorflow tensorflow-hub tensorflow-text")
    
    print()
    print("=== Status Check Complete ===")

if __name__ == "__main__":
    check_system_status()
