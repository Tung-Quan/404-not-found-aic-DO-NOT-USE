#!/usr/bin/env python3
"""
🔍 AI Video Search - Embedding System Status Checker
Kiểm tra trạng thái của hệ thống embedding và đưa ra khuyến nghị
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

def print_banner():
    print("🎯 AI EMBEDDING SYSTEM - STATUS CHECK")
    print("=" * 50)

def check_metadata():
    """Kiểm tra metadata frames"""
    print("\n📋 Checking frame metadata...")
    
    meta_files = ['index/meta.parquet', 'index/metadata.json', 'frames/search_index_lite.json']
    found_meta = []
    
    for meta_file in meta_files:
        if os.path.exists(meta_file):
            if meta_file.endswith('.parquet'):
                try:
                    df = pd.read_parquet(meta_file)
                    print(f"✅ {meta_file} - {len(df):,} frames")
                    found_meta.append((meta_file, len(df)))
                except Exception as e:
                    print(f"❌ {meta_file} - Error: {e}")
            else:
                size_kb = os.path.getsize(meta_file) / 1024
                print(f"✅ {meta_file} - {size_kb:.1f} KB")
                found_meta.append((meta_file, size_kb))
        else:
            print(f"❌ {meta_file} - Not found")
    
    return found_meta

def check_embeddings():
    """Kiểm tra embeddings files"""
    print("\n🎯 Checking embedding files...")
    
    embedding_files = {
        'index/embeddings/image_embeddings_clip_vit_base.npy': 'CLIP Standard',
        'index/embeddings/frames_chinese_clip.f16.mmap': 'Chinese-CLIP (Vietnamese optimized)',
        'index/embeddings/frames_siglip.f16.mmap': 'SigLIP Multilingual'
    }
    
    found_embeddings = []
    
    for emb_file, description in embedding_files.items():
        if os.path.exists(emb_file):
            size_mb = os.path.getsize(emb_file) / 1024 / 1024
            
            # Try to load and check shape
            try:
                if emb_file.endswith('.npy'):
                    arr = np.load(emb_file)
                    shape = arr.shape
                elif emb_file.endswith('.mmap'):
                    # Guess shape from file size (512 dim, float16)
                    expected_vectors = int(size_mb * 1024 * 1024 / (512 * 2))
                    shape = f"({expected_vectors}, 512)"
                
                print(f"✅ {description}")
                print(f"   📁 {emb_file}")
                print(f"   📊 Size: {size_mb:.1f} MB, Shape: {shape}")
                found_embeddings.append((emb_file, description, size_mb))
                
            except Exception as e:
                print(f"⚠️ {description} - File exists but error loading: {e}")
        else:
            print(f"❌ {description} - Not found")
            print(f"   📁 Expected: {emb_file}")
    
    return found_embeddings

def check_faiss_indexes():
    """Kiểm tra FAISS indexes"""
    print("\n🔍 Checking FAISS indexes...")
    
    index_dirs = ['index/faiss/', 'index/qdrant/']
    found_indexes = []
    
    for index_dir in index_dirs:
        if os.path.exists(index_dir):
            index_files = list(Path(index_dir).glob('**/*'))
            if index_files:
                print(f"✅ {index_dir} - {len(index_files)} files")
                for idx_file in index_files[:5]:  # Show first 5
                    size_kb = idx_file.stat().st_size / 1024
                    print(f"   📄 {idx_file.name} ({size_kb:.1f} KB)")
                if len(index_files) > 5:
                    print(f"   ... and {len(index_files) - 5} more files")
                found_indexes.append(index_dir)
            else:
                print(f"⚠️ {index_dir} - Directory exists but empty")
        else:
            print(f"❌ {index_dir} - Not found")
    
    return found_indexes

def check_scripts():
    """Kiểm tra embedding scripts"""
    print("\n🛠️ Checking embedding scripts...")
    
    scripts = {
        'scripts/encode_chinese_clip.py': 'Create Chinese-CLIP embeddings',
        'scripts/build_faiss_chinese_clip.py': 'Build FAISS index (Chinese-CLIP)',
        'scripts/build_faiss.py': 'Build FAISS index (Standard)',
        'scripts/text_embed.py': 'Text embedding utility'
    }
    
    available_scripts = []
    
    for script, description in scripts.items():
        if os.path.exists(script):
            print(f"✅ {description}")
            print(f"   📄 {script}")
            available_scripts.append(script)
        else:
            print(f"❌ {description} - Not found")
            print(f"   📄 Expected: {script}")
    
    return available_scripts

def recommend_next_steps(found_meta, found_embeddings, found_indexes, available_scripts):
    """Đưa ra khuyến nghị next steps"""
    print("\n🎯 RECOMMENDATIONS")
    print("=" * 30)
    
    if not found_meta:
        print("❗ Step 1: Process videos to extract frames")
        print("   💡 Run: python main_launcher.py")
        print("   💡 Choose option to process videos")
        return
    
    if not found_embeddings:
        print("❗ Step 1: Create embeddings from frames")
        if 'scripts/encode_chinese_clip.py' in available_scripts:
            print("   🚀 Run: python scripts/encode_chinese_clip.py")
            print("   💡 Chinese-CLIP recommended for Vietnamese content")
        else:
            print("   ⚠️ Missing embedding scripts - reinstall system")
        return
    
    if not found_indexes:
        print("❗ Step 1: Build FAISS index for fast search")
        if 'scripts/build_faiss_chinese_clip.py' in available_scripts:
            print("   🚀 Run: python scripts/build_faiss_chinese_clip.py")
        elif 'scripts/build_faiss.py' in available_scripts:
            print("   🚀 Run: python scripts/build_faiss.py")
        else:
            print("   ⚠️ Missing FAISS build scripts")
        return
    
    # All components available
    print("🎉 SYSTEM READY!")
    print("✅ Metadata: Available")
    print("✅ Embeddings: Available")
    print("✅ FAISS Index: Available")
    print()
    print("🚀 You can now:")
    print("   • Launch search system: python main_launcher.py")
    print("   • Test search: python -c \"from ai_search_lite import test_lite_search_engine; test_lite_search_engine()\"")
    print("   • Start web API: cd api && python app.py")

def main():
    print_banner()
    
    # Check all components
    found_meta = check_metadata()
    found_embeddings = check_embeddings()
    found_indexes = check_faiss_indexes()
    available_scripts = check_scripts()
    
    # Summary
    print(f"\n📊 SUMMARY")
    print(f"=" * 20)
    print(f"Metadata files: {len(found_meta)}")
    print(f"Embedding files: {len(found_embeddings)}")
    print(f"FAISS indexes: {len(found_indexes)}")
    print(f"Available scripts: {len(available_scripts)}")
    
    # Recommendations
    recommend_next_steps(found_meta, found_embeddings, found_indexes, available_scripts)

if __name__ == "__main__":
    main()
