"""
🔥 ENHANCED FAISS INDEX BUILDER
==============================
Build FAISS indexes for different embedding types:
- Original CLIP embeddings
- Chinese-CLIP embeddings  
- TensorFlow Hub USE embeddings (optional)
"""

import os
import faiss
import numpy as np
import pandas as pd
import json
from typing import Optional

def build_faiss_index(embedding_path: str, output_path: str, 
                     embedding_dim: int = 512, 
                     index_type: str = "flat_ip") -> bool:
    """
    Build FAISS index from embeddings
    
    Args:
        embedding_path: Path to embeddings file (.f16.mmap)
        output_path: Output path for FAISS index
        embedding_dim: Dimension of embeddings
        index_type: Type of FAISS index ("flat_ip", "flat_l2", "ivf")
    """
    try:
        print(f"🔄 Loading embeddings from: {embedding_path}")
        
        # Đếm số lượng vector embedding thực tế từ file mmap
        if not os.path.exists(embedding_path):
            print(f"❌ Embeddings not found: {embedding_path}")
            return False
        file_size = os.path.getsize(embedding_path)
        N = file_size // (2 * embedding_dim)  # float16 = 2 bytes
        print(f"   Số lượng vectors: {N:,}")
        vecs = np.memmap(embedding_path, dtype='float16', mode='r', shape=(N, embedding_dim)).astype('float32')
        print(f"   Loaded embeddings shape: {vecs.shape}")
        
        # Normalize for cosine similarity (if using IP index)
        if "ip" in index_type.lower():
            print("🔄 Normalizing vectors for cosine similarity...")
            faiss.normalize_L2(vecs)
        
        # Create FAISS index
        print(f"🔧 Building FAISS index (type: {index_type})...")
        
        if index_type == "flat_ip":
            index = faiss.IndexFlatIP(embedding_dim)
        elif index_type == "flat_l2":
            index = faiss.IndexFlatL2(embedding_dim)
        elif index_type == "ivf":
            # IVF index for larger datasets
            quantizer = faiss.IndexFlatIP(embedding_dim)
            nlist = min(4096, N // 39)  # Rule of thumb: N/39
            index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
            index.train(vecs)
        else:
            print(f"❌ Unknown index type: {index_type}")
            return False
        
        # Add vectors to index
        print("📥 Adding vectors to index...")
        index.add(vecs)
        
        # Save index
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        faiss.write_index(index, output_path)
        
        print(f"✅ FAISS index saved: {output_path}")
        print(f"   Total vectors: {index.ntotal:,}")
        print(f"   Index size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to build FAISS index: {e}")
        return False

def build_metadata_index(meta_path: str = 'index/meta.parquet', 
                        output_path: str = 'index/frames_meta.json'):
    """
    Build enhanced metadata with text content for USE scoring
    """
    try:
        print("📊 Building metadata for all .jpg frames in 'frames/' ...")
        print("   Đang quét toàn bộ file .jpg bằng os.walk...")
        enhanced_meta = []
        idx = 0
        for root, dirs, files in os.walk('frames'):
            for file in files:
                if file.lower().endswith('.jpg'):
                    frame_path = os.path.join(root, file)
                    parts = frame_path.replace('\\', '/').split('/')
                    video_folder = parts[1] if len(parts) > 2 else ''
                    frame_name = parts[-1]
                    clean_title = video_folder.replace('[', ' ').replace(']', ' ').replace('_', ' ')
                    words = clean_title.split()
                    labels_obj = []
                    labels_audio = []
                    programming_keywords = ['react', 'javascript', 'code', 'programming', 'tutorial', 'lesson']
                    for word in words:
                        word_lower = word.lower()
                        if word_lower in programming_keywords:
                            labels_obj.append(word_lower)
                    try:
                        frame_number = int(''.join(filter(str.isdigit, frame_name)))
                    except:
                        frame_number = None
                    timestamp_seconds = frame_number if frame_number is not None else None
                    timestamp = f"{timestamp_seconds//60:02}:{timestamp_seconds%60:02}" if timestamp_seconds is not None else None
                    record = {
                        'frame_id': idx,
                        'video_file': video_folder,
                        'frame_number': frame_number,
                        'frame_path': frame_path,
                        'relative_path': frame_path.replace('\\', '/'),
                        'timestamp': timestamp,
                        'timestamp_seconds': timestamp_seconds,
                        'tx': clean_title,
                        'labels_obj': labels_obj,
                        'labels_audio': labels_audio,
                        'title_words': words
                    }
                    enhanced_meta.append(record)
                    idx += 1
        print(f"   Đã tìm thấy {len(enhanced_meta):,} file .jpg")
        # Save enhanced metadata
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_meta, f, ensure_ascii=False, indent=2)
        print(f"✅ Metadata for all frames saved: {output_path}")
        print(f"   Records: {len(enhanced_meta):,}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to build metadata: {e}")
        return False

def main():
    """Main function to build all indexes"""
    print("🔥 ENHANCED FAISS INDEX BUILDER")
    print("=" * 40)
    
    success_count = 0
    total_tasks = 0
    
    # 1. Build enhanced metadata
    print("\n1. Building enhanced metadata...")
    total_tasks += 1
    if build_metadata_index():
        success_count += 1
    
    # 2. Build FAISS index for original embeddings
    if os.path.exists('index/embeddings/frames.f16.mmap'):
        print("\n2. Building FAISS index for original CLIP embeddings...")
        total_tasks += 1
        if build_faiss_index(
            embedding_path='index/embeddings/frames.f16.mmap',
            output_path='index/faiss/ip_flat.index',
            index_type='flat_ip'
        ):
            success_count += 1
    
    # 3. Build FAISS index for Chinese-CLIP embeddings
    if os.path.exists('index/embeddings/frames_chinese_clip.f16.mmap'):
        print("\n3. Building FAISS index for Chinese-CLIP embeddings...")
        total_tasks += 1
        if build_faiss_index(
            embedding_path='index/embeddings/frames_chinese_clip.f16.mmap',
            output_path='index/faiss/ip_flat_chinese_clip.index',
            index_type='flat_ip'
        ):
            success_count += 1
    
    # 4. Build IVF index for faster search (if dataset is large)
    if os.path.exists('index/embeddings/frames_chinese_clip.f16.mmap'):
        print("\n4. Building IVF index for faster search...")
        total_tasks += 1
        if build_faiss_index(
            embedding_path='index/embeddings/frames_chinese_clip.f16.mmap',
            output_path='index/faiss/ivf_chinese_clip.index',
            index_type='ivf'
        ):
            success_count += 1
    
    # Summary
    print(f"\n📊 SUMMARY")
    print("=" * 20)
    print(f"✅ Successful: {success_count}/{total_tasks}")
    print(f"📁 Output directory: index/faiss/")
    
    if success_count == total_tasks:
        print("🎉 All indexes built successfully!")
    else:
        print("⚠️  Some tasks failed. Check errors above.")

if __name__ == "__main__":
    main()