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
        
        # Load metadata to get number of vectors
        meta = pd.read_parquet('index/meta.parquet')
        N = len(meta)
        print(f"   Expected vectors: {N:,}")
        
        # Load embeddings
        if not os.path.exists(embedding_path):
            print(f"❌ Embeddings not found: {embedding_path}")
            return False
            
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
        print("📊 Building enhanced metadata...")
        
        # Load existing metadata
        meta = pd.read_parquet(meta_path)
        print(f"   Loaded {len(meta)} records")
        
        # Enhance metadata with text content
        enhanced_meta = []
        for _, row in meta.iterrows():
            # Extract text from video title
            video_title = row['video_id']
            
            # Clean up title for better text processing
            clean_title = video_title.replace('[', ' ').replace(']', ' ').replace('_', ' ')
            words = clean_title.split()
            
            # Extract potential labels/keywords from title
            labels_obj = []
            labels_audio = []
            
            # Simple keyword extraction (can be enhanced)
            programming_keywords = ['react', 'javascript', 'code', 'programming', 'tutorial', 'lesson']
            for word in words:
                word_lower = word.lower()
                if word_lower in programming_keywords:
                    labels_obj.append(word_lower)
            
            # Create enhanced metadata record
            record = {
                'frame_id': len(enhanced_meta),
                'video_id': row['video_id'],
                'ts': int(row['ts']),
                'frame_path': row['frame_path'],
                'tx': clean_title,  # Cleaned title text
                'labels_obj': labels_obj,
                'labels_audio': labels_audio,
                'title_words': words
            }
            enhanced_meta.append(record)
        
        # Save enhanced metadata
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_meta, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Enhanced metadata saved: {output_path}")
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