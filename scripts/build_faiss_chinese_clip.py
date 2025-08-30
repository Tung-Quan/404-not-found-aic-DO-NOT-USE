import numpy as np
import pandas as pd
import faiss
import os

print("🇨🇳 BUILDING FAISS INDEX FOR CHINESE-CLIP")
print("=" * 50)

# Load Chinese-CLIP embeddings
embedding_path = 'index/embeddings/frames_chinese_clip.f16.mmap'

print(f"📁 Loading embeddings from: {embedding_path}")
# Đếm số lượng embedding theo file mmap
import os
file_size = os.path.getsize(embedding_path)
DIM = 512
N = file_size // (2 * DIM)  # float16 = 2 bytes
print(f"📊 Tổng số vectors: {N:,}")
print(f"📏 Embedding dimension: {DIM}")
embeddings = np.memmap(embedding_path, dtype='float16', mode='r', shape=(N, DIM))

# Convert to float32 for FAISS
print("🔄 Converting embeddings to float32...")
embeddings_f32 = embeddings.astype('float32')

print("📊 Embedding statistics:")
print(f"   Shape: {embeddings_f32.shape}")
print(f"   Mean norm: {np.linalg.norm(embeddings_f32, axis=1).mean():.4f}")
print(f"   Memory usage: {embeddings_f32.nbytes / 1024 / 1024:.1f} MB")

# Create FAISS index
print("\n🔧 Building FAISS IndexFlatIP...")
index = faiss.IndexFlatIP(DIM)

# Add embeddings to index
print("📥 Adding embeddings to index...")
index.add(embeddings_f32)

print(f"✅ FAISS index built successfully!")
print(f"   Index type: {type(index).__name__}")
print(f"   Total vectors: {index.ntotal:,}")
print(f"   Dimension: {index.d}")

# Save index
os.makedirs('index/faiss', exist_ok=True)
output_path = 'index/faiss/ip_flat_chinese_clip.index'

print(f"\n💾 Saving index to: {output_path}")
faiss.write_index(index, output_path)

print(f"✅ FAISS index saved!")
print(f"📁 Location: {output_path}")
print(f"💾 Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

print(f"\n🎯 Chinese-CLIP index ready!")
print("Next: Update API servers to use new index")
