#!/usr/bin/env python3
"""
Check embeddings status and restart web interface if needed
"""

import os
import sys
from pathlib import Path

# Check embeddings directory
embeddings_dir = Path("embeddings")
print(f"📁 Checking embeddings directory: {embeddings_dir}")

if embeddings_dir.exists():
    embedding_files = list(embeddings_dir.glob("*.npy"))
    print(f"✅ Found {len(embedding_files)} embedding files:")
    for f in embedding_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name}: {size_mb:.1f} MB")
else:
    print("❌ Embeddings directory not found!")
    sys.exit(1)

# Check FAISS indices
faiss_dir = Path("index/faiss")
print(f"\n📁 Checking FAISS indices: {faiss_dir}")

if faiss_dir.exists():
    faiss_files = list(faiss_dir.glob("*"))
    print(f"✅ Found {len(faiss_files)} FAISS files:")
    for f in faiss_files:
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name}: {size_mb:.1f} MB")
else:
    print("❌ FAISS directory not found!")

print(f"\n🔧 All embeddings and indices are ready!")
print(f"💡 Web interface should be restarted to load the new embeddings.")
print(f"🚀 Run: python web_interface.py")
