#!/usr/bin/env python3
"""
Debug search engine embeddings loading
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from ai_search_engine import EnhancedAIVideoSearchEngine
from enhanced_hybrid_manager import EnhancedHybridModelManager

print('🔧 Testing search engine with same config as web interface...')

# Initialize exactly like web interface
model_manager = EnhancedHybridModelManager()
search_engine = EnhancedAIVideoSearchEngine(model_manager=model_manager)

print(f'📊 Loaded {len(search_engine.frames_metadata)} frame records')

# Load CLIP model
print('🚀 Loading CLIP model...')
success = model_manager.load_model("clip_vit_base")
if success:
    search_engine.set_active_model("vision_language", "clip_vit_base")
    print('✅ CLIP model loaded and set as active')
else:
    print('❌ Failed to load CLIP model')

# Check active models
active_models = search_engine.get_active_models()
print(f'🎯 Active models: {active_models}')

# Check embeddings status
print('\n📊 Embeddings status:')
print(f'Image embeddings loaded: {len(search_engine.image_embeddings)}')
for model_key, embeddings in search_engine.image_embeddings.items():
    if embeddings is not None:
        print(f'  - {model_key}: {embeddings.shape if hasattr(embeddings, "shape") else "Unknown shape"}')
    else:
        print(f'  - {model_key}: None')

# Check FAISS indices
print('\n🔍 FAISS indices status:')
for model_key, index in search_engine.faiss_indexes.items():
    if index is not None:
        total_vectors = index.ntotal if hasattr(index, 'ntotal') else 'Unknown'
        print(f'  - {model_key}: {total_vectors} vectors')
    else:
        print(f'  - {model_key}: None')

# Try to search
print('\n🔍 Testing search...')
try:
    results = search_engine.search_similar_frames('person walking', top_k=3)
    print(f'Search results: {len(results)} found')
    
    if len(results) > 0:
        print('✅ Search working!')
        for i, result in enumerate(results[:2]):
            similarity = result.get('similarity_score', result.get('similarity', 0))
            video = result.get('video_name', 'N/A')
            print(f'  Result {i+1}: similarity={similarity:.3f}, video={video}')
    else:
        print('❌ No results found')
        
        # Debug: check if we need to build embeddings
        if not search_engine.image_embeddings.get('clip_vit_base'):
            print('🚀 Embeddings not loaded, trying to build...')
            build_success = search_engine.build_embeddings_index()
            print(f'Build result: {build_success}')
            
            if build_success:
                print('🔍 Retesting search after building...')
                results = search_engine.search_similar_frames('person walking', top_k=3)
                print(f'New search results: {len(results)} found')
            
except Exception as e:
    print(f'❌ Search error: {e}')
    import traceback
    traceback.print_exc()
