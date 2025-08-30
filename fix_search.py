#!/usr/bin/env python3
"""
Fix Search Engine - Build Embeddings
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from ai_search_engine import EnhancedAIVideoSearchEngine

print('🔧 Initializing search engine and building embeddings...')
engine = EnhancedAIVideoSearchEngine()

print('📊 Current embeddings status:')
print(f'Available embeddings: {list(engine.image_embeddings.keys())}')

print('🚀 Building embeddings index...')
try:
    # Try to build embeddings index
    success = engine.build_embeddings_index()
    print(f'Build result: {success}')
    
    if success:
        print('✅ Embeddings built successfully')
        
        print('🔍 Testing search after building embeddings...')
        results = engine.search_similar_frames('person', top_k=3)
        print(f'Search results: {len(results)} found')
        
        if len(results) > 0:
            print('✅ Search working!')
            result = results[0]
            print(f'First result similarity: {result.get("similarity_score", "N/A")}')
            print(f'Video name: {result.get("video_name", "N/A")}')
        else:
            print('❌ Still no results after building embeddings')
    else:
        print('❌ Failed to build embeddings')
        
except Exception as e:
    print(f'❌ Error building embeddings: {e}')
    import traceback
    traceback.print_exc()
