#!/usr/bin/env python3
"""
Fix Search Engine - Build Embeddings with Model Loading
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from ai_search_engine import EnhancedAIVideoSearchEngine

print('🔧 Initializing search engine and building embeddings...')
engine = EnhancedAIVideoSearchEngine()

print(f'💾 Device: {engine.device}')
print(f'📊 Loaded {len(engine.frames_metadata)} frame records')

# Check available models
print(f'📋 Available models: {list(engine.model_manager.loaded_models.keys())}')

# Load a model if none are loaded
if not engine.model_manager.loaded_models:
    print('⚠️ No models loaded. Loading CLIP model...')
    try:
        # Get available models from config
        from enhanced_hybrid_manager import PYTORCH_MODELS
        clip_model = None
        for model_key, config in PYTORCH_MODELS.items():
            if 'clip' in model_key.lower():
                clip_model = model_key
                break
        
        if clip_model:
            print(f'🚀 Loading {clip_model}...')
            engine.model_manager.load_model(clip_model)
            print(f'✅ Model {clip_model} loaded')
        else:
            print('❌ No CLIP model found in config')
    except Exception as e:
        print(f'❌ Error loading model: {e}')

# Now check for active model
active_model = engine.get_active_models()
print(f'🎯 Active model: {active_model}')

if not active_model:
    print('⚠️ No active model. Setting default...')
    loaded_models = list(engine.model_manager.loaded_models.keys())
    if loaded_models:
        engine.set_active_model(loaded_models[0])
        print(f'✅ Set active model to: {loaded_models[0]}')
    else:
        print('❌ Cannot set active model - no models loaded')

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
