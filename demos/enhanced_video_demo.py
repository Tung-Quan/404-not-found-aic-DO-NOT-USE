"""
🎬 ENHANCED VIDEO PROCESSOR DEMO
===============================
Interactive demo for TensorFlow Hub model selection and video processing
"""

import os
import sys
import time

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.enhanced_video_processor import TensorFlowHubVideoManager, ModelType

def print_header():
    """Print demo header"""
    print("🎬 ENHANCED VIDEO PROCESSOR DEMO")
    print("=" * 60)
    print("Intelligent TensorFlow Hub model selection for video processing")
    print()

def demo_model_analysis():
    """Demo intelligent model analysis"""
    print("🤖 INTELLIGENT MODEL ANALYSIS DEMO")
    print("-" * 40)
    
    manager = TensorFlowHubVideoManager()
    
    # Test different user intents
    test_intents = [
        "I want to find action sequences in cooking videos",
        "Detect objects and text in tutorial videos", 
        "Search for specific movements and activities in sports videos",
        "Analyze visual scenes and understand video content",
        "Find text mentions and visual elements together"
    ]
    
    for i, intent in enumerate(test_intents, 1):
        print(f"\n📝 Test Intent {i}: {intent}")
        print("-" * 50)
        
        # Get recommendations
        recommendations = manager.suggest_model_combinations(intent, max_memory_mb=2000)
        
        # Show recommendations
        for config_type in ['lightweight', 'balanced', 'high_accuracy']:
            if config_type in recommendations and recommendations[config_type]:
                print(f"\n{config_type.upper()} Configuration:")
                for model in recommendations[config_type]:
                    if model in manager.model_configs:
                        config = manager.model_configs[model]
                        print(f"   • {config.name}")
                        print(f"     Type: {config.model_type.value}")
                        print(f"     Memory: {config.memory_usage}")
                        print(f"     Speed: {config.processing_speed}")
        
        # Show overlaps
        if 'overlaps_detected' in recommendations and recommendations['overlaps_detected']:
            print(f"\n⚠️  Overlapping Models Detected:")
            for config_type, overlaps in recommendations['overlaps_detected'].items():
                print(f"   {config_type}: {overlaps}")
        
        print("\n" + "="*60)

def demo_interactive_selection():
    """Demo interactive model selection"""
    print("\n🎯 INTERACTIVE MODEL SELECTION DEMO")
    print("-" * 40)
    
    manager = TensorFlowHubVideoManager()
    
    # Get user input
    print("Let's select models based on your specific needs!")
    user_intent = input("\n📝 What do you want to do with videos? ")
    
    try:
        max_memory = int(input("💾 Maximum memory usage (MB, default 2000): ") or "2000")
    except ValueError:
        max_memory = 2000
    
    # Get recommendations
    print(f"\n🔄 Analyzing your requirements...")
    recommendations = manager.suggest_model_combinations(user_intent, max_memory)
    
    # Show recommendations and let user choose
    selected_models = []
    
    print(f"\n📊 RECOMMENDATIONS FOR: '{user_intent}'")
    print("=" * 60)
    
    # Show all configuration options
    for config_type in ['lightweight', 'balanced', 'high_accuracy']:
        if config_type in recommendations and recommendations[config_type]:
            print(f"\n🔧 {config_type.upper()} Configuration:")
            models = recommendations[config_type]
            
            total_memory = 0
            for model in models:
                if model in manager.model_configs:
                    config = manager.model_configs[model]
                    print(f"   • {config.name}")
                    print(f"     Memory: {config.memory_usage} | Speed: {config.processing_speed}")
                    print(f"     {config.description}")
                    
                    # Estimate memory
                    try:
                        memory_str = config.memory_usage.replace('~', '').replace('MB', '')
                        memory_val = float(memory_str)
                        total_memory += memory_val
                    except:
                        pass
            
            print(f"   📊 Estimated total memory: ~{total_memory:.0f}MB")
    
    # Handle overlaps
    if 'overlaps_detected' in recommendations and recommendations['overlaps_detected']:
        print(f"\n⚠️  OVERLAPPING FUNCTIONALITY DETECTED")
        print("Some models have similar functionality. Choose wisely:")
        
        for config_type, overlaps in recommendations['overlaps_detected'].items():
            print(f"\n{config_type.upper()} Configuration Conflicts:")
            for model, overlapping_with in overlaps.items():
                if model in manager.model_configs:
                    config = manager.model_configs[model]
                    print(f"   {config.name} overlaps with:")
                    for overlap_model in overlapping_with:
                        if overlap_model in manager.model_configs:
                            overlap_config = manager.model_configs[overlap_model]
                            print(f"     - {overlap_config.name}")
    
    # Let user select configuration
    print(f"\n🎯 Select your preferred configuration:")
    print("   1. Lightweight (Fast, Low Memory)")
    print("   2. Balanced (Good Performance/Memory)")
    print("   3. High Accuracy (Best Quality, High Memory)")
    print("   4. Custom selection")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == '1':
        selected_models = recommendations.get('lightweight', [])
    elif choice == '2':
        selected_models = recommendations.get('balanced', [])
    elif choice == '3':
        selected_models = recommendations.get('high_accuracy', [])
    elif choice == '4':
        # Custom selection
        print("\nAvailable models:")
        all_models = set()
        for models in recommendations.values():
            if isinstance(models, list):
                all_models.update(models)
        
        for i, model in enumerate(sorted(all_models), 1):
            if model in manager.model_configs:
                config = manager.model_configs[model]
                print(f"   {i}. {config.name}")
        
        selected_indices = input("Select models (comma-separated numbers): ").strip()
        try:
            indices = [int(x.strip()) for x in selected_indices.split(',')]
            model_list = list(sorted(all_models))
            selected_models = [model_list[i-1] for i in indices if 1 <= i <= len(model_list)]
        except:
            print("Invalid selection, using balanced configuration")
            selected_models = recommendations.get('balanced', [])
    else:
        selected_models = recommendations.get('balanced', [])
    
    if not selected_models:
        print("❌ No models selected")
        return None
    
    print(f"\n✅ Selected models: {selected_models}")
    
    # Ask if user wants to load models
    load_choice = input(f"\n🚀 Load selected models? This may take several minutes. (y/n): ").lower().strip()
    
    if load_choice == 'y':
        print(f"\n🔄 Loading {len(selected_models)} models...")
        print("⚠️  This may take 5-10 minutes depending on your internet connection")
        
        try:
            results = manager.load_selected_models(selected_models)
            
            print(f"\n📊 LOADING RESULTS:")
            success_count = 0
            for model, success in results.items():
                if success:
                    print(f"   ✅ {model}")
                    success_count += 1
                else:
                    print(f"   ❌ {model}")
            
            print(f"\n🎉 Successfully loaded {success_count}/{len(selected_models)} models")
            
            if success_count > 0:
                # Demo video processing
                demo_choice = input(f"\n🎬 Test video processing with loaded models? (y/n): ").lower().strip()
                if demo_choice == 'y':
                    demo_video_processing(manager)
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    
    return manager

def demo_video_processing(manager):
    """Demo video processing with loaded models"""
    print(f"\n🎬 VIDEO PROCESSING DEMO")
    print("-" * 30)
    
    # Check available video files
    video_dirs = ['videos', '../videos', '.']
    available_videos = []
    
    for video_dir in video_dirs:
        if os.path.exists(video_dir):
            for file in os.listdir(video_dir):
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    available_videos.append(os.path.join(video_dir, file))
    
    if available_videos:
        print(f"📁 Available videos:")
        for i, video in enumerate(available_videos[:5], 1):  # Show first 5
            print(f"   {i}. {video}")
        
        choice = input(f"Select video (1-{min(5, len(available_videos))}) or enter custom path: ").strip()
        
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_videos):
                video_path = available_videos[choice_idx]
            else:
                video_path = choice
        except ValueError:
            video_path = choice
    else:
        video_path = input("📁 Enter video file path: ").strip()
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    query = input("🔍 Enter search query (optional): ").strip()
    
    print(f"\n🔄 Processing video: {os.path.basename(video_path)}")
    if query:
        print(f"🔍 Query: {query}")
    
    try:
        results = manager.process_video_with_selected_models(video_path, query)
        
        print(f"\n🎉 PROCESSING COMPLETED!")
        print(f"📊 Results from {len(results['processing_results'])} models")
        
        # Show processing times
        if results.get('processing_time'):
            print(f"\n⏱️  Processing Times:")
            for model, time_sec in results['processing_time'].items():
                print(f"   {model}: {time_sec:.2f}s")
        
        # Show brief results
        print(f"\n📋 Brief Results:")
        for model, result in results['processing_results'].items():
            if 'error' in result:
                print(f"   ❌ {model}: {result['error']}")
            else:
                result_type = result.get('type', 'unknown')
                print(f"   ✅ {model}: {result_type}")
                
                if result_type == 'text_embedding':
                    shape = result.get('shape', 'unknown')
                    print(f"      Embedding shape: {shape}")
                elif result_type == 'video_features':
                    frames = len(result.get('processed_frames', []))
                    print(f"      Processed frames: {frames}")
                elif result_type == 'object_detection':
                    detections = len(result.get('detections', []))
                    print(f"      Detection frames: {detections}")
        
        # Ask if user wants to see detailed results
        detail_choice = input(f"\n📖 Show detailed results? (y/n): ").lower().strip()
        if detail_choice == 'y':
            print(f"\n📖 DETAILED RESULTS:")
            import json
            print(json.dumps(results, indent=2, default=str))
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")

def main():
    """Main demo function"""
    print_header()
    
    # Check TensorFlow Hub availability
    try:
        import tensorflow_hub as hub
        print("✅ TensorFlow Hub is available")
    except ImportError:
        print("❌ TensorFlow Hub not available. Please install:")
        print("   pip install tensorflow tensorflow-hub tensorflow-text opencv-python")
        return
    
    while True:
        print("\n🎬 DEMO OPTIONS:")
        print("1. 🤖 Model Analysis Demo (Quick)")
        print("2. 🎯 Interactive Model Selection (Full)")
        print("3. 📊 Model Status Overview")
        print("0. Exit")
        
        choice = input("\nSelect option (0-3): ").strip()
        
        if choice == '1':
            demo_model_analysis()
        elif choice == '2':
            demo_interactive_selection()
        elif choice == '3':
            manager = TensorFlowHubVideoManager()
            status = manager.get_model_status()
            print(f"\n📊 MODEL STATUS:")
            print(f"   TensorFlow Hub Available: {status['tensorflow_hub_available']}")
            print(f"   Total Models Available: {status['total_models_available']}")
            print(f"   Active Models: {len(status['active_models'])}")
            
            if status['active_models']:
                print(f"   Currently Loaded:")
                for model in status['active_models']:
                    print(f"     • {model}")
        elif choice == '0':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice, please try again.")

if __name__ == "__main__":
    main()
