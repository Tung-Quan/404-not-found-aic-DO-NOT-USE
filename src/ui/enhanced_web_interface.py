"""
🎮 ENHANCED VIDEO SEARCH WEB INTERFACE
=====================================
Interactive web interface for TensorFlow Hub model selection and video processing
"""

import streamlit as st
import requests
import json
import os
import time
from typing import Dict, List
import pandas as pd
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Configuration
API_BASE_URL = "http://localhost:8000"

def init_session_state():
    """Initialize session state variables"""
    if 'api_status' not in st.session_state:
        st.session_state.api_status = None
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = []
    if 'model_recommendations' not in st.session_state:
        st.session_state.model_recommendations = None
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = None

def check_api_status():
    """Check if API is running and get status"""
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def get_model_recommendations(user_intent: str, max_memory: int, priority: str):
    """Get model recommendations from API"""
    try:
        payload = {
            "user_intent": user_intent,
            "max_memory_mb": max_memory,
            "processing_priority": priority
        }
        
        response = requests.post(f"{API_BASE_URL}/analyze_models", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None

def load_models(model_names: List[str]):
    """Load selected models via API"""
    try:
        response = requests.post(f"{API_BASE_URL}/load_selected_models", json=model_names)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None

def process_video(video_path: str, query: str, selected_models: List[str]):
    """Process video with selected models"""
    try:
        payload = {
            "video_path": video_path,
            "query": query,
            "selected_models": selected_models
        }
        
        response = requests.post(f"{API_BASE_URL}/process_video", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="🎥 Enhanced Video Search",
        page_icon="🎥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    # Header
    st.title("🎥 Enhanced Video Search with TensorFlow Hub")
    st.markdown("---")
    
    # Sidebar for API status and settings
    with st.sidebar:
        st.header("🔧 System Status")
        
        # API Status Check
        if st.button("🔄 Check API Status"):
            with st.spinner("Checking API..."):
                st.session_state.api_status = check_api_status()
        
        if st.session_state.api_status:
            st.success("✅ API Connected")
            
            # Show basic status info
            status = st.session_state.api_status
            st.write(f"**Total Frames:** {status.get('total_frames', 'Unknown')}")
            st.write(f"**TensorFlow Hub:** {'✅' if status.get('tensorflow_hub_available') else '❌'}")
            st.write(f"**Enhanced Processor:** {'✅' if status.get('enhanced_video_processor_available') else '❌'}")
            
            # Show enhanced models if available
            if 'enhanced_models' in status:
                enhanced = status['enhanced_models']
                st.write(f"**Available Models:** {enhanced.get('total_models_available', 0)}")
                st.write(f"**Active Models:** {len(enhanced.get('active_models', []))}")
                
                if enhanced.get('active_models'):
                    st.write("**Currently Loaded:**")
                    for model in enhanced['active_models']:
                        st.write(f"   • {model}")
        else:
            st.error("❌ API Not Connected")
            st.write("Make sure the API server is running:")
            st.code("python api/app.py")
            return
    
    # Main content area
    if not st.session_state.api_status:
        st.warning("⚠️ Please check API status in the sidebar first.")
        return
    
    # Check if enhanced processor is available
    if not st.session_state.api_status.get('enhanced_video_processor_available'):
        st.error("❌ Enhanced Video Processor not available. Please check the API setup.")
        return
    
    # Tabs for different functionality
    tab1, tab2, tab3, tab4 = st.tabs(["🤖 Model Selection", "🎥 Video Processing", "📊 Results", "🔍 Standard Search"])
    
    with tab1:
        st.header("🤖 Intelligent Model Selection")
        st.write("Describe what you want to do, and we'll recommend the best TensorFlow Hub models.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # User intent input
            user_intent = st.text_area(
                "📝 What do you want to do with videos?",
                placeholder="e.g., 'Find action sequences in cooking videos', 'Detect objects in tutorial videos', 'Search for specific activities'",
                height=100
            )
            
            # Processing priority
            priority = st.selectbox(
                "⚡ Processing Priority",
                ["lightweight", "balanced", "high_accuracy"],
                index=1,
                help="Lightweight: Fast & low memory, Balanced: Good performance/memory ratio, High Accuracy: Best quality but more memory"
            )
        
        with col2:
            # Memory constraint
            max_memory = st.number_input(
                "💾 Max Memory (MB)",
                min_value=500,
                max_value=8000,
                value=2000,
                step=500,
                help="Maximum memory usage for models"
            )
            
            # Get recommendations button
            if st.button("🔍 Analyze Requirements", type="primary"):
                if user_intent.strip():
                    with st.spinner("Analyzing requirements..."):
                        st.session_state.model_recommendations = get_model_recommendations(
                            user_intent, max_memory, priority
                        )
                else:
                    st.warning("Please describe what you want to do first.")
        
        # Show recommendations
        if st.session_state.model_recommendations:
            recs = st.session_state.model_recommendations
            
            st.markdown("### 📋 Recommendations")
            
            # Show suggested models
            if recs.get('suggested_models'):
                st.success(f"**Suggested Configuration ({priority}):**")
                
                # Create a nice display of suggested models
                models_df = []
                for model_name in recs['suggested_models']:
                    # Get model details from API status
                    enhanced_models = st.session_state.api_status.get('enhanced_models', {})
                    model_details = enhanced_models.get('model_details', {}).get(model_name, {})
                    
                    models_df.append({
                        'Model': model_details.get('name', model_name),
                        'Type': model_details.get('type', 'unknown'),
                        'Memory': model_details.get('memory_usage', 'unknown'),
                        'Speed': model_details.get('processing_speed', 'unknown'),
                        'Description': model_details.get('description', 'No description')
                    })
                
                if models_df:
                    st.dataframe(pd.DataFrame(models_df), use_container_width=True)
                
                st.info(f"**Estimated Memory Usage:** {recs.get('estimated_memory_usage', 'Unknown')}")
                
                # Model selection
                st.markdown("### ✅ Select Models to Load")
                selected_models = st.multiselect(
                    "Choose models to load:",
                    recs['suggested_models'],
                    default=recs['suggested_models'],
                    help="You can modify the selection based on your needs"
                )
                
                # Show overlaps if any
                if recs.get('overlaps_detected'):
                    st.warning("⚠️ **Overlapping Functionality Detected**")
                    for config_type, overlaps in recs['overlaps_detected'].items():
                        st.write(f"**{config_type.title()} Configuration:**")
                        for model, overlapping_with in overlaps.items():
                            st.write(f"   • {model} overlaps with {overlapping_with}")
                
                # Load models button
                if st.button("🚀 Load Selected Models", type="secondary"):
                    if selected_models:
                        with st.spinner("Loading models... This may take a few minutes."):
                            load_result = load_models(selected_models)
                            
                        if load_result and load_result.get('success'):
                            st.success(f"✅ {load_result['message']}")
                            st.session_state.selected_models = load_result.get('active_models', [])
                            
                            # Show load results
                            for model, success in load_result.get('results', {}).items():
                                if success:
                                    st.write(f"   ✅ {model}")
                                else:
                                    st.write(f"   ❌ {model}")
                        else:
                            st.error("❌ Failed to load models")
                    else:
                        st.warning("Please select at least one model")
            else:
                st.warning("No model recommendations available. Try describing your requirements differently.")
    
    with tab2:
        st.header("🎥 Video Processing")
        
        if not st.session_state.selected_models:
            st.warning("⚠️ Please select and load models in the Model Selection tab first.")
        else:
            st.success(f"✅ Active Models: {', '.join(st.session_state.selected_models)}")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Video path input
                video_path = st.text_input(
                    "📁 Video File Path",
                    placeholder="e.g., videos/sample.mp4",
                    help="Enter the path to your video file"
                )
                
                # Search query
                search_query = st.text_input(
                    "🔍 Search Query (Optional)",
                    placeholder="e.g., 'cooking action', 'person walking'",
                    help="Describe what you're looking for in the video"
                )
            
            with col2:
                # Processing options
                st.write("**Processing Options:**")
                process_text = st.checkbox("Process Text Query", value=bool(search_query))
                process_video = st.checkbox("Process Video Frames", value=True)
                process_audio = st.checkbox("Process Audio (if available)", value=False, disabled=True)
                
                # Processing button
                if st.button("🔄 Process Video", type="primary"):
                    if video_path.strip():
                        if os.path.exists(video_path):
                            with st.spinner("Processing video... This may take several minutes."):
                                st.session_state.processing_results = process_video(
                                    video_path, search_query, st.session_state.selected_models
                                )
                        else:
                            st.error(f"❌ Video file not found: {video_path}")
                    else:
                        st.warning("Please enter a video file path")
    
    with tab3:
        st.header("📊 Processing Results")
        
        if st.session_state.processing_results:
            results = st.session_state.processing_results
            
            # Basic info
            st.success(f"✅ Processed: {os.path.basename(results.get('video_path', ''))}")
            if results.get('query'):
                st.info(f"🔍 Query: {results['query']}")
            
            # Active models
            if results.get('active_models'):
                st.write(f"**Active Models:** {', '.join(results['active_models'])}")
            
            # Processing times
            if results.get('processing_time'):
                st.markdown("### ⏱️ Processing Times")
                time_df = []
                for model, time_sec in results['processing_time'].items():
                    time_df.append({
                        'Model': model,
                        'Processing Time': f"{time_sec:.2f}s"
                    })
                st.dataframe(pd.DataFrame(time_df), use_container_width=True)
            
            # Processing results
            if results.get('processing_results'):
                st.markdown("### 🔬 Detailed Results")
                
                for model_name, model_results in results['processing_results'].items():
                    with st.expander(f"📊 {model_name} Results"):
                        if 'error' in model_results:
                            st.error(f"❌ Error: {model_results['error']}")
                        else:
                            st.json(model_results)
            
            # Combined features (if available)
            if results.get('combined_features'):
                st.markdown("### 🔗 Combined Features")
                st.json(results['combined_features'])
        
        else:
            st.info("📋 Process a video in the Video Processing tab to see results here.")
    
    with tab4:
        st.header("🔍 Standard Search")
        st.write("Use the existing search functionality for quick queries.")
        
        # Simple search interface
        query = st.text_input("🔍 Search Query", placeholder="Enter your search query...")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            search_type = st.selectbox("Search Type", ["enhanced_search", "search", "search_frames"])
        with col2:
            top_k = st.number_input("Top Results", min_value=1, max_value=100, value=10)
        with col3:
            if st.button("🔍 Search", type="primary"):
                if query.strip():
                    try:
                        url = f"{API_BASE_URL}/{search_type}"
                        params = {"q": query, "topk_frames": top_k} if search_type == "enhanced_search" else {"q": query}
                        
                        response = requests.get(url, params=params)
                        if response.status_code == 200:
                            search_results = response.json()
                            
                            st.success(f"✅ Found {len(search_results.get('results', []))} results")
                            
                            # Display results
                            for i, result in enumerate(search_results.get('results', [])[:top_k], 1):
                                with st.expander(f"Result {i} - Score: {result.get('score', 0):.3f}"):
                                    st.write(f"**Video ID:** {result.get('video_id', 'Unknown')}")
                                    st.write(f"**Timestamp:** {result.get('timestamp', 0):.1f}s")
                                    if result.get('frame_path'):
                                        st.write(f"**Frame:** {result['frame_path']}")
                        else:
                            st.error(f"Search failed: {response.status_code}")
                    except Exception as e:
                        st.error(f"Search error: {e}")
                else:
                    st.warning("Please enter a search query")

if __name__ == "__main__":
    main()
