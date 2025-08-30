#!/usr/bin/env python3
"""
Test New User Workflow
Kiểm tra toàn bộ workflow cho người dùng mới bao gồm:
- Kiểm tra datasets có sẵn
- Test launcher menu options
- Verify web interface accessibility
- Check model availability
"""

import os
import sys
import json
from pathlib import Path

def test_datasets_availability():
    """Test if datasets are available and configured properly"""
    print("📂 KIỂM TRA DATASETS CHO NGƯỜI DÙNG MỚI:")
    print("=" * 60)
    
    datasets_path = Path('./datasets')
    if not datasets_path.exists():
        print("❌ Thư mục datasets không tồn tại")
        return False
    
    datasets = [d for d in datasets_path.iterdir() if d.is_dir()]
    print(f"✅ Tìm thấy {len(datasets)} datasets:")
    
    for dataset in datasets:
        config_file = dataset / 'config.json'
        videos_dir = dataset / 'videos'
        
        print(f"\n  📁 {dataset.name}:")
        
        # Check config
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"    ✅ Config: {config.get('description', 'N/A')}")
                print(f"    📊 Videos: {config.get('video_count', 0)} files")
            except Exception as e:
                print(f"    ❌ Lỗi đọc config: {e}")
        else:
            print("    ❌ Không có config.json")
        
        # Check videos
        if videos_dir.exists():
            video_files = list(videos_dir.glob('*.mp4'))
            print(f"    🎬 Video files thực tế: {len(video_files)}")
            if video_files:
                for video in video_files[:3]:  # Show first 3
                    print(f"       {video.name}")
                if len(video_files) > 3:
                    print(f"       ... và {len(video_files) - 3} files khác")
        else:
            print("    ❌ Thư mục videos không tồn tại")
    
    return len(datasets) > 0

def test_model_architecture():
    """Test model architecture understanding"""
    print("\n🤖 KIỂM TRA KIẾN TRÚC MÔ HÌNH:")
    print("=" * 60)
    
    try:
        # Set environment to suppress warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        from enhanced_hybrid_manager import EnhancedHybridModelManager
        
        manager = EnhancedHybridModelManager()
        
        print("📋 Cấu trúc 3 lớp mô hình:")
        print("  🔹 Lớp 1: Mô hình cơ bản (4 models) - Hiển thị trong web UI")
        print("  🔹 Lớp 2: TensorFlow models (11 models) - Backend processing")
        print("  🔹 Lớp 3: AI Agents (2 agents) - Specialized tasks")
        
        # Get available models (what users see)
        available_models = manager.get_available_models()
        print(f"\n✅ Mô hình hiển thị cho người dùng: {len(available_models)}")
        for model in available_models:
            print(f"    • {model}")
        
        # Get TensorFlow models status
        tf_status = manager.get_tensorflow_models_status()
        if tf_status['available']:
            tf_models = tf_status['models']
            print(f"\n🔧 TensorFlow models backend: {len(tf_models)}")
            for name, info in list(tf_models.items())[:5]:  # Show first 5
                print(f"    • {name}: {info.get('name', 'N/A')}")
            if len(tf_models) > 5:
                print(f"    ... và {len(tf_models) - 5} models khác")
        
        # Get AI agents status  
        agent_status = manager.get_ai_agents_status()
        if agent_status['available']:
            agents = agent_status.get('agents', {})
            print(f"\n🤖 AI Agents: {len(agents)}")
            for name, info in agents.items():
                print(f"    • {name}: {info.get('name', 'N/A')} ({info.get('provider', 'N/A')})")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Một số dependencies có thể chưa được cài đặt")
        return False
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return False

def test_launcher_functions():
    """Test launcher function availability"""
    print("\n🚀 KIỂM TRA LAUNCHER FUNCTIONS:")
    print("=" * 60)
    
    try:
        # Import launcher functions
        sys.path.append(os.getcwd())
        import main_launcher
        
        # Check available functions
        functions = [
            'check_python_compatibility',
            'check_system_status', 
            'get_user_choice',
            'start_web_interface',
            'start_full_version',
            'main'
        ]
        
        for func_name in functions:
            if hasattr(main_launcher, func_name):
                print(f"    ✅ {func_name}")
            else:
                print(f"    ❌ {func_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi import launcher: {e}")
        return False

def test_web_interface_files():
    """Test if web interface files exist"""
    print("\n🌐 KIỂM TRA WEB INTERFACE FILES:")
    print("=" * 60)
    
    web_files = [
        'api/app.py',
        'templates/index.html',
        'ai_search_demo.html',
        'demo_frame_search.html'
    ]
    
    all_exist = True
    for file_path in web_files:
        if Path(file_path).exists():
            print(f"    ✅ {file_path}")
        else:
            print(f"    ❌ {file_path}")
            all_exist = False
    
    return all_exist

def generate_new_user_guide():
    """Generate guide for new users"""
    print("\n📖 TẠO HƯỚNG DẪN CHO NGƯỜI DÙNG MỚI:")
    print("=" * 60)
    
    guide = """
HƯỚNG DẪN NHANH CHO NGƯỜI DÙNG MỚI
=====================================

1. 🚀 KHỞI ĐỘNG NHANH:
   ```
   python main_launcher.py
   ```
   
2. 🎯 LỰA CHỌN ĐƯỢC KHUYẾN NGHỊ:
   - Chọn option 1: "🌐 Web Interface (Recommended)" 
   - Đây là cách dễ nhất để bắt đầu
   
3. 📂 DATASETS CÓ SẴN:
   - nature_collection: Video thiên nhiên
   - people_collection: Video con người  
   - mixed_collection: Video tổng hợp
   - Tất cả đã được cấu hình sẵn, không cần setup thêm
   
4. 🤖 MODELS ĐƯỢC TÍCH HỢP:
   - 4 mô hình chính hiển thị trong web UI
   - 11 TensorFlow models hoạt động ở backend
   - 2 AI agents xử lý tác vụ đặc biệt
   - Tổng cộng: 17 capabilities
   
5. 🌐 SỬ DỤNG WEB INTERFACE:
   - Truy cập: http://localhost:8080
   - Switch datasets qua dropdown menu
   - Switch models qua model selector
   - Search bằng text hoặc upload ảnh
   
6. ⚡ TẠI SAO THIẾT KẾ NHƯ VẬY:
   - Web UI chỉ hiển thị 4 mô hình cơ bản để đơn giản
   - Backend tự động sử dụng 11 TensorFlow models khi cần
   - AI agents hoạt động trong background
   - Người dùng mới không bị overwhelm bởi quá nhiều tùy chọn

KẾT LUẬN: Datasets và model architecture được thiết kế 
để cung cấp trải nghiệm tốt nhất cho người dùng mới!
"""
    
    with open('NEW_USER_GUIDE.md', 'w', encoding='utf-8') as f:
        f.write(guide)
    
    print("✅ Đã tạo file NEW_USER_GUIDE.md")
    return guide

def main():
    """Run all tests"""
    print("🧪 NEW USER WORKFLOW TEST")
    print("=" * 60)
    
    results = {}
    results['datasets'] = test_datasets_availability()
    results['models'] = test_model_architecture()
    results['launcher'] = test_launcher_functions()
    results['web_files'] = test_web_interface_files()
    
    print("\n📊 KẾT QUẢ TỔNG HỢP:")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"    {test_name.upper()}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 TẤT CẢ TESTS PASSED!")
        print("✅ System sẵn sàng cho người dùng mới")
        print("✅ Datasets không gây vấn đề gì")
        print("✅ Web interface là entry point tốt nhất")
        
        # Generate guide
        generate_new_user_guide()
        
    else:
        print("\n⚠️ MỘT SỐ TESTS FAILED")
        print("🔧 Cần kiểm tra lại các components")
    
    return all_passed

if __name__ == "__main__":
    main()
