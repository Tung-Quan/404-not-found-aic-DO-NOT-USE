"""
🔍 Interactive Search Test cho 2 Video
Test tìm kiếm với query mô tả để xem có thể tìm đúng video từ frame không
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from simplified_search_engine import SimplifiedSearchEngine
from pathlib import Path

def main():
    print("🚀 Khởi động Interactive Search cho 2 Video")
    print("=" * 60)
    
    # Initialize engine
    print("🎯 Đang khởi tạo engine...")
    engine = SimplifiedSearchEngine()
    
    print("📂 Đang load index...")
    engine.load_index("index")
    
    print(f"✅ Đã load {len(engine.frame_paths)} frames")
    
    # Show available videos
    videos = {}
    for path in engine.frame_paths:
        video = Path(path).parent.name
        videos[video] = videos.get(video, 0) + 1
    
    print("\n📹 Video có sẵn:")
    for video, count in videos.items():
        print(f"  📁 {video}: {count} frames")
    
    # Test queries về programming/coding
    test_queries = [
        "programming tutorial",
        "code editor", 
        "software development",
        "computer screen",
        "coding lesson",
        "typing code",
        "learning programming",
        "development environment"
    ]
    
    print(f"\n🔍 TEST CÁC QUERY TỰ ĐỘNG:")
    print("-" * 40)
    
    for query in test_queries:
        print(f"\n🔎 Query: '{query}'")
        try:
            results = engine.search(query, top_k=3)
            
            if results:
                print(f"✅ Tìm thấy {len(results)} kết quả:")
                for i, result in enumerate(results, 1):
                    path = result['path']
                    score = result['score']
                    video = Path(path).parent.name
                    frame = Path(path).name
                    print(f"   {i}. 📹 {video} -> {frame} (score: {score:.3f})")
            else:
                print("❌ Không tìm thấy kết quả")
                
        except Exception as e:
            print(f"❌ Lỗi search: {e}")
    
    # Interactive search
    print(f"\n🎯 CHẾ ĐỘ TÌM KIẾM TƯƠNG TÁC:")
    print("-" * 40)
    print("Nhập query để tìm kiếm (gõ 'quit' để thoát)")
    
    while True:
        try:
            query = input("\n🔍 Nhập query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q', 'thoat']:
                break
                
            if not query:
                continue
            
            print(f"🔎 Đang tìm kiếm: '{query}'...")
            results = engine.search(query, top_k=5)
            
            if results:
                print(f"✅ Tìm thấy {len(results)} kết quả:")
                for i, result in enumerate(results, 1):
                    path = result['path']
                    score = result['score']
                    video = Path(path).parent.name
                    frame = Path(path).name
                    print(f"   {i}. 📹 {video} -> {frame} (score: {score:.3f})")
                    
                # Show detailed info for top result
                if results:
                    top_result = results[0]
                    print(f"\n🎯 Kết quả tốt nhất:")
                    print(f"   📂 Path: {top_result['path']}")
                    print(f"   📊 Score: {top_result['score']:.3f}")
                    print(f"   📝 Has text: {top_result['has_text']}")
                    if top_result['extracted_text']:
                        print(f"   💬 Text: {top_result['extracted_text'][:100]}...")
            else:
                print("❌ Không tìm thấy kết quả nào")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Lỗi: {e}")
    
    print("\n👋 Kết thúc test search!")

if __name__ == "__main__":
    main()
