# 🤖 AI Agents Guide - Hướng dẫn chi tiết các AI Agent

## 📋 Tổng quan AI Agents trong hệ thống

Hệ thống Enhanced AI Video Search tích hợp **4 loại AI Agents** chính để hỗ trợ tìm kiếm và phân tích video thông minh:

---

## 🧠 1. OpenAI GPT-4 Vision Agent

### **Chức năng chính:**
- **Phân tích hình ảnh chi tiết**: Mô tả nội dung frame video
- **Trả lời câu hỏi về hình ảnh**: Visual Question Answering (VQA)
- **Tối ưu query tìm kiếm**: Chuyển đổi ngôn ngữ tự nhiên thành query tối ưu

### **Cách sử dụng:**
```python
from ai_agent_manager import AIAgentManager, AgentConfig

# Cấu hình GPT-4 Vision
agent_config = AgentConfig(
    provider="openai",
    model="gpt-4-vision-preview",
    max_tokens=4000,
    temperature=0.1
)

manager = AIAgentManager()

# Phân tích frame video
result = manager.analyze_frame(
    image_path="frame_001.jpg",
    prompt="Mô tả chi tiết những gì bạn thấy trong hình này",
    config=agent_config
)

# Tối ưu query tìm kiếm
optimized_query = manager.generate_search_query(
    user_query="tìm cảnh người nói chuyện điện thoại",
    config=agent_config
)
```

### **Lợi ích:**
✅ **Độ chính xác cao** - Hiểu rõ ngữ cảnh và chi tiết  
✅ **Hỗ trợ tiếng Việt** - Xử lý câu hỏi tiếng Việt tự nhiên  
✅ **Reasoning ability** - Có khả năng suy luận phức tạp  
✅ **Multi-modal** - Kết hợp text và image analysis  

### **Khi nào sử dụng:**
- Cần phân tích chi tiết nội dung frame
- Tìm kiếm phức tạp với ngữ cảnh
- Trả lời câu hỏi về nội dung video
- Tạo mô tả tự động cho video

---

## 🎯 2. Anthropic Claude Agent

### **Chức năng chính:**
- **Xử lý ngôn ngữ tự nhiên**: Hiểu ý định người dùng
- **Tối ưu query**: Chuyển đổi thành search terms hiệu quả
- **Phân tích ngữ cảnh**: Hiểu context và intent

### **Cách sử dụng:**
```python
# Cấu hình Claude
agent_config = AgentConfig(
    provider="anthropic", 
    model="claude-3-sonnet-20240229",
    max_tokens=4000
)

# Tối ưu query tìm kiếm thông minh
optimized_query = manager.generate_search_query(
    user_query="tìm những cảnh có người đàn ông mặc vest trong phòng họp",
    config=agent_config
)

# Phân tích ý định tìm kiếm
intent_analysis = manager.analyze_search_intent(
    query="người phụ nữ đang thuyết trình về dự án",
    config=agent_config
)
```

### **Lợi ích:**
✅ **Xử lý ngôn ngữ tự nhiên tốt** - Hiểu câu hỏi phức tạp  
✅ **Reasoning mạnh** - Phân tích ý định chính xác  
✅ **Context awareness** - Hiểu ngữ cảnh tìm kiếm  
✅ **Optimization** - Tối ưu query cho kết quả tốt nhất  

### **Khi nào sử dụng:**
- Query tìm kiếm phức tạp, nhiều điều kiện
- Cần hiểu ý định tìm kiếm của người dùng
- Tối ưu performance tìm kiếm
- Xử lý câu hỏi mơ hồ

---

## 🔬 3. Local BLIP Models Agent

### **Chức năng chính:**
- **Image Captioning**: Tạo caption tự động cho frame
- **Visual Question Answering**: Trả lời câu hỏi về hình ảnh
- **Hoạt động offline**: Không cần API key

### **Cách sử dụng:**
```python
# Cấu hình BLIP model (offline)
agent_config = AgentConfig(
    provider="local",
    model="Salesforce/blip-image-captioning-base"
)

# Tạo caption cho frame
caption = manager.analyze_frame(
    image_path="frame_002.jpg", 
    prompt="Describe this image",
    config=agent_config
)

# Visual Question Answering
answer = manager.visual_qa(
    image_path="frame_002.jpg",
    question="What is the person doing?",
    config=agent_config
)
```

### **Lợi ích:**
✅ **Hoạt động offline** - Không cần internet hoặc API key  
✅ **Tốc độ cao** - Xử lý local trên GPU  
✅ **Privacy** - Dữ liệu không gửi ra ngoài  
✅ **Cost-effective** - Không tốn phí API  

### **Khi nào sử dụng:**
- Xử lý batch lớn frames
- Môi trường bảo mật, không kết nối internet
- Tạo metadata tự động cho video
- Chi phí API quan trọng

---

## 🎭 4. Hybrid Model Manager

### **Chức năng chính:**
- **Kết hợp multiple models**: CLIP + BLIP + Vision models
- **Intelligent routing**: Chọn model phù hợp cho từng task
- **Performance optimization**: Tối ưu GPU memory và speed

### **Cách sử dụng:**
```python
from enhanced_hybrid_manager import EnhancedHybridModelManager

manager = EnhancedHybridModelManager()

# Text-to-image search với multiple models
results = manager.search_by_text(
    query="người phụ nữ đang nói chuyện điện thoại",
    top_k=10,
    use_models=["clip", "chinese-clip", "blip"]
)

# Image similarity với ensemble approach
similar_frames = manager.search_by_image(
    image_path="query_frame.jpg",
    top_k=5,
    similarity_threshold=0.8,
    ensemble=True  # Kết hợp kết quả từ nhiều model
)

# Advanced search với filters
filtered_results = manager.advanced_search(
    query="meeting room presentation",
    filters={
        "video_name": ["business_meeting.mp4"],
        "timestamp_range": (100, 500),
        "min_confidence": 0.7
    },
    use_agents=True  # Sử dụng AI agents để optimize
)
```

### **Lợi ích:**
✅ **Best of all worlds** - Kết hợp ưu điểm của nhiều model  
✅ **Intelligent selection** - Tự động chọn model tối ưu  
✅ **High accuracy** - Ensemble approach cho độ chính xác cao  
✅ **GPU optimized** - Tối ưu memory và performance  

---

## 🎯 Tích hợp AI Agents vào workflow

### **1. Search Workflow với AI Agents**

```python
# Bước 1: User input
user_query = "tìm cảnh có người đàn ông đang thuyết trình"

# Bước 2: Claude tối ưu query
optimized_query = claude_agent.optimize_query(user_query)
# Output: "man giving presentation business meeting"

# Bước 3: Hybrid manager tìm kiếm
results = hybrid_manager.search_by_text(optimized_query, top_k=20)

# Bước 4: GPT-4 Vision re-rank results
for result in results:
    confidence = gpt4_vision.analyze_relevance(
        image=result.frame_path,
        query=user_query
    )
    result.ai_confidence = confidence

# Bước 5: Trả về kết quả được AI optimize
final_results = sorted(results, key=lambda x: x.ai_confidence, reverse=True)[:10]
```

### **2. Auto-tagging với AI Agents**

```python
# Tự động tạo tags cho video mới
def auto_tag_video(video_path):
    frames = extract_key_frames(video_path)
    
    for frame in frames:
        # BLIP tạo caption cơ bản
        basic_caption = blip_agent.generate_caption(frame)
        
        # GPT-4 Vision phân tích chi tiết
        detailed_analysis = gpt4_vision.analyze_frame(
            frame, 
            prompt="Describe the scene, objects, people, and activities in detail"
        )
        
        # Claude extract keywords
        keywords = claude_agent.extract_keywords(detailed_analysis)
        
        # Lưu metadata
        save_frame_metadata(frame, {
            'basic_caption': basic_caption,
            'detailed_analysis': detailed_analysis,
            'keywords': keywords,
            'ai_processed': True
        })
```

---

## 💡 Các tình huống sử dụng thực tế

### **🎬 Tìm kiếm nội dung video**
**Tình huống**: Tìm cảnh cụ thể trong video dài  
**Agents sử dụng**: Claude (optimize query) + Hybrid Manager (search) + GPT-4 (re-rank)  
**Lợi ích**: Tìm chính xác, hiểu ngữ cảnh, kết quả relevance cao  

### **📊 Phân tích nội dung tự động**
**Tình huống**: Tạo metadata cho thư viện video lớn  
**Agents sử dụng**: BLIP (batch captioning) + GPT-4 (detailed analysis)  
**Lợi ích**: Tự động hóa, chi phí thấp, quality cao  

### **🔍 Visual Question Answering**
**Tình huống**: "Có bao nhiêu người trong cảnh này?"  
**Agents sử dụng**: GPT-4 Vision (primary) + BLIP (backup)  
**Lợi ích**: Trả lời chính xác câu hỏi phức tạp về hình ảnh  

### **🎯 Content Moderation**
**Tình huống**: Kiểm tra nội dung không phù hợp  
**Agents sử dụng**: Multiple models với voting mechanism  
**Lợi ích**: Độ chính xác cao, giảm false positive/negative  

---

## ⚙️ Cấu hình và Tối ưu

### **API Keys Setup**
```bash
# Tạo file .env
cp .env.example .env

# Thêm API keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

### **Performance Tuning**
```python
# Tối ưu cho GPU memory
config = {
    'batch_size': 16,  # Giảm nếu GPU memory thấp
    'mixed_precision': True,
    'cache_models': True,
    'max_concurrent_requests': 4
}

# Tối ưu cho cost
config = {
    'use_local_first': True,  # Ưu tiên BLIP trước API
    'api_fallback': True,     # Chỉ dùng API khi cần
    'cache_api_results': True # Cache để tránh duplicate calls
}
```

---

## 🚀 Kết luận

AI Agents trong hệ thống giúp:

1. **🎯 Tăng độ chính xác tìm kiếm** - Hiểu ý định và ngữ cảnh
2. **⚡ Tối ưu performance** - Intelligent routing và caching  
3. **💰 Giảm chi phí** - Kết hợp local models và cloud APIs
4. **🔧 Tự động hóa** - Auto-tagging và content analysis
5. **🌍 Đa ngôn ngữ** - Hỗ trợ tiếng Việt và English tự nhiên

**Khuyến nghị**: Bắt đầu với BLIP (local) cho basic features, sau đó thêm Claude và GPT-4 Vision cho advanced capabilities.
