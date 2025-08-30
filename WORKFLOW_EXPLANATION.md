# 🎯 **Flow Làm Việc Chi Tiết của Hệ Thống AI Video Search**

## 🏗️ **Kiến Trúc 3 Lớp và Workflow**

### **📋 Tổng Quan Hệ Thống**

```
🤖 Enhanced AI Video Search System (3-Layer Architecture)

┌─────────────────────────────────────────────────────────────────┐
│                    🌐 LAYER 1: WEB INTERFACE                    │
│                     (4 Core Models - Frontend)                  │
├─────────────────────────────────────────────────────────────────┤
│  🚀 CLIP ViT Base  │  🔥 CLIP ViT Large  │  🌏 Chinese CLIP    │
│  📝 Sentence Transformers                                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                🔧 LAYER 2: TENSORFLOW BACKEND                   │
│                   (11 Specialized Models)                       │
├─────────────────────────────────────────────────────────────────┤
│  📱 MobileNet V2   │  🎯 Inception V3    │  🏗️ ResNet 50       │
│  ⚡ EfficientNet   │  🌐 USE v4/Large    │  🔍 SSD MobileNet    │
│  🎨 Faster R-CNN   │  📊 ImageNet        │  🎭 BiT ResNet      │
│  🌍 USE Multilingual │  🔧 Custom Models                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                🤖 LAYER 3: AI AGENTS                           │
│                (2 Premium + 1 Local Agent)                     │
├─────────────────────────────────────────────────────────────────┤
│  🤖 OpenAI GPT-4   │  🧠 Anthropic       │  🏠 Local BLIP      │
│     Vision API     │     Claude 3        │     Models          │
│  (Requires API)    │  (Requires API)     │  (Free)             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 **LAYER 1: Web Interface Flow (4 Mô Hình Chính)**

### **🎯 Mô Hình Nào Được Sử Dụng Khi Nào**

#### **1. 🚀 CLIP ViT Base (Mặc Định)**
```python
# File: web_interface.py
@app.on_event("startup")
async def startup_event():
    # Auto-load CLIP ViT Base làm mô hình mặc định
    success = model_manager.load_model("clip_vit_base")
    search_engine.set_active_model("vision_language", "clip_vit_base")
    
    # Build embeddings cho tìm kiếm ngay lập tức
    embeddings_success = search_engine.build_embeddings_index()
```

**Flow Hoạt Động:**
1. **Khởi tạo**: Tự động load khi start web interface
2. **Xử lý query**: User nhập "person coding" → CLIP encode text
3. **So sánh**: Tính similarity với 3,801 frame embeddings
4. **Trả kết quả**: Top 10 frames với similarity scores

#### **2. 🔥 CLIP ViT Large (Chất Lượng Cao)**
```python
# Khi user chọn model CLIP Large
@app.post("/api/models/switch")
async def switch_model(request: Request):
    if model_id == "clip_vit_large":
        # Load model mới
        success = model_manager.set_vision_language_model("CLIP ViT Large")
        
        # Rebuild toàn bộ embeddings với model mới
        search_engine.build_embeddings_index()  # 3,801 frames
```

**Flow Hoạt Động:**
1. **Switching**: User chọn CLIP Large trong dropdown
2. **Reload Model**: Unload CLIP Base, load CLIP Large
3. **Rebuild Index**: Xử lý lại 3,801 frames (batch 32, GPU accelerated)
4. **Update Search**: Search engine sử dụng embeddings mới

#### **3. 🌏 Chinese CLIP (Tối Ưu Tiếng Việt)**
```python
# Optimized cho Vietnamese queries
if model_id == "chinese_clip":
    success = model_manager.set_vision_language_model("Chinese CLIP")
    
# Xử lý query tiếng Việt
query = "tìm người đàn ông"  # Vietnamese input
chinese_clip_embeddings = model.encode(query)  # Tối ưu cho tiếng Việt
```

**Flow Hoạt Động:**
1. **Language Detection**: Detect Vietnamese/Chinese text
2. **Optimized Encoding**: Chinese CLIP xử lý tốt hơn tiếng Việt
3. **Cross-Language**: Hiểu context văn hóa Á Đông
4. **Better Results**: Accuracy cao hơn cho Vietnamese queries

#### **4. 📝 Sentence Transformers (Text-Only)**
```python
# Pure text similarity
if model_id == "sentence_transformers":
    text_embeddings = sentence_transformer.encode(query)
    # Chỉ so sánh text metadata, không xử lý hình ảnh
```

**Flow Hoạt Động:**
1. **Text-Only**: Chỉ xử lý text, bỏ qua visual features
2. **Fast Processing**: Không cần GPU cho image processing
3. **Metadata Search**: Tìm kiếm dựa trên frame descriptions
4. **Ultra Fast**: Tốc độ nhanh nhất trong 4 mô hình

---

## 🔧 **LAYER 2: TensorFlow Backend Flow (11 Mô Hình)**

### **🎭 Cách 11 Mô Hình TensorFlow Hoạt Động**

#### **🎯 Image Feature Extraction Models (4 models)**

```python
# File: tensorflow_model_manager.py
class TensorFlowModelManager:
    def _setup_tensorflow_models(self):
        # 1. MobileNet V2 - Lightweight
        self.configs["mobilenet_v2"] = TensorFlowModelConfig(
            url="https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
            input_shape=(224, 224, 3),
            capabilities=["image_embedding", "feature_extraction"]
        )
        
        # 2. Inception V3 - Balanced
        self.configs["inception_v3"] = TensorFlowModelConfig(
            url="https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4",
            input_shape=(299, 299, 3),
            capabilities=["image_embedding", "feature_extraction"]
        )
        
        # 3. ResNet-50 - Deep Learning
        self.configs["resnet50"] = TensorFlowModelConfig(
            url="https://tfhub.dev/tensorflow/resnet_50/feature_vector/1",
            capabilities=["image_embedding", "feature_extraction"]
        )
        
        # 4. EfficientNet B0 - State-of-art
        self.configs["efficientnet_b0"] = TensorFlowModelConfig(
            url="https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1",
            capabilities=["image_embedding", "feature_extraction"]
        )
```

**Khi Nào Được Sử Dụng:**
- **Background Processing**: Chạy song song với Layer 1
- **Advanced Analysis**: Khi cần feature extraction chuyên sâu
- **Batch Processing**: Xử lý hàng loạt frames với different perspectives

#### **🌐 Text Embedding Models (3 models)**

```python
# 5. Universal Sentence Encoder v4
self.configs["universal_sentence_encoder"] = TensorFlowModelConfig(
    url="https://tfhub.dev/google/universal-sentence-encoder/4",
    capabilities=["text_embedding", "semantic_search"]
)

# 6. USE Multilingual
self.configs["use_multilingual"] = TensorFlowModelConfig(
    url="https://tfhub.dev/google/universal-sentence-encoder-multilingual/3",
    capabilities=["text_embedding", "multilingual", "semantic_search"]
)

# 7. USE Large
self.configs["use_large"] = TensorFlowModelConfig(
    url="https://tfhub.dev/google/universal-sentence-encoder-large/5",
    capabilities=["text_embedding", "semantic_search", "high_quality"]
)
```

**Flow Hoạt Động:**
```python
# Backend text processing workflow
def encode_text_tensorflow(self, text: str, model_key: str = "universal_sentence_encoder"):
    # 1. Load TensorFlow model
    model = hub.load(config.url)
    
    # 2. Preprocess text
    processed_text = self._preprocess_text(text)
    
    # 3. Generate embeddings
    embeddings = model([processed_text])
    
    # 4. Return numpy array
    return embeddings.numpy()
```

#### **🔍 Object Detection Models (2 models)**

```python
# 8. SSD MobileNet - Fast Detection
self.configs["ssd_mobilenet"] = TensorFlowModelConfig(
    url="https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2",
    model_type="object_detection",
    input_shape=(320, 320, 3),
    capabilities=["object_detection", "real_time"]
)

# 9. Faster R-CNN - High Accuracy
self.configs["faster_rcnn"] = TensorFlowModelConfig(
    url="https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1",
    model_type="object_detection",
    input_shape=(640, 640, 3),
    capabilities=["object_detection", "high_accuracy"]
)
```

**Detection Workflow:**
```python
def detect_objects_tensorflow(self, image_path: str, model_key: str = "ssd_mobilenet"):
    # 1. Load detection model
    detector = hub.load(config.url)
    
    # 2. Preprocess image
    image = self.preprocess_image(image_path, config)
    
    # 3. Run detection
    detections = detector(image)
    
    # 4. Process results
    boxes = detections["detection_boxes"]
    classes = detections["detection_classes"]
    scores = detections["detection_scores"]
    
    return {
        "objects": processed_detections,
        "confidence_scores": scores,
        "bounding_boxes": boxes
    }
```

#### **📊 Specialized Models (2 models)**

```python
# 10. ImageNet MobileNet - Classification
self.configs["imagenet_mobilenet"] = TensorFlowModelConfig(
    url="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4",
    model_type="classification",
    capabilities=["classification", "imagenet"]
)

# 11. BiT ResNet-50 - Transfer Learning
self.configs["bit_resnet50"] = TensorFlowModelConfig(
    url="https://tfhub.dev/google/bit/s-r50x1/1",
    model_type="image_embedding",
    capabilities=["image_embedding", "transfer_learning"]
)
```

### **🔄 Backend Processing Workflow**

```python
# File: enhanced_hybrid_manager.py
class EnhancedHybridModelManager:
    def __init__(self):
        # Initialize TensorFlow Manager
        if TENSORFLOW_MODELS_AVAILABLE:
            self.tensorflow_manager = TensorFlowModelManager()
            
        # Load AI Agents
        if AI_AGENTS_AVAILABLE:
            self.ai_agent_manager = AIAgentManager()
    
    def process_with_tensorflow_backend(self, frame_path: str):
        """Background processing với 11 TensorFlow models"""
        results = {}
        
        # Feature extraction với multiple models
        for model_key in ["mobilenet_v2", "inception_v3", "resnet50"]:
            features = self.tensorflow_manager.extract_image_features(
                frame_path, model_key
            )
            results[model_key] = features
        
        # Object detection
        detections = self.tensorflow_manager.detect_objects(
            frame_path, "ssd_mobilenet"
        )
        results["detections"] = detections
        
        # Text analysis (if metadata available)
        if frame_metadata:
            text_embeddings = self.tensorflow_manager.encode_text(
                frame_metadata, "universal_sentence_encoder"
            )
            results["text_features"] = text_embeddings
        
        return results
```

---

## 🤖 **LAYER 3: AI Agents Flow (2 Premium + 1 Local)**

### **🔑 API Keys Integration Workflow**

#### **1. 🤖 OpenAI GPT-4 Vision**

```python
# File: ai_agent_manager.py
class AIAgentManager:
    def analyze_frame(self, image_path: str, agent_name: str = "gpt4_vision"):
        if config.provider == "openai" and "vision" in config.capabilities:
            # Convert image to base64
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Call GPT-4 Vision API
            response = agent.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert video frame analyzer"
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this video frame in detail"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ]
            )
            
            return response.choices[0].message.content
```

**Khi Nào Được Sử Dụng:**
- **Professional Analysis**: Video analysis chuyên nghiệp
- **Detailed Descriptions**: Mô tả chi tiết frame content
- **Context Understanding**: Hiểu context phức tạp

#### **2. 🧠 Anthropic Claude 3**

```python
def generate_search_query(self, user_query: str, agent_name: str = "claude3"):
    if config.provider == "anthropic":
        response = agent.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[
                {
                    "role": "user",
                    "content": f"Optimize this search query for video content: {user_query}"
                }
            ]
        )
        
        return response.content[0].text
```

**Khi Nào Được Sử Dụng:**
- **Query Optimization**: Tối ưu hóa search queries
- **Reasoning**: Phân tích logic phức tạp
- **Smart Suggestions**: Gợi ý thông minh

#### **3. 🏠 Local BLIP Models (Free)**

```python
def analyze_frame_local(self, image_path: str):
    # Local BLIP model - no API key needed
    if "blip" in config.model.lower():
        processor = BlipProcessor.from_pretrained(config.model)
        model = BlipForConditionalGeneration.from_pretrained(config.model)
        
        # Process image
        image = Image.open(image_path).convert('RGB')
        inputs = processor(image, return_tensors="pt")
        
        # Generate caption
        out = model.generate(**inputs, max_length=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        return {"caption": caption, "confidence": "local_model"}
```

**Khi Nào Được Sử Dụng:**
- **Free Alternative**: Không cần API keys
- **Basic Captioning**: Mô tả cơ bản frames
- **Offline Processing**: Hoạt động offline

---

## 🚀 **Complete Workflow Example**

### **📝 User Search: "tìm người đàn ông"**

```
1. 🌐 LAYER 1: Web Interface
   ├── User types "tìm người đàn ông"
   ├── Chinese CLIP model detects Vietnamese
   ├── Encode query to 512D vector
   └── Search 3,801 frame embeddings
   
2. 🔧 LAYER 2: TensorFlow Backend (Parallel)
   ├── MobileNet V2: Extract lightweight features
   ├── ResNet-50: Deep feature analysis
   ├── USE Multilingual: Multilingual text understanding
   └── SSD MobileNet: Detect objects in frames
   
3. 🤖 LAYER 3: AI Agents (If API keys available)
   ├── Claude 3: "Optimize query for male person detection"
   ├── GPT-4 Vision: Analyze top result frames
   └── Local BLIP: Generate captions for context

4. 📊 Result Fusion
   ├── Combine Layer 1 similarity scores (25.4%)
   ├── Enhance with Layer 2 object detection
   ├── Add Layer 3 intelligent descriptions
   └── Return enriched results to user
```

### **⚡ Performance Characteristics**

| Layer | Processing Time | Memory Usage | GPU Usage | Accuracy |
|-------|----------------|--------------|-----------|----------|
| **Layer 1** | 0.1-0.3s | 2-4 GB | High | Good-Excellent |
| **Layer 2** | 0.5-2.0s | 1-8 GB | Medium | Specialized |
| **Layer 3** | 1-5s | Low | None | Excellent |

### **🎯 Model Selection Strategy**

```python
def select_optimal_models(query_type: str, performance_priority: str):
    if query_type == "vietnamese_text":
        primary = "chinese_clip"              # Layer 1
        secondary = "use_multilingual"        # Layer 2
        enhancement = "claude3"               # Layer 3
        
    elif query_type == "fast_search":
        primary = "clip_vit_base"            # Layer 1
        secondary = "mobilenet_v2"           # Layer 2  
        enhancement = "blip_local"           # Layer 3
        
    elif query_type == "high_accuracy":
        primary = "clip_vit_large"           # Layer 1
        secondary = ["resnet50", "inception_v3"]  # Layer 2
        enhancement = "gpt4_vision"          # Layer 3
        
    return {
        "layer1": primary,
        "layer2": secondary,
        "layer3": enhancement
    }
```

---

## 📋 **Tóm Tắt Flow Hoạt Động**

### **🎯 4 Mô Hình Chính (Layer 1)**
- **Always Active**: Luôn sẵn sàng cho user interaction
- **Real-time**: Xử lý search queries ngay lập tức
- **User Choice**: User chọn model qua dropdown
- **Primary Results**: Cung cấp kết quả chính

### **🔧 11 Mô Hình TensorFlow (Layer 2)**  
- **Background**: Chạy ngầm để enhance results
- **Specialized**: Mỗi model có chuyên môn riêng
- **Parallel Processing**: Xử lý song song nhiều models
- **Feature Enhancement**: Bổ sung thông tin chuyên sâu

### **🤖 AI Agents (Layer 3)**
- **Premium Enhancement**: Nâng cao với API keys
- **Intelligent Analysis**: Phân tích thông minh
- **Context Understanding**: Hiểu context phức tạp
- **Optional**: Hoạt động mà không cần API keys (Local BLIP)

**Kết quả**: Hệ thống 3 lớp hoạt động song song, bổ sung cho nhau tạo ra trải nghiệm search thông minh và chính xác!
