# 🎥 Enhanced Video Search System with TensorFlow Hub

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)

**Advanced AI-powered video search system with intelligent TensorFlow Hub model selection**

[🚀 Quick Start](#-quick-start) • [📖 Features](#-features) • [🤖 TensorFlow Hub Integration](#-tensorflow-hub-integration) • [💻 Usage](#-usage) • [🔧 API](#-api-reference)

</div>

---

## 📋 Overview

Enhanced Video Search System là một hệ thống tìm kiếm video thông minh sử dụng AI, được tích hợp với nhiều mô hình TensorFlow Hub để cung cấp khả năng xử lý đa phương thức (text, video, audio) tiên tiến.

### ✨ Key Features

- 🤖 **Intelligent Model Selection**: Tự động đề xuất mô hình TensorFlow Hub phù hợp
- 🔍 **Multi-modal Search**: Tìm kiếm kết hợp text, hình ảnh và video
- 🌍 **Multilingual Support**: Hỗ trợ tiếng Việt và tiếng Anh
- ⚡ **Smart Overlap Detection**: Phát hiện và giải quyết xung đột giữa các mô hình
- 🎯 **Memory Optimization**: Quản lý bộ nhớ thông minh
- 🌐 **Multiple Interfaces**: Web UI, API REST, và CLI

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd enhanced-video-search

# Run one-click setup
python scripts/setup_complete.py

# Or quick launch
launch.bat  # Windows
# python scripts/setup_complete.py  # All platforms
```

### 2. Launch System

```bash
# Quick launch menu
launch.bat

# Or use detailed startup script
scripts/start_server.bat

# Or manually start specific component:
python src/api/app.py                    # Enhanced API
streamlit run src/ui/enhanced_web_interface.py  # Advanced Web UI
python demos/enhanced_video_demo.py        # Interactive Demo
```

### 3. Access Interfaces

- **🌐 Enhanced Web Interface**: http://localhost:8501
- **📡 API Documentation**: http://localhost:8000/docs
- **🔍 Standard Web UI**: http://localhost:5000

---

## 🤖 TensorFlow Hub Integration

### 📊 Available Models & Capabilities

Hệ thống tích hợp nhiều mô hình TensorFlow Hub mạnh mẽ từ [TensorFlow Hub](https://tfhub.dev/), mỗi mô hình đảm nhận một chức năng cụ thể trong việc xử lý và tìm kiếm video:

#### 🔤 **Text Understanding Models**

##### **Universal Sentence Encoder Multilingual v3**
- **🔗 Link**: https://tfhub.dev/google/universal-sentence-encoder-multilingual/3
- **🎯 Chức năng**: Mã hóa văn bản đa ngôn ngữ thành vectors 512-dimensional
- **💡 Ứng dụng trong project**:
  - Hiểu ngữ nghĩa của query tiếng Việt và tiếng Anh
  - Tạo embeddings cho subtitle và metadata
  - So sánh semantic similarity giữa query và nội dung video
  - Hỗ trợ cross-language search (tìm bằng tiếng Việt, kết quả tiếng Anh)

```python
# Example usage in project:
query = "hướng dẫn nấu ăn"  # Vietnamese query
embedding = use_model([query])
# → Finds cooking tutorials in English videos
```

##### **Universal Sentence Encoder v4**
- **🔗 Link**: https://tfhub.dev/google/universal-sentence-encoder/4
- **🎯 Chức năng**: Phiên bản tối ưu cho tiếng Anh, tốc độ cao
- **💡 Ứng dụng**: Fallback option cho English-only content, faster processing

#### 🎬 **Video Understanding Models**

##### **MoViNet A0 Action Recognition**
- **🔗 Link**: https://tfhub.dev/tensorflow/movinet/a0/base/kinetics-600/classification/3
- **🎯 Chức năng**: Nhận diện 600 loại hành động trong video (Kinetics-600 dataset)
- **💡 Ứng dụng trong project**:
  - Phát hiện activities: "cooking", "programming", "teaching", "exercising"
  - Temporal action localization trong video
  - Content-based video categorization
  - Smart highlights extraction

```python
# Example: Tìm video có hành động "cooking"
query = "cooking tutorial"
# MoViNet detects: stirring, chopping, frying actions
# → Returns timestamped cooking segments
```

##### **MoViNet A2 Action Recognition**
- **🔗 Link**: https://tfhub.dev/tensorflow/movinet/a2/base/kinetics-600/classification/3
- **🎯 Chức năng**: Version có độ chính xác cao hơn, slower processing
- **💡 Ứng dụng**: High-precision action detection cho important content

#### 🖼️ **Visual Feature Models**

##### **EfficientNet V2 B0**
- **🔗 Link**: https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2
- **🎯 Chức năng**: Trích xuất visual features từ frames (1280-dim vectors)
- **💡 Ứng dụng trong project**:
  - Scene understanding: office, kitchen, outdoor, classroom
  - Visual similarity search giữa các frames
  - Thumbnail generation và key frame selection
  - Visual content clustering

```python
# Example: Tìm scenes tương tự
frame_features = efficientnet_model(frame)
# → Groups similar visual scenes together
# → Finds frames with similar composition/lighting
```

##### **EfficientNet V2 B3**
- **🔗 Link**: https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b3/feature_vector/2
- **🎯 Chức năng**: Higher quality visual features (1536-dim), better accuracy
- **💡 Ứng dụng**: Fine-grained visual analysis cho detailed searches

#### 🔍 **Object Detection Models**

##### **SSD MobileNet v2**
- **🔗 Link**: https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2
- **🎯 Chức năng**: Real-time object detection, 80+ object classes
- **💡 Ứng dụng trong project**:
  - Detect objects: person, laptop, book, food, car
  - Content-aware search: "videos with computers", "tutorials with books"
  - Automatic tagging và metadata enrichment
  - Smart content filtering

```python
# Example: Tìm video có laptop
detections = ssd_model(frame)
# → Finds frames containing laptops
# → Tags videos as "programming" or "tutorial"
```

##### **Faster R-CNN**
- **🔗 Link**: https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1
- **🎯 Chức năng**: High-precision object detection với bounding boxes
- **💡 Ứng dụng**: Detailed object localization cho professional analysis

### 🧠 **Intelligent Model Selection System**

#### **Smart Overlap Detection**

Hệ thống tự động phát hiện khi các mô hình có chức năng trùng lặp:

```
⚠️  OVERLAP DETECTED:
📝 Text Encoding: USE Multilingual ↔ USE v4
🎬 Action Recognition: MoViNet A0 ↔ MoViNet A2  
🖼️  Visual Features: EfficientNet B0 ↔ EfficientNet B3
🔍 Object Detection: SSD MobileNet ↔ Faster R-CNN

💡 RECOMMENDATION: Choose based on priority
   - Speed: USE v4, MoViNet A0, EfficientNet B0, SSD MobileNet
   - Accuracy: USE Multilingual, MoViNet A2, EfficientNet B3, Faster R-CNN
   - Balance: Mix of both depending on use case
```

#### **Complementary Model Combinations**

Hệ thống đề xuất kết hợp mô hình bổ trợ:

```
✅ COMPLEMENTARY PAIRS:
🔤 + 🎬: Text Understanding + Action Recognition
   → "Find cooking tutorials" = Text query + Cooking actions

🖼️ + 🔍: Visual Features + Object Detection  
   → Scene understanding + Specific object finding

🌍 + 🎯: Multilingual + Specialized models
   → Vietnamese query + English content analysis
```

### 💾 **Memory Usage & Performance**

| Model | Memory | Speed | Use Case |
|-------|---------|--------|-----------|
| USE Multilingual | ~500MB | Fast | Multilingual text search |
| USE v4 | ~300MB | Very Fast | English-only content |
| MoViNet A0 | ~100MB | Fast | Real-time action detection |
| MoViNet A2 | ~300MB | Medium | High-precision actions |
| EfficientNet B0 | ~50MB | Very Fast | Quick visual features |
| EfficientNet B3 | ~200MB | Medium | Detailed visual analysis |
| SSD MobileNet | ~100MB | Fast | Real-time object detection |
| Faster R-CNN | ~500MB | Slow | Precise object localization |

### 🎯 **Practical Examples**

#### **Example 1: Cooking Tutorial Search**
```
User Intent: "Tìm video hướng dẫn nấu ăn"

Selected Models:
✅ USE Multilingual (Vietnamese query understanding)
✅ MoViNet A0 (Detect cooking actions: stirring, chopping)
✅ EfficientNet B0 (Kitchen scene detection)
✅ SSD MobileNet (Detect kitchen objects: knife, pan, food)

Result: Comprehensive cooking video search với:
- Semantic understanding của "hướng dẫn nấu ăn"
- Action recognition cho cooking activities
- Visual scene matching cho kitchen environments
- Object detection cho cooking utensils
```

#### **Example 2: Programming Tutorial Analysis**
```
User Intent: "Find coding tutorials with screen recordings"

Selected Models:
✅ USE v4 (English programming terms)
✅ EfficientNet B3 (Screen/UI pattern recognition)  
✅ SSD MobileNet (Detect: laptop, monitor, keyboard)

Result: Precise programming content với:
- Technical term understanding
- Code editor visual patterns
- Development environment detection
```

---

## 💻 Usage

### 🌐 Web Interface Usage

1. **Launch Enhanced Interface**:
   ```bash
   streamlit run enhanced_web_interface.py
   ```

2. **Model Selection Process**:
   - Describe your intent: "Find action sequences in cooking videos"
   - System analyzes và recommends optimal models
   - Review overlaps và select preferred configuration
   - Load models (may take 5-10 minutes first time)

3. **Video Processing**:
   - Upload or specify video path
   - Enter search query (optional)
   - Process với selected models
   - View detailed results và features

### 📡 API Usage

#### **Intelligent Model Analysis**
```python
import requests

# Analyze requirements
response = requests.post('http://localhost:8000/analyze_models', json={
    "user_intent": "Find action sequences in cooking videos",
    "max_memory_mb": 2000,
    "processing_priority": "balanced"
})

recommendations = response.json()
print(recommendations['suggested_models'])
# → ['use_multilingual', 'movinet_a0', 'efficientnet_v2_b0', 'ssd_mobilenet']
```

#### **Load Selected Models**
```python
# Load models
response = requests.post('http://localhost:8000/load_selected_models', 
    json=['use_multilingual', 'movinet_a0', 'efficientnet_v2_b0'])

print(response.json()['message'])
# → "Loaded 3/3 models successfully"
```

#### **Process Video**
```python
# Process video với loaded models
response = requests.post('http://localhost:8000/process_video', json={
    "video_path": "videos/cooking_tutorial.mp4",
    "query": "chopping vegetables",
    "selected_models": ["use_multilingual", "movinet_a0"]
})

results = response.json()
print(f"Processing time: {results['processing_time']}")
print(f"Results: {results['processing_results']}")
```

### 🖥️ CLI Demo Usage

```bash
python enhanced_video_demo.py

# Interactive options:
# 1. Model Analysis Demo (Quick preview)
# 2. Interactive Model Selection (Full workflow)  
# 3. Model Status Overview
```

---

## 🔧 API Reference

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze_models` | POST | Analyze user intent & suggest models |
| `/load_selected_models` | POST | Load specified TensorFlow Hub models |
| `/process_video` | POST | Process video với selected models |
| `/enhanced_search` | GET | Advanced search với TF Hub integration |
| `/status` | GET | System status & loaded models |

### Enhanced Endpoints

```python
# Model analysis request
{
    "user_intent": "string",
    "max_memory_mb": 2000,
    "processing_priority": "lightweight|balanced|high_accuracy"
}

# Model analysis response  
{
    "recommendations": {
        "lightweight": ["model1", "model2"],
        "balanced": ["model1", "model3"],
        "high_accuracy": ["model2", "model4"]
    },
    "overlaps_detected": {
        "balanced": {"model1": ["model2"]}
    },
    "suggested_models": ["model1", "model3"],
    "estimated_memory_usage": "~650MB"
}
```

---

## 📁 Project Structure

```
enhanced-video-search/
├── � launch.bat                  # Quick launch menu
├── 
├── 📖 docs/
│   ├── README.md                  # Comprehensive documentation
│   ├── QUICK_START.md            # 5-minute setup guide
│   └── CHANGELOG.md              # Version history
├── 
├── 🧠 src/
│   ├── 🤖 core/
│   │   └── enhanced_video_processor.py  # Smart model manager
│   ├── 📡 api/
│   │   ├── app.py                # Enhanced API với TF Hub
│   │   ├── simple_enhanced_api.py # Lightweight fallback
│   │   └── vietnamese_translator.py # Vietnamese support
│   └── 🌐 ui/
│       ├── enhanced_web_interface.py # Advanced Streamlit UI
│       └── web_search_app.py     # Standard web UI
├── 
├── 🎬 demos/
│   ├── enhanced_video_demo.py    # Interactive TF Hub demo
│   └── interactive_search_demo.py # CLI demo
├── 
├── ⚙️ scripts/
│   ├── setup_complete.py         # One-click setup script
│   ├── start_server.bat         # Detailed startup menu
│   ├── start_server_advanced.bat # Advanced startup
│   └── start_server_simple.bat  # Simple startup
├── 
├── 📦 configs/
│   ├── requirements.txt         # Core dependencies
│   └── requirements_enhanced.txt # TensorFlow Hub dependencies
├── 
└── � index/, 🎥 videos/, 🖼️ frames/ # Data directories
```

---

## 🛠️ Configuration Options

### Memory Configurations

| Configuration | Memory Usage | Models Included | Best For |
|---------------|-------------|-----------------|----------|
| **Lightweight** | ~650MB | USE v4, MoViNet A0, EfficientNet B0, SSD MobileNet | Fast processing, limited resources |
| **Balanced** | ~1200MB | USE Multilingual, MoViNet A0, EfficientNet B0, SSD MobileNet | Good performance/memory ratio |
| **High Accuracy** | ~2000MB+ | USE Multilingual, MoViNet A2, EfficientNet B3, Faster R-CNN | Best quality, high-end systems |

### Processing Priorities

- **Speed**: Optimize for fast results
- **Accuracy**: Optimize for best quality  
- **Balanced**: Balance speed và accuracy
- **Custom**: Manual model selection

---

## 🚨 Troubleshooting

### Common Issues

1. **TensorFlow Hub models failing to load**:
   ```bash
   pip install --upgrade tensorflow tensorflow-hub tensorflow-text
   ```

2. **Memory errors**:
   - Use Lightweight configuration
   - Reduce max_memory_mb parameter
   - Close other applications

3. **Slow processing**:
   - Check GPU availability: `nvidia-smi`
   - Use GPU-optimized TensorFlow: `pip install tensorflow-gpu`

4. **Model compatibility issues**:
   - Update to latest model versions
   - Check TensorFlow version compatibility

### Performance Tips

- **First run**: Models download automatically (may take 10+ minutes)
- **GPU acceleration**: Install CUDA for faster processing
- **Memory optimization**: Use model caching và lazy loading
- **Batch processing**: Process multiple videos together

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **TensorFlow Hub**: For providing powerful pre-trained models
- **Google Research**: Universal Sentence Encoder và EfficientNet models
- **DeepMind**: MoViNet action recognition models
- **OpenAI**: For AI development inspiration
- **Streamlit**: For beautiful web interface framework

---

<div align="center">

**🎥 Enhanced Video Search System - Powered by TensorFlow Hub & AI**

Made with ❤️ for the AI community

</div>
