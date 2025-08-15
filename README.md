# 🎥 Enhanced Video Search System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)
![Mode](https://img.shields.io/badge/Mode-Simple%20%26%20Fast-brightgreen.svg)

**🧠 Modern video search system - Simple & Reliable**

[🚀 Quick Start](#-quick-start) • [📖 Features](#-features) • [🎯 Usage](#-usage) • [🔧 API](#-api-reference)

</div>

---

## 📋 Overview

Enhanced Video Search System là hệ thống tìm kiếm video thông minh với khả năng **auto-detection** và **intelligent fallback**. Hệ thống tự động phát hiện TensorFlow availability và chọn mode phù hợp.

### ✨ Key Features

- � **Auto-Detection**: Tự động phát hiện TensorFlow Hub và chọn mode tối ưu
- � **Intelligent Fallback**: Full mode → Simple mode khi TensorFlow không available  
- 🌍 **Cross-Platform**: Windows, Linux, macOS với 1 launcher thống nhất
- ⚡ **Smart Search**: Vector search với TensorFlow Hub hoặc keyword search
- 🎯 **Unified Experience**: 1 cách duy nhất để khởi động và sử dụng
- 🌐 **Multiple Interfaces**: API, Web UI đều tích hợp auto-detection

---

## 🚀 Quick Start

### 1. One-Click Launch (Cross-Platform)

**🪟 Windows - Multiple Options:**
```cmd
# Option 1: Simple batch (recommended for beginners)
start.bat

# Option 2: Modern PowerShell (advanced features)
powershell -ExecutionPolicy Bypass -File start.ps1

# Option 3: Direct Python
python start.py
```

**🐧 Linux:**
```bash
# Make executable (first time only)
chmod +x start.sh

# Launch
./start.sh

# Or direct Python
python3 start.py
```

**🍎 macOS:**
```bash
# Make executable (first time only)
chmod +x start.sh

# Launch
./start.sh

# Or direct Python
python3 start.py
```

**🌍 Universal (Any OS):**
```bash
python start.py
```

```

### 2. Smart Auto-Install Features

Launcher mới có khả năng **Smart Auto-Install**:

```
🚀==========================================================🚀
    Enhanced Video Search - Smart Auto-Install
🚀==========================================================🚀

🎯 CURRENT MODE: ⚡ SIMPLE MODE / 🔥 FULL MODE

CHỌN CHỨC NĂNG:
1. 🧠 Tìm kiếm video (AI-Powered) / 🔍 Simple Mode
2. � Cài đặt TensorFlow (Upgrade to Full Mode)
3. �📊 Xem thông tin hệ thống
4. 🌐 Khởi chạy API Server
5. 🛠️  Cài đặt dependencies
6. � Test lại TensorFlow
7. �🚪 Thoát
```

**🔥 Key Features:**
- ✅ **Auto-Detection**: Tự động phát hiện TensorFlow có sẵn
- ✅ **Smart Install**: Cài đặt TensorFlow với phiên bản tương thích
- ✅ **Mode Switching**: Chuyển đổi linh hoạt Simple ↔ Full Mode
- ✅ **Cross-Platform**: Chạy mượt trên Windows/Linux/macOS
- ✅ **Virtual Env**: Tự động kích hoạt virtual environment
- ✅ **Error Handling**: Xử lý lỗi thông minh và gợi ý khắc phục

### 3. Access Points

- **API Server**: http://localhost:8000 (chọn option 4)
- **API Docs**: http://localhost:8000/docs
- **Search Function**: Tích hợp trong launcher (option 1)

---

## 🎯 Modern Simple Features

### 🚀 Simple & Fast System

Hệ thống hiện tại tập trung vào:
- ✅ **Tốc độ**: Khởi động nhanh, không dependency phức tạp
- ✅ **Ổn định**: Không lỗi TensorFlow compatibility 
- ✅ **Dễ sử dụng**: Menu interactive thân thiện
- ✅ **Cross-platform**: Chạy được trên Windows/Linux/macOS

### 🌐 Unified API Server

```python
# API endpoints hiện tại
GET /api/health      # Kiểm tra trạng thái server
GET /api/search      # Tìm kiếm với keyword matching
GET /system/info     # Thông tin chi tiết hệ thống
```

**Simple Mode** (Current implementation):
- Keyword-based search hiệu quả
- Metadata search trong parquet files
- Fast response time
- No complex dependencies
- Fallback functionality

---

## 📖 System Architecture

1. **Quick Setup & Status Check** - Check system status and dependencies
2. **Install Dependencies** - Install required Python packages
3. **Start Enhanced API Server** - Launch FastAPI server (http://localhost:8000)
4. **Start Web Interface** - Launch Streamlit interface (http://localhost:8501)
5. **Check System Status** - Detailed system and dependency status
6. **Run Complete Setup** - Full system setup and testing

## Requirements

- Python 3.8 or higher
- Internet connection (for downloading dependencies)
- At least 2GB RAM (for TensorFlow operations)
- 1GB free disk space

## First Time Setup

1. Clone/download the project
2. Navigate to the project directory
3. Run the appropriate launcher for your OS:
   - Windows: `run.bat`
   - Linux/macOS: `./run.sh`
4. Choose option [6] for complete setup
5. Add video files to the `videos/` directory
6. Start using the system!

## Troubleshooting

### Windows
- If you get encoding errors, the launcher automatically sets UTF-8 encoding
- Run as Administrator if you have permission issues

### Linux/macOS
- Make sure the script is executable: `chmod +x run.sh`
- Use `python3` if `python` points to Python 2

### All Platforms
- Make sure you're in the project root directory
- Check that Python 3.8+ is installed
- Ensure you have internet connection for package downloads

## Manual Commands

If the launchers don't work, you can run commands manually:

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/macOS)
source .venv/bin/activate

# Install dependencies
python scripts/install_dependencies.py

# Check status
python scripts/check_status.py

# Start API server
python src/api/app.py

# Start web interface
python -m streamlit run src/ui/enhanced_web_interface.py
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

## 🧮 Mathematical Foundations & Academic Background

### 📚 **Theoretical Foundations Behind TensorFlow Hub Models**

Mỗi mô hình trong hệ thống được xây dựng trên các nền tảng toán học và nghiên cứu học thuật sâu sắc. Phần này giải thích các công thức, thuật toán và lý thuyết đằng sau mỗi mô hình.

#### 🔤 **Universal Sentence Encoder (USE) - Mathematical Foundation**

##### **Core Architecture: Transformer + Deep Averaging Network (DAN)**

**📈 Mathematical Formula:**
```
Text Embedding = f(x) = Transformer(x) ⊕ DAN(x)

Where:
- x = input text sequence [w₁, w₂, ..., wₙ]  
- Transformer(x) = MultiHead(Q, K, V) + PositionalEncoding
- DAN(x) = ReLU(W₂ · ReLU(W₁ · average(embed(x))))
- ⊕ = element-wise combination operator
```

**🧠 Academic Background:**
- **Paper**: "Universal Sentence Encoder" (Cer et al., 2018)
- **Key Innovation**: Dual-encoder architecture combining:
  - **Transformer path**: Captures complex syntactic relationships
  - **DAN path**: Efficient semantic averaging for speed
- **Training Objective**: 
  ```
  L = -log σ(cos(u, v⁺)) - Σᵢ log σ(-cos(u, vᵢ⁻))
  
  Where:
  - u, v⁺ = positive sentence pair embeddings
  - vᵢ⁻ = negative sentence embeddings  
  - σ = sigmoid function
  - cos = cosine similarity
  ```

**🌍 Multilingual Extension (USE-M):**
- **Cross-lingual Training**: Trained on parallel corpora (16 languages)
- **Shared Embedding Space**: 
  ```
  d(embed_en(s), embed_vi(t)) ≈ semantic_similarity(s, t)
  
  Where s = English sentence, t = Vietnamese sentence
  ```
- **Application trong project**: Enables Vietnamese ↔ English cross-lingual search

##### **Embedding Space Properties:**
```
Embedding Dimension: ℝ⁵¹²
Distance Metric: Cosine Similarity ∈ [-1, 1]
Semantic Clustering: ||embed(s₁) - embed(s₂)||₂ ∝ semantic_distance(s₁, s₂)
```

#### 🎬 **MoViNet (Mobile Video Networks) - Mathematical Foundation**

##### **Core Innovation: Causal 3D Convolutions with Stream Buffers**

**📈 Mathematical Formula:**
```
Video Feature = MoViNet(X) = Ψ(Φ₁(X), Φ₂(X), ..., Φₜ(X))

Where:
- X ∈ ℝᵀˣᴴˣᵂˣᶜ (T=time, H=height, W=width, C=channels)
- Φₜ = Causal3DConv + StreamBuffer + TemporalPooling
- Ψ = Classification head with 600 Kinetics action classes
```

**🧠 Academic Background:**
- **Paper**: "MoViNets: Mobile Video Networks for Efficient Video Recognition" (Kondratyuk et al., 2021)
- **Key Innovation**: 
  - **Causal Convolutions**: Only use past frames → Real-time processing
  - **Stream Buffers**: Maintain temporal state across video chunks
  - **Neural Architecture Search (NAS)**: Automated architecture optimization

**⚡ Efficiency Architecture:**
```
MoViNet Architecture:
Input → [Causal3D Blocks] → [Stream Buffers] → [Classification Head]

Stream Buffer Update:
S_t = α·S_{t-1} + β·Features_t
Where α, β are learned parameters for temporal memory
```

**🎯 Action Recognition Math:**
- **Temporal Receptive Field**: Exponentially growing with depth
- **Classification**: Softmax over 600 Kinetics classes
  ```
  P(action = k | video) = exp(w_k · f_video) / Σⱼ exp(w_j · f_video)
  ```

#### 🖼️ **EfficientNet V2 - Mathematical Foundation**

##### **Neural Architecture Search + Progressive Learning**

**📈 Mathematical Formula:**
```
EfficientNet Scaling:
depth = α^φ
width = β^φ  
resolution = γ^φ

Subject to: α·β²·γ² ≈ 2^φ
Where φ = compound scaling coefficient
```

**🧠 Academic Background:**
- **Papers**: 
  - "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" (Tan & Le, 2019)
  - "EfficientNetV2: Smaller Models and Faster Training" (Tan & Le, 2021)
- **Key Innovation**: 
  - **Compound Scaling**: Uniformly scale depth, width, resolution
  - **Progressive Learning**: Gradually increase image size during training
  - **Fused-MBConv blocks**: Improved speed-accuracy tradeoff

**⚙️ MBConv Block Mathematics:**
```
MBConv(x) = x + DropPath(SE(DWConv(Expand(x))))

Where:
- Expand: 1×1 conv, expansion ratio = 6
- DWConv: Depthwise 3×3 or 5×5 convolution  
- SE: Squeeze-and-Excitation attention
- DropPath: Stochastic depth regularization
```

**🎯 Feature Extraction trong Project:**
- **Output**: Dense feature vectors ∈ ℝ¹²⁸⁰ (B0) or ℝ¹⁵³⁶ (B3)
- **Semantic Similarity**: Cosine distance between feature vectors
- **Application**: Visual scene matching, thumbnail generation

#### 🔍 **Object Detection Models - Mathematical Foundation**

##### **SSD (Single Shot MultiBox Detector)**

**📈 Mathematical Formula:**
```
SSD Loss = L_conf + α·L_loc

Confidence Loss:
L_conf = -Σᵢ∈Pos x_{ij}^p log(ĉᵢ^p) - Σᵢ∈Neg log(ĉᵢ⁰)

Localization Loss:  
L_loc = Σᵢ∈Pos Σₘ∈{cx,cy,w,h} x_{ij}^m SmoothL1(lᵢᵐ - ĝⱼᵐ)

Where:
- x_{ij}^p = {1 if box i matches object j of class p, 0 otherwise}
- ĉᵢ^p = predicted confidence for class p in box i
- α = weight balancing term (typically α = 1)
```

**🧠 Academic Background:**
- **Paper**: "SSD: Single Shot MultiBox Detector" (Liu et al., 2016)
- **Key Innovation**: Multi-scale feature maps for different object sizes
- **Architecture**: 
  ```
  Input Image → VGG16 Base → Feature Pyramid → [Classification + Regression] Heads
  ```

##### **Faster R-CNN - Two-Stage Detection**

**📈 Mathematical Formula:**
```
Two-Stage Process:
Stage 1: RPN(x) → {proposals, objectness_scores}
Stage 2: R-CNN(proposals) → {classes, refined_boxes}

RPN Loss:
L_RPN = L_cls + λ·L_reg
Where L_cls = binary classification (object/background)
      L_reg = bounding box regression loss
```

**🧠 Academic Background:**
- **Paper**: "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks" (Ren et al., 2017)
- **Innovation**: End-to-end trainable region proposal network
- **Advantage**: Higher precision for complex scenes

### 🔬 **Information Theory & Embedding Space Analysis**

#### **Semantic Embedding Metrics**

**📊 Cosine Similarity Distribution:**
```
sim(u, v) = (u · v) / (||u|| · ||v||)

Semantic Properties:
- Synonyms: sim ∈ [0.7, 1.0]
- Related concepts: sim ∈ [0.4, 0.7]  
- Unrelated: sim ∈ [-0.2, 0.4]
- Antonyms: sim ∈ [-1.0, -0.2]
```

**📈 Entropy Analysis trong Video Search:**
```
Information Gain = H(Results) - H(Results|Query)

Where:
H(Results) = -Σᵢ P(video_i) log P(video_i)
H(Results|Query) = -Σᵢ P(video_i|query) log P(video_i|query)
```

#### **Multi-modal Fusion Mathematics**

**🔗 Feature Fusion Strategy:**
```
Combined_Score = w₁·Text_Similarity + w₂·Visual_Similarity + w₃·Action_Similarity

Subject to: w₁ + w₂ + w₃ = 1, wᵢ ≥ 0

Optimization:
w* = argmax Σₖ log P(relevant_k | combined_score_k)
```

**🎯 Attention Mechanism trong Multi-modal Search:**
```
Attention Weights:
αᵢ = exp(score_i) / Σⱼ exp(score_j)

Final Representation:
r = Σᵢ αᵢ · feature_i

Where score_i = query_embedding · feature_i
```

### 📊 **Computational Complexity Analysis**

| Model | Time Complexity | Space Complexity | FLOPs |
|-------|----------------|------------------|-------|
| **USE Multilingual** | O(n log n) | O(d·n) | ~1.2B per sentence |
| **MoViNet A0** | O(T·H·W·C) | O(buffer_size) | ~2.7B per video chunk |
| **EfficientNet B0** | O(H·W·C·D) | O(D) | ~0.39B per image |
| **SSD MobileNet** | O(N·K·A) | O(N·A) | ~1.2B per frame |

**Legend:**
- n = sequence length, d = embedding dimension
- T = temporal frames, H×W = spatial resolution, C = channels
- N = number of boxes, K = number of classes, A = number of anchors
- D = network depth

### 🧪 **Experimental Validation & Benchmarks**

#### **Performance Metrics trong Academic Literature:**

**Text Understanding (USE):**
- **STS Benchmark**: Pearson correlation = 0.78-0.82
- **Cross-lingual Transfer**: BLEU score improvement = +15-20%

**Action Recognition (MoViNet):**
- **Kinetics-600**: Top-1 Accuracy = 72.8% (A0), 76.5% (A2)
- **Real-time Performance**: 54 FPS (A0), 28 FPS (A2) on mobile GPU

**Visual Features (EfficientNet):**
- **ImageNet Top-1**: 77.1% (B0), 82.0% (B3)
- **Transfer Learning**: +5-10% accuracy on downstream tasks

**Object Detection:**
- **COCO mAP**: 22.2 (SSD MobileNet), 37.4 (Faster R-CNN)
- **Inference Speed**: 22ms vs 89ms per frame

### 🔄 **Mathematical Optimization trong Project**

#### **Memory-Accuracy Tradeoffs:**
```
Optimization Problem:
maximize: Σᵢ wᵢ · accuracy_i
subject to: Σᵢ memory_i ≤ M_max
           Σᵢ latency_i ≤ L_max
           wᵢ ∈ {0, 1} (binary selection)

Solution: Dynamic programming with Lagrange multipliers
```

#### **Multi-objective Model Selection:**
```
Pareto Optimal Solutions:
f₁(x) = accuracy(x)  (maximize)
f₂(x) = memory(x)    (minimize)  
f₃(x) = latency(x)   (minimize)

Non-dominated solutions form the efficient frontier
```

### 🎯 **Practical Mathematical Applications trong Video Search**

#### **Semantic Search Score Calculation:**
```python
def compute_semantic_score(query_embedding, video_features):
    """
    Compute weighted semantic similarity score
    
    Mathematical foundation:
    score = Σᵢ wᵢ · cos_sim(query, feature_i) · confidence_i
    """
    text_sim = cosine_similarity(query_embedding, video_features['text'])
    visual_sim = cosine_similarity(query_embedding, video_features['visual'])
    action_sim = cosine_similarity(query_embedding, video_features['action'])
    
    # Learned weights from validation data
    w_text, w_visual, w_action = 0.5, 0.3, 0.2
    
    combined_score = (w_text * text_sim + 
                     w_visual * visual_sim + 
                     w_action * action_sim)
    
    return combined_score
```

#### **Temporal Action Localization Algorithm:**
```python
def temporal_action_localization(video_features, action_query, threshold=0.7):
    """
    Mathematical approach: Sliding window with exponential smoothing
    
    Smoothing formula:
    S_t = α·S_{t-1} + (1-α)·raw_score_t
    
    Peak detection:
    peaks = {t : S_t > threshold AND S_t > S_{t-1} AND S_t > S_{t+1}}
    """
    smoothed_scores = []
    alpha = 0.3  # smoothing parameter
    
    for t, features in enumerate(video_features):
        raw_score = cosine_similarity(action_query, features)
        if t == 0:
            smoothed_score = raw_score
        else:
            smoothed_score = alpha * smoothed_scores[-1] + (1-alpha) * raw_score
        smoothed_scores.append(smoothed_score)
    
    # Find peaks above threshold
    action_segments = find_peaks(smoothed_scores, threshold)
    return action_segments
```

#### **Cross-lingual Embedding Alignment:**
```python
def cross_lingual_search(vietnamese_query, english_content_embeddings):
    """
    Mathematical foundation: Shared multilingual embedding space
    
    Property: d(embed_vi(q), embed_en(c)) ≈ semantic_distance(q, c)
    Where d = cosine distance, q = Vietnamese query, c = English content
    """
    # USE Multilingual maps both languages to same space
    vi_embedding = use_multilingual.encode([vietnamese_query])
    
    # Direct comparison in shared space
    similarities = cosine_similarity(vi_embedding, english_content_embeddings)
    
    # Ranking with confidence adjustment
    ranked_results = sorted(enumerate(similarities), 
                           key=lambda x: x[1], reverse=True)
    
    return ranked_results
```

### 📈 **Advanced Mathematical Techniques**

#### **Graph-based Video Similarity:**
```
Video Similarity Graph Construction:
G = (V, E) where V = {videos}, E = {(v₁, v₂) : sim(v₁, v₂) > τ}

PageRank-style Ranking:
R(v) = (1-d)/N + d · Σᵤ∈neighbors(v) R(u)/|out_links(u)|

Where:
- d = damping factor (0.85)
- N = total number of videos
- R(v) = importance score of video v
```

#### **Attention-based Feature Fusion:**
```
Multi-head Attention for Feature Fusion:

Attention(Q,K,V) = softmax(QK^T/√d_k)V

Where:
- Q = query features (user intent)
- K = key features (video content)  
- V = value features (video representations)
- d_k = key dimension for scaling

Multi-modal Application:
Q_text = W_q · text_embedding
K_video = [W_k1·visual_features, W_k2·audio_features, W_k3·action_features]
V_video = [W_v1·visual_features, W_v2·audio_features, W_v3·action_features]
```

#### **Bayesian Model Selection:**
```
Bayesian Model Evidence:
P(M_i | Data) = P(Data | M_i) · P(M_i) / P(Data)

For model selection:
log P(M_i | Data) = log P(Data | M_i) + log P(M_i) - log P(Data)

Practical implementation:
- P(M_i) = prior belief about model i
- P(Data | M_i) = likelihood of data given model i
- Choose model with highest posterior probability
```

### 🔍 **Information Retrieval Theory Applications**

#### **TF-IDF for Video Content:**
```
Extended TF-IDF for Multi-modal Content:

TF-IDF_multimodal = w₁·TF-IDF_text + w₂·TF-IDF_visual + w₃·TF-IDF_action

Where:
TF(t,d) = count(t,d) / total_terms(d)
IDF(t) = log(N / |{d : t ∈ d}|)

Visual Terms: Quantized visual features
Action Terms: Detected action categories
Text Terms: Subtitle and metadata words
```

#### **Learning to Rank (LTR) for Video Search:**
```
RankNet Loss Function:
L = Σᵢⱼ C_ij · log(1 + exp(-(s_i - s_j)))

Where:
- C_ij = 1 if document i should rank higher than j, 0 otherwise
- s_i, s_j = model scores for documents i, j
- Optimization: Gradient descent on neural ranking model
```

### 🧮 **Statistical Learning Theory**

#### **Generalization Bounds:**
```
PAC-Bayes Bound for Model Selection:
With probability ≥ 1-δ:

R(h) ≤ R̂(h) + √[(KL(P||P₀) + log(2m/δ)) / (2m-1)]

Where:
- R(h) = true risk
- R̂(h) = empirical risk  
- KL(P||P₀) = KL divergence between posterior and prior
- m = number of training samples
- δ = confidence parameter
```

#### **Active Learning for Video Annotation:**
```
Uncertainty Sampling:
x* = argmax H(p(y|x, θ))

Where:
H(p) = -Σᵢ p(yᵢ) log p(yᵢ)  (entropy)

Query by Committee:
x* = argmax Variance_θ [p(y|x, θ)]

Application: Select most informative videos for manual annotation
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

## 📚 Academic References & Theoretical Sources

### 📖 **Core Papers & Research**

#### **Fundamental Text Representation:**
1. **Cer, D., et al.** (2018). "Universal Sentence Encoder." *arXiv:1803.11175*
   - 🔗 https://arxiv.org/abs/1803.11175
   - **Contribution**: Dual-encoder architecture, multilingual embeddings
   - **Mathematical Innovation**: DAN + Transformer fusion

2. **Yang, Y., et al.** (2019). "Multilingual Universal Sentence Encoder for Semantic Retrieval." *ACL 2019*
   - 🔗 https://arxiv.org/abs/1907.04307  
   - **Contribution**: Cross-lingual transfer learning
   - **Mathematical Foundation**: Shared embedding space across languages

#### **Video Understanding & Action Recognition:**
3. **Kondratyuk, D., et al.** (2021). "MoViNets: Mobile Video Networks for Efficient Video Recognition." *CVPR 2021*
   - 🔗 https://arxiv.org/abs/2103.11511
   - **Innovation**: Causal 3D convolutions, stream buffers
   - **Mathematical Core**: Temporal modeling with mobile efficiency

4. **Carreira, J., Zisserman, A.** (2017). "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset." *CVPR 2017*
   - 🔗 https://arxiv.org/abs/1705.07750
   - **Foundation**: 3D CNNs for video understanding
   - **Dataset**: Kinetics-600 action categories

#### **Efficient Neural Architecture:**
5. **Tan, M., Le, Q.** (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." *ICML 2019*
   - 🔗 https://arxiv.org/abs/1905.11946
   - **Key Insight**: Compound scaling (depth × width × resolution)
   - **Mathematical Formula**: α·β²·γ² ≈ 2^φ

6. **Tan, M., Le, Q.** (2021). "EfficientNetV2: Smaller Models and Faster Training." *ICML 2021*
   - 🔗 https://arxiv.org/abs/2104.00298
   - **Improvements**: Progressive learning, Fused-MBConv blocks

#### **Object Detection Foundations:**
7. **Liu, W., et al.** (2016). "SSD: Single Shot MultiBox Detector." *ECCV 2016*
   - 🔗 https://arxiv.org/abs/1512.02325
   - **Innovation**: Multi-scale feature maps, single-shot detection
   - **Mathematical Core**: Combined classification + localization loss

8. **Ren, S., et al.** (2017). "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks." *NIPS 2015*
   - 🔗 https://arxiv.org/abs/1506.01497
   - **Breakthrough**: End-to-end trainable region proposals
   - **Two-stage Architecture**: RPN + Fast R-CNN

### 🧠 **Theoretical Foundations**

#### **Information Theory & Embedding Spaces:**
9. **Mikolov, T., et al.** (2013). "Distributed Representations of Words and Phrases and their Compositionality." *NIPS 2013*
   - **Foundation**: Word embedding principles
   - **Mathematical**: Skip-gram with negative sampling

10. **Pennington, J., et al.** (2014). "GloVe: Global Vectors for Word Representation." *EMNLP 2014*
    - **Innovation**: Global matrix factorization for embeddings
    - **Mathematical**: Co-occurrence matrix optimization

#### **Attention & Transformer Mechanisms:**
11. **Vaswani, A., et al.** (2017). "Attention is All You Need." *NIPS 2017*
    - 🔗 https://arxiv.org/abs/1706.03762
    - **Revolutionary**: Self-attention mechanism
    - **Mathematical**: Scaled dot-product attention

12. **Devlin, J., et al.** (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL 2019*
    - 🔗 https://arxiv.org/abs/1810.04805
    - **Impact**: Bidirectional context understanding
    - **Foundation**: Masked language modeling

#### **Multi-modal Learning:**
13. **Radford, A., et al.** (2021). "Learning Transferable Visual Representations from Natural Language Supervision." *ICML 2021*
    - 🔗 https://arxiv.org/abs/2103.00020
    - **CLIP Innovation**: Vision-language contrastive learning
    - **Mathematical**: Contrastive loss in joint embedding space

14. **Jia, C., et al.** (2021). "Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision." *ICML 2021*
    - 🔗 https://arxiv.org/abs/2102.05918
    - **ALIGN Contribution**: Large-scale noisy supervision

### 📊 **Optimization & Learning Theory**

#### **Neural Architecture Search:**
15. **Zoph, B., Le, Q.V.** (2017). "Neural Architecture Search with Reinforcement Learning." *ICLR 2017*
    - **Foundation**: Automated architecture design
    - **Mathematical**: RL-based search in architecture space

16. **Tan, M., et al.** (2019). "MnasNet: Platform-Aware Neural Architecture Search for Mobile." *CVPR 2019*
    - **Mobile Focus**: Efficiency-accuracy tradeoffs
    - **Multi-objective**: Latency-aware optimization

#### **Information Retrieval & Ranking:**
17. **Burges, C., et al.** (2005). "Learning to Rank using Gradient Descent." *ICML 2005*
    - **RankNet**: Pairwise ranking with neural networks
    - **Mathematical**: Probability-based ranking loss

18. **Liu, T.Y.** (2009). "Learning to Rank for Information Retrieval." *Foundations and Trends in Information Retrieval*
    - **Comprehensive**: Survey of ranking algorithms
    - **Theory**: Statistical learning for ranking

### 🔍 **Mathematical Background References**

#### **Linear Algebra & Optimization:**
19. **Boyd, S., Vandenberghe, L.** (2004). "Convex Optimization." *Cambridge University Press*
    - **Foundation**: Optimization theory used in model training
    - **Applications**: Loss function optimization, constraint handling

20. **Golub, G.H., Van Loan, C.F.** (2012). "Matrix Computations." *4th Edition, Johns Hopkins University Press*
    - **Core**: Matrix operations in neural networks
    - **Numerical**: SVD, eigendecomposition for embeddings

#### **Probability & Statistics:**
21. **Bishop, C.M.** (2006). "Pattern Recognition and Machine Learning." *Springer*
    - **Comprehensive**: Bayesian methods, probabilistic models
    - **Relevance**: Model selection, uncertainty quantification

22. **Hastie, T., et al.** (2009). "The Elements of Statistical Learning." *2nd Edition, Springer*
    - **Statistical**: Learning theory foundations
    - **Mathematical**: Generalization bounds, regularization

### 🎯 **Implementation & Engineering References**

#### **TensorFlow & Deep Learning Frameworks:**
23. **Abadi, M., et al.** (2016). "TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems." *OSDI 2016*
    - **Framework**: Computational graph optimization
    - **Engineering**: Distributed training, GPU acceleration

24. **Bisong, E.** (2019). "Building Machine Learning and Deep Learning Models on Google Cloud Platform." *Apress*
    - **Practical**: TensorFlow Hub integration patterns
    - **Cloud**: Scalable model deployment

#### **Video Processing & Computer Vision:**
25. **Szeliski, R.** (2010). "Computer Vision: Algorithms and Applications." *Springer*
    - **Foundation**: Image processing mathematics
    - **Applications**: Feature extraction, object detection

26. **Goodfellow, I., et al.** (2016). "Deep Learning." *MIT Press*
    - **Comprehensive**: Neural network theory
    - **Mathematical**: Backpropagation, optimization landscapes

### 📈 **Evaluation Metrics & Benchmarks**

#### **Standard Datasets & Metrics:**
- **ImageNet**: Visual recognition benchmark (Russakovsky et al., 2015)
- **Kinetics**: Action recognition dataset (Kay et al., 2017)  
- **COCO**: Object detection benchmark (Lin et al., 2014)
- **STS Benchmark**: Semantic textual similarity (Cer et al., 2017)

#### **Performance Evaluation:**
- **mAP**: Mean Average Precision for object detection
- **Top-k Accuracy**: Classification performance metrics
- **BLEU/ROUGE**: Text generation quality
- **Cosine Similarity**: Embedding space evaluation

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
