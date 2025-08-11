# 🎬 AI Video Search System

Hệ thống tìm kiếm video thông minh sử dụng AI để tìm những khoảnh khắc cụ thể trong video dựa trên mô tả văn bản. Sử dụng mô hình CLIP để hiểu cả nội dung hình ảnh và văn bản.

> ✅ **Portable**: Tất cả batch files sử dụng đường dẫn tương đối - hoạt động trên mọi máy tính!

## ⚡ Khởi Động Nhanh

### 1. Server Tối Ưu Memory (Khuyến Nghị) 
```bash
# Chạy server ổn định, ít tốn memory
start_server_simple.bat

# Truy cập: http://localhost:8001/docs
```

### 2. Server Đầy Đủ Tính Năng
```bash  
# Chạy server với tất cả tính năng (cần nhiều RAM)
start_server_advanced.bat

# Truy cập: http://localhost:8000/docs
```

### 3. Setup Từ Đầu (Nếu Chưa Có Dữ Liệu)
```bash
# Xử lý video, tạo index, khởi động server
setup_and_run.bat
```

## 🎯 So Sánh 2 Phiên Bản

| Tính Năng | Simple (Port 8001) | Advanced (Port 8000) |
|-----------|-------------------|---------------------|
| **Memory** | ⚡ Thấp (~1GB) | 🔥 Cao (~4GB+) |
| **Khởi động** | 🚀 Nhanh (15s) | 🐌 Chậm (60s) |
| **Ổn định** | ✅ Rất ổn định | ⚠️ Có thể lỗi memory |
| **Tìm frame** | ✅ 5 frame tốt nhất | ✅ 5 frame tốt nhất |
| **Tìm video** | ⚡ Tìm cơ bản | 🔥 Tìm nâng cao + TF-IDF |
| **Top frames/video** | ❌ Không | ✅ Top 5 frames mỗi video |
| **Phù hợp** | Sử dụng hàng ngày | Development/Testing |

## 📋 Yêu Cầu Hệ Thống

- **Python 3.8+** 
- **FFmpeg** (xử lý video)
- **4GB+ RAM** (Simple) / **8GB+ RAM** (Advanced)

### Cài Đặt FFmpeg:
```bash
# Windows (với Chocolatey)
choco install ffmpeg

# Hoặc tải từ: https://ffmpeg.org/download.html
```

## � Hướng Dẫn Cài Đặt Từng Bước

### 📋 Yêu Cầu Hệ Thống
- ✅ **Python 3.8+** đã cài đặt
- ✅ **FFmpeg** đã cài đặt (xử lý video)
- ✅ **4GB+ RAM** (8GB+ khuyến nghị cho Advanced server)
- ✅ **Kết nối Internet** (tải AI model lần đầu)

### 🔧 Cài Đặt Nhanh (5 Phút)

#### Bước 1: Tải Project
```bash
# Clone từ GitLab
git clone https://gitlab.com/404-not-found-2/ai-challenge-404-not-found-2.git
cd ai-challenge-404-not-found-2

# Hoặc tải ZIP và giải nén
```

#### Bước 2: Tạo Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac  
python3 -m venv .venv
source .venv/bin/activate
```

#### Bước 4: Chuẩn Bị Video Files
```bash
# Tạo thư mục videos (nếu chưa có)
mkdir videos

# Đặt file video vào thư mục videos/
# Hỗ trợ: .mp4, .avi, .mov, .mkv
```

#### Bước 5: Khởi Động Lần Đầu (Chọn 1 Trong 2)

**Option A: Tự Động (Khuyến Nghị)**
```bash
# Xử lý video, tạo index, khởi động server một lần
setup_and_run.bat
```

**Option B: Thủ Công (Nếu Muốn Hiểu Từng Bước)**
```bash
# 1. Xử lý video thành frames
python extract_frames.py

# 2. Tạo metadata index  
python build_meta.py

# 3. Tạo AI embeddings
python build_embeddings.py

# 4. Khởi động server
start_server_simple.bat
```

#### Bước 6: Sử Dụng Hàng Ngày
```bash
# Chỉ cần chạy server (dữ liệu đã có)
start_server_simple.bat

# Truy cập: http://localhost:8001/docs
```

### ⚡ Quick Start (Cho Người Đã Biết)
```bash
git clone https://gitlab.com/404-not-found-2/ai-challenge-404-not-found-2.git
cd ai-challenge-404-not-found-2
python -m venv .venv && .venv\Scripts\activate  
pip install -r requirements.txt
# Đặt videos vào videos/
setup_and_run.bat
```

## 🌐 API Endpoints

### Server Simple (Port 8001)
- `GET /docs` - 📚 API Documentation  
- `GET /health` - ✅ Trạng thái server
- `GET /search_frames?q=query` - 🔍 Tìm frame riêng lẻ
- `GET /search_simple?q=query` - 🎯 Tìm video cơ bản

### Server Advanced (Port 8000)  
- `GET /docs` - 📚 API Documentation
- `GET /health` - ✅ Trạng thái server
- `GET /search?q=query` - 🔥 Tìm video nâng cao + aggregation
- `GET /search_frames?q=query` - 🔍 Tìm frame riêng lẻ

## 💡 Ví Dụ Sử Dụng

### Tìm Frame
```bash
# Tìm 5 frame gần nhất với query
curl "http://localhost:8001/search_frames?q=person walking&top_frames=5"
```

### Tìm Video (Advanced)
```bash
# Tìm video với top 5 frames tốt nhất mỗi video
curl "http://localhost:8000/search?q=car driving&clip_weight=0.8"
```

## 🆘 Xử Lý Lỗi

| Lỗi | Nguyên Nhân | Giải Pháp |
|-----|-------------|-----------|
| **Memory Error** | Không đủ RAM cho Advanced server | Dùng `start_server_simple.bat` |
| **Port Conflict** | Port đã được sử dụng | Thay đổi port trong file .bat |
| **No Videos Found** | Không có video trong thư mục | Đặt file .mp4 vào `videos/` |
| **Missing Index** | Chưa xử lý video | Chạy `setup_and_run.bat` |
| **FFmpeg Error** | Chưa cài FFmpeg | Cài FFmpeg từ ffmpeg.org |
- `GET /search?q=query` - Advanced video search with aggregation
- `GET /search_frames?q=query` - Individual frame search
- `GET /docs` - API documentation

## 🛠️ Manual Setup (If Needed)

### Step 1: Extract Video Frames
```bash
# Windows PowerShell
$ErrorActionPreference = "Stop"
Get-ChildItem videos -File | ForEach-Object {
  $name = $_.BaseName
  New-Item -ItemType Directory -Force -Path "frames/$name" | Out-Null
  ffmpeg -y -i $_.FullName -vf fps=1 "frames/$name/frame_%06d.jpg"
}

# Linux/Mac Bash
for video in videos/*; do
  name=$(basename "$video" | sed 's/\.[^.]*$//')
```
## 📁 Cấu Trúc Project

```
📦 AI Video Search
├── 🚀 start_server_simple.bat     # Server tối ưu memory  
├── 🔥 start_server_advanced.bat   # Server đầy đủ tính năng
├── 🎛️ start_server.bat            # Menu lựa chọn
├── 🛠️ setup_and_run.bat           # Setup từ đầu
├── 📂 api/
│   ├── app_simple.py              # Code server tối ưu
│   └── app.py                     # Code server advanced
├── 📂 videos/                     # Đặt file video vào đây
├── 📂 frames/                     # Frame được extract
├── 📂 index/                      # Metadata và indexes
│   ├── meta.parquet              # Thông tin frames
│   ├── embeddings/               # AI vectors
│   └── faiss/                    # Search index
└── 📚 README.md                  # File này
```

## ✅ Kiểm Tra Hoạt Động

Sau khi khởi động server, bạn sẽ thấy:
```
✅ API server ready!
🔧 Memory optimized version loaded  
📊 Ready to search 14402 frames
```

Test các endpoints:
- **Health Check**: http://localhost:8001/health
- **API Docs**: http://localhost:8001/docs  
- **Tìm Frame**: http://localhost:8001/search_frames?q=person walking

## � Liên Kết

- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **CLIP Model**: https://github.com/openai/CLIP
- **FAISS**: https://github.com/facebookresearch/faiss

## 📞 Hỗ Trợ

Nếu gặp vấn đề, hãy:
1. Kiểm tra file `QUICK_START.md` 
2. Dùng server Simple thay vì Advanced
3. Đảm bảo đã cài FFmpeg
4. Kiểm tra log lỗi trong terminal

## 🆘 Xử Lý Lỗi Thường Gặp

### ❌ Memory Error (Lỗi Bộ Nhớ)
```
OSError: [WinError 8] Not enough memory resources
```
**Giải pháp**: Dùng Simple Server
```bash
start_server_simple.bat  # Thay vì start_server_advanced.bat
```

### ❌ Port Already in Use (Port Đã Được Sử Dụng)
```
OSError: [Errno 48] Address already in use
```
**Giải pháp**: 
1. Đóng server cũ (Ctrl+C)
2. Hoặc đổi port trong file .bat

### ❌ FFmpeg Not Found
```
'ffmpeg' is not recognized as an internal or external command
```
**Giải pháp**:
```bash
# Windows (Chocolatey)
choco install ffmpeg

# Hoặc tải từ https://ffmpeg.org/download.html
# Thêm vào PATH environment variable
```

### ❌ Python Not Found
```
'python' is not recognized as an internal or external command
```
**Giải pháp**: Cài Python 3.8+ từ https://python.org

### ❌ No Videos Found
```
WARNING: No video files found in videos/
```
**Giải pháp**: Đặt file .mp4/.avi/.mov vào thư mục `videos/`

### ❌ Missing Dependencies
```
ModuleNotFoundError: No module named 'torch'
```
**Giải pháp**:
```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

## 🔗 Liên Kết Hữu Ích

- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **CLIP Model**: https://github.com/openai/CLIP
- **FAISS**: https://github.com/facebookresearch/faiss
- **GitLab Repository**: https://gitlab.com/404-not-found-2/ai-challenge-404-not-found-2

## 📞 Hỗ Trợ

Nếu gặp vấn đề:
1. 📖 Đọc phần "Xử Lý Lỗi" ở trên
2. ⚡ Dùng Simple Server thay vì Advanced
3. 🔍 Kiểm tra log lỗi trong terminal
4. 💬 Tạo issue trên GitLab

---
**Made with ❤️ using AI and CLIP** 🤖
   - Saved as `index/faiss/ip_flat.index`

5. **Search API**:
   - `api/app.py` provides FastAPI endpoints
   - Converts text queries to vectors using CLIP
   - Finds similar frames using FAISS
   - Returns ranked video results with timestamps

## 🔧 API Reference

### Video Search Endpoint (Grouped by Video)

```http
GET /search?q={query}&clip_weight={float}&query_weight={float}&topk_mean={int}&topk_frames={int}
```

### Frame Search Endpoint (Individual Frames) 🆕

```http
GET /search_frames?q={query}&top_frames={int}
```

Returns the top N individual frames closest to your query, potentially from different videos or the same video.

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `q` | string | **required** | - | Text query to search for |
| `top_frames` | int | 5 | 1-100 | Number of top frames to return |

### Video Search Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `q` | string | **required** | - | Text query to search for |
| `clip_weight` | float | 1.0 | 0.0+ | Weight for semantic (CLIP) search |
| `query_weight` | float | 0.0 | 0.0+ | Weight for keyword (TF-IDF) search |
| `topk_mean` | int | 200 | 1-2000 | Number of top frames to average per video |
| `topk_frames` | int | 8000 | 100-50000 | Initial frames retrieved from index |

### Parameter Effects

#### `clip_weight` vs `query_weight`
```bash
# Pure semantic search (understands meaning)
clip_weight=1.0, query_weight=0.0

# Balanced search (meaning + keywords)  
clip_weight=0.7, query_weight=0.3

# Pure keyword search (exact text matching)
clip_weight=0.0, query_weight=1.0
```

#### `topk_mean` (Precision Control)
```bash
topk_mean=50   # Conservative: Top 50 frames → Basic precision
topk_mean=200  # Balanced: Top 200 frames → Good for 7000+ frame videos  
topk_mean=500  # Comprehensive: Top 500 frames → Maximum recall
```

#### `topk_frames` (Search Scope)
```bash
topk_frames=2000  # Fast: Limited scope → Quick results
topk_frames=8000  # Balanced: Good for 7000+ frame videos → Recommended
topk_frames=15000 # Comprehensive: Full coverage → Slower but thorough
```

### Video Search Response Format

```json
{
  "query": "React components tutorial",
  "clip_weight": 1.0,
  "query_weight": 0.0,
  "topk_mean": 50,
  "results": [
    {
      "video_id": "React_Tutorial_Video",
      "video_path": "videos/React_Tutorial_Video.mp4",
      "score": 0.87,
      "frames_used": 25,
      "best_frame_path": "frames/React_Tutorial_Video/frame_000156.jpg",
      "best_frame_timestamp": 156.0,
      "top_frames": [
        {
          "frame_path": "frames/React_Tutorial_Video/frame_000156.jpg",
          "timestamp": 156.0,
          "score": 0.89
        },
        {
          "frame_path": "frames/React_Tutorial_Video/frame_000157.jpg",
          "timestamp": 157.0,
          "score": 0.87
        },
        {
          "frame_path": "frames/React_Tutorial_Video/frame_000158.jpg",
          "timestamp": 158.0,
          "score": 0.85
        },
        {
          "frame_path": "frames/React_Tutorial_Video/frame_000159.jpg",
          "timestamp": 159.0,
          "score": 0.83
        },
        {
          "frame_path": "frames/React_Tutorial_Video/frame_000160.jpg",
          "timestamp": 160.0,
          "score": 0.81
        }
      ]
    }
  ]
}
```

### Frame Search Response Format 🆕

```json
{
  "query": "React components",
  "total_frames_searched": 5,
  "results": [
    {
      "frame_path": "frames/React_Tutorial/frame_000156.jpg",
      "video_id": "React_Tutorial", 
      "timestamp": 156.0,
      "score": 0.89,
      "video_path": "videos/React_Tutorial"
    },
    {
      "frame_path": "frames/JavaScript_Guide/frame_000245.jpg",
      "video_id": "JavaScript_Guide",
      "timestamp": 245.0, 
      "score": 0.87,
      "video_path": "videos/JavaScript_Guide"
    }
  ]
}
```

### Additional Endpoints

```http
# Get video file
GET /videos/{filename}

# Get frame image  
GET /frames/{video_id}/{frame_filename}

# Get video thumbnail
GET /thumbnail/{video_id}

# API documentation
GET /docs
```

## 💡 Usage Examples

### Search for Individual Frames 🆕
```bash
# Find 5 most relevant frames (can be from different videos)
curl "http://localhost:8000/search_frames?q=React components&top_frames=5"

# Find 10 frames about Python programming
curl "http://localhost:8000/search_frames?q=Python tutorial&top_frames=10"

# Quick search for coding content
curl "http://localhost:8000/search_frames?q=coding&top_frames=3"
```

### Search for Programming Content (Grouped by Video)
```bash
curl "http://localhost:8000/search?q=Python programming tutorial"
curl "http://localhost:8000/search?q=React components and hooks"
curl "http://localhost:8000/search?q=machine learning explanation"
```

### Adjust Search Behavior
```bash
# Optimized for 7000+ frame videos (recommended)
curl "http://localhost:8000/search?q=coding tutorial&topk_mean=200&topk_frames=8000"

# High precision search (strict matching)
curl "http://localhost:8000/search?q=coding tutorial&topk_mean=100&topk_frames=5000"

# Comprehensive search (maximum coverage)
curl "http://localhost:8000/search?q=coding tutorial&topk_mean=500&topk_frames=15000"

# Combine semantic + keyword search
curl "http://localhost:8000/search?q=React&clip_weight=0.8&query_weight=0.2"
```

### Frontend Integration
```javascript
// Search videos
const response = await fetch('/search?q=React components');
const data = await response.json();

// Display results with thumbnails
data.results.forEach(result => {
  const thumbnail = `/thumbnail/${result.video_id}`;
  const videoUrl = `/videos/${result.video_id}`;
  const jumpTime = result.best_frame_timestamp;
  
  // Create video player that jumps to specific moment
  const video = document.createElement('video');
  video.src = videoUrl;
  video.currentTime = jumpTime;
});
```

## 🗂️ Project Structure

```
├── api/
│   └── app.py              # FastAPI server
├── scripts/
│   ├── encode_siglip.py    # AI embedding generation
│   ├── build_faiss.py      # Vector index creation
│   └── text_embed.py       # Text embedding utilities
├── index/
│   ├── meta.parquet        # Frame metadata
│   ├── embeddings/         # AI-generated vectors
│   │   └── frames.f16.mmap
│   └── faiss/              # Search indexes
│       └── ip_flat.index
├── frames/                 # Extracted video frames
├── videos/                 # Source video files
├── build_meta.py          # Metadata generation script
└── README.md              # This documentation
```

## 🛠️ Troubleshooting

### Common Issues

**1. FFmpeg not found**
```bash
# Windows: Install via chocolatey or download binary
choco install ffmpeg

# Linux: Install via package manager
sudo apt install ffmpeg
```

**2. CUDA out of memory**
```python
# In scripts/encode_siglip.py, use CPU instead
DEVICE = 'cpu'  # Change from 'cuda'
```

**3. Import errors**
```bash
# Reinstall dependencies
pip install --upgrade torch transformers faiss-cpu
```

**4. Slow search performance**
```bash
# For 7000+ frame videos, use optimized parameters
curl "http://localhost:8000/search?q=tutorial&topk_frames=5000&topk_mean=150"

# For faster results with good accuracy
curl "http://localhost:8000/search?q=tutorial&topk_frames=3000&topk_mean=100"
```

### Performance Tuning

**For Large Video Collections (7000+ frames per video):**
- Use `topk_frames=8000-15000` for comprehensive search
- Set `topk_mean=200-500` for balanced precision/recall
- Use GPU if available (`DEVICE = 'cuda'`)
- Consider using `faiss-gpu` instead of `faiss-cpu`

**For Real-time Applications:**
- Use `topk_frames=3000-5000` for faster response
- Set `topk_mean=100-200` for good accuracy
- Cache frequent queries
- Use smaller batch sizes during indexing

## 🔬 Technical Details

### AI Models Used
- **CLIP**: `openai/clip-vit-base-patch32`
  - 512-dimensional embeddings
  - Understands both text and images
  - Pre-trained on 400M image-text pairs

### Vector Search
- **FAISS IndexFlatIP**: Inner product similarity
- **Normalization**: L2 normalized vectors for cosine similarity
- **Search complexity**: O(N) where N = number of frames

### File Formats
- **Embeddings**: Float16 memory-mapped files (.mmap)
- **Metadata**: Parquet format for fast loading
- **Index**: FAISS binary format

## 🧮 **TOÁN HỌC VÀ KIẾN THỨC ẨN TRONG HỆ THỐNG**

### **1. 🎯 CLIP MODEL - CONTRASTIVE LEARNING**

#### **A. Text Embedding Process:**
```python
# Quá trình chuyển đổi text thành vector 512 chiều
def embed_text(query: str) -> np.ndarray:
    # 1. Tokenization: "person walking" → [49406, 2533, 3788, 49407]
    # 2. Transformer: tokens → hidden_states (512 dim)
    # 3. Normalization: vector / ||vector||₂
```

**🔬 Toán học ẩn:**

**Self-Attention Mechanism:**
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V

Trong đó:
- Q = Query matrix (512 dim)
- K = Key matrix (512 dim)  
- V = Value matrix (512 dim)
- d_k = 512 (dimension of keys)
- √d_k = √512 ≈ 22.6 (scaling factor)
```

**Layer Normalization:**
```
LN(x) = γ * (x - μ)/σ + β

μ = mean(x)     # Trung bình của vector
σ = std(x)      # Độ lệch chuẩn  
γ, β = learnable parameters
```

**L2 Normalization (Final Step):**
```
normalized_vector = vector / ||vector||₂

||vector||₂ = √(∑ᵢ₌₁⁵¹² vᵢ²)  # Euclidean norm

Kết quả: ||normalized_vector||₂ = 1.0
```

#### **B. Vision Transformer (ViT) cho Images:**

**Patch Embedding:**
```
Image (224×224×3) → Patches (196×768)

- Mỗi patch = 16×16 pixels
- Số patches = (224/16)² = 196 patches  
- Embedding dim = 768 → project to 512
```

**Positional Encoding:**
```
PE(pos,2i) = sin(pos/10000^(2i/d_model))
PE(pos,2i+1) = cos(pos/10000^(2i/d_model))

pos = position of patch (0-195)
i = dimension index (0-255)  
d_model = 512
```

### **2. 🎯 FAISS VECTOR SEARCH**

#### **A. Cosine Similarity via Inner Product:**

```python
# Với normalized vectors: cosine_similarity = inner_product
similarity = query_vector · frame_vector = ∑ᵢ₌₁⁵¹² qᵢ × fᵢ
```

**🔬 Toán học ẩn:**

**Cosine Similarity Formula:**
```
cos_sim(a,b) = (a·b)/(||a|| × ||b||)

Vì vectors đã normalized (||a|| = ||b|| = 1):
cos_sim(a,b) = a·b = inner_product(a,b)

Kết quả ∈ [-1, 1]:
- 1.0 = hoàn toàn giống nhau
- 0.0 = không liên quan  
- -1.0 = hoàn toàn đối lập
```

**FAISS IndexFlatIP Search:**
```
Brute Force Algorithm:
for each frame_vector in database:
    score = query_vector · frame_vector  
    if score > threshold:
        add (score, frame_index) to results

Sort results by score (descending)
Return top_k results

Time Complexity: O(N × D)  
N = số frames (~14,402)
D = embedding dimension (512)
```

#### **B. Memory-Mapped Storage:**

```python
# Float16 precision cho tiết kiệm memory
mem = np.memmap(path, dtype='float16', shape=(N, 512))
```

**🔬 Toán học ẩn:**

**Float16 vs Float32:**
```
Float16: 1 bit sign + 5 bits exponent + 10 bits mantissa
- Range: ±6.55×10⁴
- Precision: ~3-4 decimal digits
- Memory: N × 512 × 2 bytes = N × 1KB per frame

Float32: 1 bit sign + 8 bits exponent + 23 bits mantissa  
- Range: ±3.4×10³⁸
- Precision: ~6-7 decimal digits
- Memory: N × 512 × 4 bytes = N × 2KB per frame

Quantization Error: |float32_value - float16_value| < 0.1%
```

### **3. 🎯 SCORE AGGREGATION & RANKING**

#### **A. Video-Level Scoring:**

```python
# Max pooling strategy
video_score = max(frame_scores)
= max{cos_sim(query, frame₁), cos_sim(query, frame₂), ...}
```

**🔬 Alternative Aggregation Methods:**

**Mean Pooling:**
```
video_score = (1/n) × ∑ᵢ₌₁ⁿ cos_sim(query, frameᵢ)
```

**Weighted Average:**
```
video_score = ∑ᵢ₌₁ⁿ wᵢ × cos_sim(query, frameᵢ)
where ∑wᵢ = 1
```

**Top-K Mean (Advanced Server):**
```python
# topk_mean = 50 (parameter)
sorted_scores = sorted(frame_scores, reverse=True)
top_k_scores = sorted_scores[:50]
video_score = mean(top_k_scores)
```

#### **B. Statistical Interpretation:**

**Score Ranges:**
```
Score ≥ 0.9:  Excellent match (>90% similarity)
Score 0.7-0.9: Good match (70-90% similarity)  
Score 0.5-0.7: Moderate match (50-70% similarity)
Score < 0.5:   Poor match (<50% similarity)
```

### **4. 🎯 BATCH PROCESSING & OPTIMIZATION**

#### **A. GPU Memory Management:**

```python
# Batch processing để tối ưu GPU
BATCH_SIZE = 64  # Tùy thuộc GPU memory
```

**🔬 Memory Calculations:**

**Single Image Processing:**
```
Input: 224×224×3×4 bytes = 600KB (float32)
Intermediate activations: ~50-100MB per image
Peak GPU memory: ~2-4GB for CLIP-ViT-Base
```

**Batch Processing:**
```
Batch of 64 images:
- Input memory: 64 × 600KB = 38.4MB  
- Activation memory: 64 × 100MB = 6.4GB
- Total GPU memory needed: ~8-10GB
```

#### **B. Numerical Stability:**

**Normalization with Epsilon:**
```python
# Tránh division by zero
normalized = vector / (||vector||₂ + ε)
where ε = 1e-12
```

**Gradient Flow:**
```
∂Loss/∂vector = ∂Loss/∂normalized × ∂normalized/∂vector

∂normalized/∂vector = (I - normalized⊗normalized) / ||vector||
```

### **5. 🎯 ADVANCED CONCEPTS**

#### **A. Contrastive Learning (CLIP Training):**

**Contrastive Loss Function:**
```
L = -log(exp(sim(text,image⁺)/τ) / ∑ⱼ exp(sim(text,imageⱼ)/τ))

τ = temperature parameter = 0.07
image⁺ = positive pair (correct image for text)
imageⱼ = all images in batch (including negatives)
```

**Symmetrical Training:**
```
Total_Loss = L(text→image) + L(image→text)
```

#### **B. Dimensional Analysis:**

**512-Dimensional Embedding Space:**
```
Information capacity: 2^(512×16) possible vectors (float16)
≈ 10^2466 possible embeddings

Unit sphere S^511:
- Most vectors are nearly orthogonal
- Average cosine similarity ≈ 0 for random vectors
- Meaningful clusters form in high-density regions
```

### **6. 🎯 PERFORMANCE METRICS**

#### **A. Search Performance:**

```
Query Processing: ~50-100ms (CPU) / ~10-20ms (GPU)
FAISS Search: ~5-15ms for 14K frames
Total Response: ~100-200ms per query

Throughput: ~10-50 queries/second (depending on hardware)
```

#### **B. Memory Usage:**

```
Metadata (meta.parquet): ~1-2MB  
Embeddings (14K×512×2): ~14.8MB
FAISS Index: ~15-20MB
Model weights: ~150MB (CLIP)
Runtime memory: 1-4GB (Simple vs Advanced)
```

### **📊 Tóm Tắt Công Thức Chính:**

1. **Text/Image → Embedding**: `CLIP(input) → normalize(vector₅₁₂)`
2. **Similarity Search**: `score = query · frame` (inner product)  
3. **Video Ranking**: `video_score = max(frame_scores)` or `mean(top_k)`
4. **Memory Efficiency**: `float16` storage, memory mapping
5. **Batch Processing**: Parallel GPU computation for speed

**Hệ thống kết hợp Linear Algebra, Deep Learning, Information Retrieval, và Computer Vision để tạo ra AI search engine mạnh mẽ!**

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- **OpenAI** for the CLIP model
- **Facebook Research** for FAISS vector search
- **FastAPI** for the web framework
- **FFmpeg** for video processing

---

**Built with ❤️ for intelligent video search**
