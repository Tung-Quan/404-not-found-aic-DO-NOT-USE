# 🎬 AI Video Search System

An intelligent video search system that uses AI to find specific moments in videos based on text descriptions. The system uses CLIP (Contrastive Language-Image Pre-training) model to understand both text queries and video content for semantic search.

## 🚀 Features

- **Semantic Video Search**: Find videos using natural language descriptions
- **Frame-level Precision**: Locate exact moments within videos with timestamp accuracy
- **CLIP-powered AI**: Uses OpenAI's CLIP model for understanding visual content
- **Fast Vector Search**: FAISS-based indexing for quick retrieval
- **RESTful API**: Easy-to-use FastAPI endpoints
- **Multiple Search Modes**: Semantic search + optional keyword-based search

## 📋 Prerequisites

- **Python 3.8+**
- **FFmpeg** (for video processing)
- **8GB+ RAM** (for AI models)
- **Git**

### Windows Installation:
```bash
# Install FFmpeg via chocolatey
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

### Linux/Mac Installation:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

## 🛠️ Installation & Setup

### 1. Clone Repository
```bash
git clone https://gitlab.com/404-not-found-2/ai-challenge-404-not-found-2.git
cd ai-challenge-404-not-found-2
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac  
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install torch torchvision transformers
pip install faiss-cpu pandas pyarrow
pip install fastapi uvicorn scikit-learn
pip install pillow tqdm
```

### 4. Prepare Video Files
Place your video files (.mp4, .avi, .mov) in the `videos/` directory:
```
videos/
├── video1.mp4
├── video2.mp4
└── video3.mp4
```

## 🚀 Quick Start Options

### Option 1: Automated Setup (Recommended)
For complete setup from scratch:
```bash
# Windows
.\setup_and_run.bat

# Linux/Mac
chmod +x setup_and_run.sh
./setup_and_run.sh
```

### Option 2: Quick Server Start
If you've already set up the system:
```bash
# Windows
.\start_server.bat

# Linux/Mac
chmod +x start_server.sh
./start_server.sh
```

### Option 3: Manual Setup

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
  mkdir -p "frames/$name"
  ffmpeg -y -i "$video" -vf fps=1 "frames/$name/frame_%06d.jpg"
done
```

### Step 2: Build Metadata Index
```bash
python build_meta.py
```

### Step 3: Generate AI Embeddings
```bash
python scripts/encode_siglip.py
```

### Step 4: Build Search Index
```bash
python scripts/build_faiss.py
```

### Step 5: Start API Server
```bash
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

## ✅ Verify Installation

After starting the server, you should see:
```
INFO:     Application startup complete.
```

Then test these endpoints:
- **Health Check**: http://localhost:8000/health
- **API Documentation**: http://localhost:8000/docs  
- **Root Info**: http://localhost:8000/
- **Search Example**: http://localhost:8000/search?q=coding%20tutorial

## 📁 Project Files

After setup, your project structure should look like:
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
│   ├── video1/
│   │   ├── frame_000001.jpg
│   │   ├── frame_000002.jpg
│   │   └── ...
│   └── video2/
├── videos/                 # Source video files
├── build_meta.py          # Metadata generation script
├── start_server.bat       # Windows server start script
├── setup_and_run.bat      # Windows full setup script
└── README.md              # This documentation
```

## 📚 How It Works

### Architecture Overview
```
Videos → Frame Extraction → AI Embeddings → Vector Index → Search API
  📼         🖼️               🧠            ⚡           🔍
```

### Detailed Pipeline

1. **Frame Extraction**: 
   - FFmpeg extracts 1 frame per second from each video
   - Frames saved as JPG images with timestamp info

2. **Metadata Generation**:
   - `build_meta.py` creates a pandas DataFrame with frame paths, video IDs, and timestamps
   - Saved as `index/meta.parquet`

3. **AI Embedding Creation**:
   - `scripts/encode_siglip.py` uses CLIP model to convert each frame to a 512-dimensional vector
   - Vectors represent the visual content semantically
   - Saved as `index/embeddings/frames.f16.mmap`

4. **Vector Indexing**:
   - `scripts/build_faiss.py` builds a FAISS index for fast similarity search
   - Uses Inner Product (IP) for cosine similarity
   - Saved as `index/faiss/ip_flat.index`

5. **Search API**:
   - `api/app.py` provides FastAPI endpoints
   - Converts text queries to vectors using CLIP
   - Finds similar frames using FAISS
   - Returns ranked video results with timestamps

## 🔧 API Reference

### Search Endpoint

```http
GET /search?q={query}&clip_weight={float}&query_weight={float}&topk_mean={int}&topk_frames={int}
```

### Parameters Explained

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

### Response Format

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
      "best_frame_timestamp": 156.0
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

### Search for Programming Content
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
