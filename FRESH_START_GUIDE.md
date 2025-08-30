# 🚀 **Hướng Dẫn Bắt Đầu Lại Từ Con Số 0 với 4 Video Mới**

## ⚡ **PHƯƠNG PHÁP ĐỀ XUẤT - Đơn Giản Nhất (Chỉ 3 Bước)**

### **� Đây Là Cách Bạn Nên Làm**

```bash
# Bước 1: Copy 4 video mới vào thư mục videos/ (thay thế video cũ)
# Drag & drop hoặc copy video mới vào videos/

# Bước 2: Xóa folder frames để hệ thống rebuild
# Windows: Delete thư mục frames/ bằng File Explorer
# Hoặc: Remove-Item -Recurse -Force frames

# Bước 3: Chạy lại setup - Tự động rebuild mọi thứ
python setup.py
```

### **📝 Chi Tiết Từng Bước**

#### **Bước 1: Thay Thế Video**
```bash
# Cách 1: Drag & Drop (Dễ nhất)
# - Mở thư mục videos/
# - Xóa 4 video cũ
# - Drag & drop 4 video mới vào

# Cách 2: Command Line
# Windows PowerShell:
Copy-Item "C:\path\to\your\new\videos\*" "videos\"

# Linux/Mac:
cp /path/to/your/new/videos/* videos/
```

#### **Bước 2: Xóa Frames Cũ**
```bash
# Windows - File Explorer:
# Chuột phải folder "frames" → Delete

# Windows - PowerShell:
Remove-Item -Recurse -Force frames

# Linux/Mac:
rm -rf frames/
```

#### **Bước 3: Rebuild Tự Động**
```bash
# Kích hoạt virtual environment (nếu chưa)
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate     # Linux/Mac

# Chạy setup - Tự động xử lý mọi thứ
python setup.py

# Setup sẽ tự động:
# ✅ Phát hiện video mới trong videos/
# ✅ Extract frames từ 4 video mới
# ✅ Build embeddings với CLIP model
# ✅ Tạo FAISS index cho search
# ✅ Chuẩn bị sẵn sàng cho web interface
```

### **✅ Kết Quả Sau Khi Hoàn Thành**

```
Project/
├── videos/                        # 4 video mới của bạn
│   ├── your_new_video1.mp4
│   ├── your_new_video2.mp4  
│   ├── your_new_video3.mp4
│   └── your_new_video4.mp4
├── frames/                        # Frames mới (tự động tạo)
│   ├── your_new_video1/
│   │   ├── frame_000001.jpg
│   │   ├── frame_000002.jpg
│   │   └── ...
│   ├── your_new_video2/
│   ├── your_new_video3/
│   └── your_new_video4/
├── index/                         # Index mới (tự động tạo)
│   ├── metadata.json             # Metadata của frames mới
│   └── faiss/                    # FAISS search index
├── embeddings/                    # Embeddings mới (tự động tạo)
└── models_cache/                  # Models cache (giữ nguyên)
```

### **🚀 Test Hệ Thống Mới**

```bash
# Sau khi setup xong, start web interface
python web_interface.py

# Truy cập: http://localhost:8080
# Test search với nội dung video mới
```

---

## 📦 **PHƯƠNG PHÁP BACKUP (Nếu Muốn Giữ Dữ Liệu Cũ)**

### **🔒 An Toàn Hơn - Backup Trước Khi Xóa**

```bash
# Bước 1: Tạo thư mục backup với timestamp
mkdir backup_$(Get-Date -Format "yyyyMMdd_HHmmss")  # Windows PowerShell
mkdir backup_$(date +%Y%m%d_%H%M%S)                # Linux/Mac

# Bước 2: Backup dữ liệu quan trọng
# Windows PowerShell:
Move-Item videos backup_*/videos_old
Move-Item frames backup_*/frames_old  
Move-Item index backup_*/index_old
Move-Item embeddings backup_*/embeddings_old

# Linux/Mac:
mv videos backup_*/videos_old
mv frames backup_*/frames_old
mv index backup_*/index_old  
mv embeddings backup_*/embeddings_old

# Bước 3: Tạo thư mục mới và setup
mkdir videos frames index embeddings
# Copy video mới vào videos/
python setup.py
```

### **🗂️ Hoặc Tạo Dataset Song Song**

```bash
# Giữ nguyên dataset cũ, tạo dataset mới
mkdir videos_new frames_new index_new embeddings_new

# Copy video mới vào videos_new/
# Sẽ cần script đặc biệt để xử lý (hướng dẫn bên dưới)
```

---

## � **Câu Hỏi Thường Gặp**

### **❓ Tôi Có Cần Xóa Folder Nào Không?**

**Đáp**: **KHÔNG bắt buộc!** Bạn có 2 lựa chọn:

#### **🎯 Lựa chọn 1: Chạy đè (Đề xuất)**
```bash
# Chỉ cần copy video mới vào videos/ và chạy
python setup.py

# Hệ thống tự động ghi đè:
# ✅ frames/ → Frames mới từ video mới
# ✅ index/ → Metadata và index mới  
# ✅ embeddings/ → Embeddings mới
```

#### **🔒 Lựa chọn 2: Xóa clean (An toàn hơn)**
```bash
# Xóa để đảm bảo không bị conflict
Remove-Item -Recurse -Force frames  # Windows
rm -rf frames/                      # Linux/Mac

# Rồi chạy setup
python setup.py
```

### **❓ Folder Models_cache/ Có Cần Xóa Không?**

**Đáp**: **KHÔNG!** Giữ nguyên để tránh download lại models:
```bash
# GIỮ NGUYÊN - Đừng xóa
models_cache/           # CLIP, BLIP models đã download
.venv/                  # Python environment
config/                 # Configuration files
```

### **❓ File frames/search_index_lite.json và .gitkeep Có Cần Tạo Lại Không?**

**Đáp**: **TỰ ĐỘNG TẠO LẠI!** Không cần lo lắng:

#### **📄 search_index_lite.json**
```python
# File này chứa metadata của frames cho AI Search Lite
# SẼ ĐƯỢC TẠO TỰ ĐỘNG khi:
# 1. Chạy ai_search_lite.py
# 2. Process frames mới
# 3. Extract features (brightness, contrast, color...)

# Vị trí: frames/search_index_lite.json
# Tạo bởi: ai_search_lite.py → _save_index()
```

#### **📁 .gitkeep** 
```bash
# File này chỉ để Git tracking folder rỗng
# CÓ THỂ TẠO LẠI đơn giản:

# Windows PowerShell:
New-Item -ItemType File -Path "frames/.gitkeep" -Force

# Linux/Mac:
touch frames/.gitkeep

# Hoặc để Git tự tạo khi commit
```

#### **⚡ Quy Trình Tự Động:**
```bash
# Khi xóa folder frames/ và chạy setup.py:
# 1. setup.py → tạo folder frames/ mới
# 2. Extract frames → frame_000001.jpg, frame_000002.jpg...
# 3. ai_search_lite.py → tự động tạo search_index_lite.json
# 4. .gitkeep → tạo thủ công (không bắt buộc)
```

### **❓ Folder datasets/ Có Cần Thay Đổi Gì Không?**

**Đáp án**: **TÙY CHỌN!** Đây là system quản lý multi-dataset:

#### **📊 Hiện Tại: Multi-Dataset System**
```bash
datasets/
├── mixed_collection/        # Collection 1
│   ├── videos/             # Videos của collection này
│   ├── frames/             # Frames riêng
│   ├── index/              # Index riêng  
│   └── config.json         # Metadata
├── nature_collection/       # Collection 2
│   ├── videos/
│   ├── frames/ 
│   ├── index/
│   └── config.json
└── people_collection/       # Collection 3
    └── ...
```

#### **🎯 2 Lựa Chọn:**

**🔹 Lựa chọn 1: Chỉ dùng Default Dataset (Đơn giản)**
```bash
# KHÔNG CẦN động vào datasets/
# Chỉ cần làm việc với:
videos/                     # ← 4 video mới ở đây
frames/                     # ← Xóa và tạo lại
index/                      # ← Tự động ghi đè
embeddings/                 # ← Tự động ghi đè

# datasets/ → GIỮ NGUYÊN (không ảnh hưởng)
```

**🔹 Lựa chọn 2: Reset Cả Multi-Dataset System**
```bash
# Nếu muốn clean hoàn toàn:
Remove-Item -Recurse -Force datasets    # Windows
rm -rf datasets/                        # Linux

# Khi chạy web_interface.py → tự tạo lại datasets/ rỗng
```

#### **⚡ Khuyến Nghị:**
```bash
# 🎯 GIỮ NGUYÊN datasets/ 
# Vì:
# ✅ Không ảnh hưởng default dataset
# ✅ Có thể dùng multi-dataset sau này
# ✅ Web interface vẫn hoạt động bình thường
# ✅ Không cần setup thêm gì

# Chỉ cần focus vào 3 bước chính:
# 1. videos/ ← 4 video mới
# 2. frames/ ← Xóa
# 3. python setup.py ← Chạy
```

### **❓ Có Bị Conflict Giữa Dữ Liệu Cũ và Mới Không?**

**Đáp**: **KHÔNG!** Setup.py thông minh:
```python
# setup.py tự động:
# 1. Detect video files trong videos/
# 2. Ghi đè frames cũ với frames mới
# 3. Rebuild embeddings hoàn toàn mới
# 4. Overwrite metadata và index
# → Không có conflict!
```

### **❓ Bao Lâu Setup Mới Hoàn Thành?**

**Đáp**: Tùy thuộc vào video:
```bash
# Ước tính thời gian:
- Extract frames: ~1-2 phút/video
- Build embeddings: ~3-5 phút (GPU RTX 3060)
- Create index: ~30 giây

# Tổng: ~15-20 phút cho 4 video
```

---

## 🚀 **Script Tự Động Hoàn Chỉnh**

### **📜 Script All-in-One**

```bash
# Tạo file: quick_reset.ps1 (Windows)
# Copy 4 video mới vào videos/
Write-Host "🎬 Copying new videos..."
# (Manual step: user copies videos)

# Xóa frames cũ để rebuild clean
Write-Host "🗑️ Removing old frames..."
if (Test-Path "frames") {
    Remove-Item -Recurse -Force frames
}

# Chạy setup tự động
Write-Host "🚀 Running setup..."
python setup.py

# Start web interface
Write-Host "🌐 Starting web interface..."
python web_interface.py
```

```bash
# Tạo file: quick_reset.sh (Linux/Mac)
#!/bin/bash
echo "🎬 Ready for new videos in videos/ folder"
echo "🗑️ Removing old frames..."
rm -rf frames/

echo "🚀 Running setup..."
python setup.py

echo "🌐 Starting web interface..."
python web_interface.py
``` 
# - Build lại embeddings
# - Cập nhật FAISS index

echo "✅ Không cần xóa gì - hệ thống sẽ tự động ghi đè!"
```

### **� Cấu Trúc Thư Mục - Ghi Đè**

```
Project/
├── videos/                        # Thay 4 video cũ bằng 4 video mới
│   ├── new_video1.mp4             # ← Thay thế video cũ
│   ├── new_video2.mp4             # ← Thay thế video cũ  
│   ├── new_video3.mp4             # ← Video mới
│   └── new_video4.mp4             # ← Video mới
├── frames/                        # Sẽ được ghi đè tự động
├── index/                         # Metadata mới sẽ ghi đè
└── embeddings/                    # Embeddings mới sẽ ghi đè
```

### **🗑️ TÙY CHỌN: Xóa Dữ Liệu Cũ (Nếu Muốn Clean Start)**

```bash
# Option 1: Backup dữ liệu cũ (An toàn)
mkdir backup_$(date +%Y%m%d_%H%M%S)
cp -r frames/ backup_$(date +%Y%m%d_%H%M%S)/frames_old
cp -r index/ backup_$(date +%Y%m%d_%H%M%S)/index_old  
cp -r embeddings/ backup_$(date +%Y%m%d_%H%M%S)/embeddings_old

# Sau đó xóa để clean start
rm -rf frames/* index/* embeddings/*

# Option 2: Xóa hoàn toàn (CẢNH BÁO: Mất dữ liệu cũ)
# CHỈ dùng khi chắc chắn không cần dữ liệu cũ
rm -rf frames/ index/ embeddings/
mkdir frames index embeddings

# Option 3: Xóa cache models (nếu muốn download lại models)
rm -rf models_cache/
```

### **🔄 Hoặc Tạo Dataset Song Song**

```bash
# Tạo dataset mới song song với dataset cũ
mkdir videos_new frames_new index_new embeddings_new

# Đặt video mới vào videos_new/
# Sau này có thể switch giữa 2 datasets
```

---

## ⚙️ **Bước 3: Cấu Hình Hệ Thống**

### **🔧 Cập Nhật Configuration**

```python
# Tạo file config_new.py
import os
from pathlib import Path

class NewDatasetConfig:
    """Configuration cho dataset mới"""
    
    # Paths for new dataset
    PROJECT_ROOT = Path(__file__).parent
    VIDEOS_PATH = PROJECT_ROOT / "videos_new"
    FRAMES_PATH = PROJECT_ROOT / "frames_new"
    INDEX_PATH = PROJECT_ROOT / "index_new"
    EMBEDDINGS_PATH = PROJECT_ROOT / "embeddings_new"
    
    # Video processing settings
    FPS_EXTRACTION = 1  # Extract 1 frame per second
    MAX_FRAMES_PER_VIDEO = 1000  # Giới hạn frames mỗi video
    
    # Model settings
    DEFAULT_MODEL = "clip_vit_base"
    BATCH_SIZE = 32
    GPU_MEMORY_FRACTION = 0.8
    
    # Frame processing
    FRAME_SIZE = (224, 224)
    IMAGE_QUALITY = 95
```

---

## 🎬 **Bước 4: Extract Frames từ 4 Video Mới (Ghi Đè)**

### **📝 Script Extract Frames - Phiên Bản Ghi Đè**

```python
# Tạo file: extract_frames_overwrite.py
import cv2
import os
from pathlib import Path
import json
from datetime import datetime

def extract_frames_overwrite_old_data():
    """
    Extract frames từ 4 video mới - GHI ĐÈ dữ liệu cũ
    KHÔNG cần xóa gì trước đó
    """
    
    videos_path = Path("videos")          # Sử dụng thư mục videos hiện tại
    frames_path = Path("frames")          # Ghi đè vào thư mục frames cũ
    index_path = Path("index")            # Ghi đè metadata cũ
    
    # Tạo thư mục nếu chưa có
    frames_path.mkdir(exist_ok=True)
    index_path.mkdir(exist_ok=True)
    
    print("🚀 Starting frame extraction - OVERWRITE MODE")
    print("📁 Videos from:", videos_path)
    print("📁 Frames to:", frames_path)
    print("⚠️  Will overwrite existing data!")
    
    # Xóa nội dung cũ trong frames (giữ lại thư mục)
    import shutil
    for item in frames_path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    
    metadata = []
    total_frames = 0
    
    # Lấy tất cả video files
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(videos_path.glob(ext))
    
    print(f"🎬 Found {len(video_files)} video files")
    
    if len(video_files) == 0:
        print("❌ No video files found in videos/ folder!")
        print("📝 Please put your 4 new videos in the videos/ folder")
        return False
    
    for video_file in video_files:
        print(f"\n🔄 Processing: {video_file.name}")
        
        # Tạo thư mục cho video
        video_name = video_file.stem
        video_frames_dir = frames_path / video_name
        video_frames_dir.mkdir(exist_ok=True)
        
        # Extract frames
        cap = cv2.VideoCapture(str(video_file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0:
            print(f"   ❌ Invalid FPS for {video_file.name}")
            continue
            
        duration = total_video_frames / fps
        
        print(f"   📊 FPS: {fps:.1f}, Total frames: {total_video_frames}, Duration: {duration:.1f}s")
        
        frame_count = 0
        extracted_count = 0
        
        # Extract 1 frame per second
        frame_interval = max(1, int(fps))  # Extract every fps frames (1 per second)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame every second
            if frame_count % frame_interval == 0:
                frame_filename = f"frame_{extracted_count+1:06d}.jpg"
                frame_path = video_frames_dir / frame_filename
                
                # Save frame
                success = cv2.imwrite(str(frame_path), frame)
                
                if success:
                    # Calculate timestamp
                    timestamp = frame_count / fps
                    
                    # Add to metadata
                    metadata.append({
                        "frame_id": f"{video_name}_{extracted_count+1:06d}",
                        "video_name": video_name,
                        "video_path": str(video_file),
                        "frame_path": str(frame_path),
                        "frame_number": frame_count,
                        "timestamp_seconds": timestamp,
                        "timestamp_formatted": f"{int(timestamp//60):02d}:{int(timestamp%60):02d}",
                        "extraction_date": datetime.now().isoformat()
                    })
                    
                    extracted_count += 1
                    
                    if extracted_count % 100 == 0:
                        print(f"   🔄 Extracted {extracted_count} frames...")
            
            frame_count += 1
        
        cap.release()
        total_frames += extracted_count
        print(f"   ✅ Extracted {extracted_count} frames from {video_name}")
    
    # Save metadata (GHI ĐÈ file cũ)
    metadata_path = index_path / "metadata.json"
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Tạo file frames_meta.json cho compatibility
    frames_meta_path = Path("frames") / "search_index_lite.json"
    frames_meta = {
        "total_frames": total_frames,
        "videos": len(video_files),
        "last_updated": datetime.now().isoformat(),
        "metadata": metadata
    }
    
    with open(frames_meta_path, 'w', encoding='utf-8') as f:
        json.dump(frames_meta, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 OVERWRITE EXTRACTION COMPLETE!")
    print(f"📊 Total videos processed: {len(video_files)}")
    print(f"📊 Total frames extracted: {total_frames}")
    print(f"📁 Frames overwritten in: {frames_path}")
    print(f"📄 Metadata overwritten: {metadata_path}")
    print(f"🔄 Old data has been replaced with new data")
    
    return metadata

if __name__ == "__main__":
    print("🚀 FRAME EXTRACTION - OVERWRITE MODE")
    print("=" * 50)
    
    # Kiểm tra thư mục videos
    videos_path = Path("videos")
    if not videos_path.exists():
        print("❌ videos/ folder not found!")
        print("📝 Creating videos/ folder - please put your 4 videos there")
        videos_path.mkdir()
        exit(1)
    
    # Confirm overwrite
    response = input("⚠️  This will OVERWRITE existing frames and metadata. Continue? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("❌ Operation cancelled")
        exit(0)
    
    metadata = extract_frames_overwrite_old_data()
    
    if metadata:
        print(f"\n✅ Ready for next step: build embeddings")
        print(f"▶️  Run: python build_embeddings_overwrite.py")
```

### **🚀 Chạy Frame Extraction (Ghi Đè)**

```bash
# Bước 1: Đặt 4 video mới vào thư mục videos/
cp /path/to/your/new_video1.mp4 videos/
cp /path/to/your/new_video2.mp4 videos/
cp /path/to/your/new_video3.mp4 videos/
cp /path/to/your/new_video4.mp4 videos/

# Bước 2: Chạy script extract (sẽ ghi đè tự động)
python extract_frames_overwrite.py

# Kiểm tra kết quả
ls -la frames/
ls -la index/metadata.json
```

---

## 🧠 **Bước 5: Build Embeddings cho Dataset Mới (Ghi Đè)**

### **📝 Script Build Embeddings - Phiên Bản Ghi Đè**

```python
# Tạo file: build_embeddings_overwrite.py
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
from tqdm import tqdm

def build_embeddings_overwrite():
    """
    Build embeddings cho dataset mới - GHI ĐÈ embeddings cũ
    Sử dụng metadata từ extract_frames_overwrite.py
    """
    
    # Load metadata từ index/metadata.json
    metadata_path = Path("index") / "metadata.json"
    
    if not metadata_path.exists():
        print("❌ metadata.json not found!")
        print("📝 Please run extract_frames_overwrite.py first")
        return False
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"📊 Loading {len(metadata)} frames metadata")
    
    # Initialize CLIP model
    model_name = "openai/clip-vit-base-patch32"
    print(f"🚀 Loading CLIP model: {model_name}")
    
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    print(f"🚀 CLIP model loaded on {device}")
    
    # Tạo thư mục embeddings (ghi đè)
    embeddings_path = Path("embeddings")
    embeddings_path.mkdir(exist_ok=True)
    
    # Xóa embeddings cũ
    for file in embeddings_path.glob("*.npy"):
        file.unlink()
    for file in embeddings_path.glob("*.pkl"):
        file.unlink()
    
    print("🗑️  Cleared old embeddings")
    
    # Build embeddings
    embeddings = []
    valid_metadata = []
    
    batch_size = 32
    
    print("🔄 Building new embeddings...")
    
    for i in tqdm(range(0, len(metadata), batch_size), desc="Processing batches"):
        batch_metadata = metadata[i:i+batch_size]
        batch_images = []
        batch_valid_metadata = []
        
        # Load batch images
        for item in batch_metadata:
            frame_path = Path(item["frame_path"])
            if frame_path.exists():
                try:
                    image = Image.open(frame_path).convert('RGB')
                    batch_images.append(image)
                    batch_valid_metadata.append(item)
                except Exception as e:
                    print(f"⚠️ Error loading {frame_path}: {e}")
        
        if batch_images:
            # Process batch
            inputs = processor(images=batch_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            embeddings.extend(image_features.cpu().numpy())
            valid_metadata.extend(batch_valid_metadata)
    
    embeddings = np.array(embeddings)
    
    print(f"✅ Built embeddings for {len(embeddings)} frames")
    print(f"📊 Embedding shape: {embeddings.shape}")
    
    # Save embeddings (GHI ĐÈ)
    np.save(embeddings_path / "clip_vit_base_embeddings.npy", embeddings)
    
    # Save valid metadata (GHI ĐÈ)
    with open(Path("index") / "valid_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(valid_metadata, f, indent=2, ensure_ascii=False)
    
    # Build FAISS index (GHI ĐÈ)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
    index.add(embeddings.astype('float32'))
    
    # Save FAISS index (GHI ĐÈ)
    faiss_path = Path("index") / "faiss"
    faiss_path.mkdir(exist_ok=True)
    
    # Xóa FAISS index cũ
    for file in faiss_path.glob("*.index"):
        file.unlink()
    for file in faiss_path.glob("*.faiss"):
        file.unlink()
    
    # Save index mới
    faiss.write_index(index, str(faiss_path / "clip_vit_base.index"))
    
    # Tạo file embeddings cho compatibility
    embeddings_dict = {
        "clip_vit_base": {
            "embeddings": embeddings.tolist(),
            "metadata": valid_metadata,
            "model_name": model_name,
            "dimension": dimension,
            "total_frames": len(embeddings)
        }
    }
    
    with open(embeddings_path / "clip_vit_base.json", 'w', encoding='utf-8') as f:
        json.dump(embeddings_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Embeddings saved to: {embeddings_path}")
    print(f"🔍 FAISS index saved to: {faiss_path}")
    print(f"📄 Valid metadata: {len(valid_metadata)} items")
    print(f"🔄 Old embeddings have been replaced with new embeddings")
    
    return embeddings, valid_metadata

if __name__ == "__main__":
    print("🧠 EMBEDDINGS BUILDER - OVERWRITE MODE")
    print("=" * 50)
    
    # Kiểm tra metadata
    metadata_path = Path("index") / "metadata.json"
    if not metadata_path.exists():
        print("❌ No metadata found!")
        print("📝 Please run extract_frames_overwrite.py first")
        exit(1)
    
    # Confirm overwrite
    response = input("⚠️  This will OVERWRITE existing embeddings. Continue? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("❌ Operation cancelled")
        exit(0)
    
    embeddings, metadata = build_embeddings_overwrite()
    
    if embeddings is not False:
        print(f"\n✅ Ready for testing!")
        print(f"▶️  Run: python web_interface.py")
        print(f"🌐 Then visit: http://localhost:8080")
```

### **🚀 Chạy Embedding Builder (Ghi Đè)**

```bash
# Build embeddings cho dataset mới (ghi đè cũ)
python build_embeddings_overwrite.py

# Kiểm tra kết quả
ls -la embeddings/
ls -la index/faiss/
```

---

## 🌐 **Bước 6: Cập Nhật Web Interface**

### **🔧 Sửa web_interface.py**

```python
# Thêm vào web_interface.py
class NewDatasetManager:
    """Manager cho dataset mới"""
    
    def __init__(self):
        self.frames_path = Path("frames_new")
        self.index_path = Path("index_new") 
        self.embeddings_path = Path("embeddings_new")
        
    def load_new_dataset(self):
        """Load dataset mới"""
        # Load metadata
        metadata_path = self.index_path / "valid_metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Load embeddings
        embeddings_file = self.embeddings_path / "clip_vit_base_embeddings.npy"
        self.embeddings = np.load(embeddings_file)
        
        # Load FAISS index
        faiss_file = self.index_path / "faiss" / "clip_vit_base.index"
        self.faiss_index = faiss.read_index(str(faiss_file))
        
        print(f"📊 Loaded new dataset: {len(self.metadata)} frames")
        
        return True

# Thêm endpoint mới
@app.post("/api/switch_dataset")
async def switch_to_new_dataset():
    """Switch to new dataset"""
    global search_engine
    
    try:
        new_manager = NewDatasetManager()
        success = new_manager.load_new_dataset()
        
        if success:
            # Update search engine
            search_engine.frame_records = new_manager.metadata
            search_engine.embeddings = new_manager.embeddings
            search_engine.faiss_index = new_manager.faiss_index
            
            return {
                "status": "success", 
                "message": f"Switched to new dataset with {len(new_manager.metadata)} frames"
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}
```

### **🖥️ Thêm Button Switch Dataset trong UI**

```html
<!-- Thêm vào templates/index.html -->
<div class="dataset-controls">
    <h3>📊 Dataset Management</h3>
    <button onclick="switchToNewDataset()" class="btn btn-primary">
        🔄 Switch to New Dataset (4 videos)
    </button>
    <button onclick="switchToOldDataset()" class="btn btn-secondary">
        📚 Switch to Original Dataset  
    </button>
</div>

<script>
async function switchToNewDataset() {
    showLoading('Switching to new dataset...');
    
    try {
        const response = await fetch('/api/switch_dataset', {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            showSuccess(result.message);
            // Refresh page để load dataset mới
            setTimeout(() => location.reload(), 1000);
        } else {
            showError(result.message);
        }
    } catch (error) {
        showError('Failed to switch dataset: ' + error.message);
    } finally {
        hideLoading();
    }
}
</script>
```

---

## 🚀 **Bước 7: Test Hệ Thống Mới**

### **🧪 Quick Test Script**

```python
# Tạo file: test_new_system.py
import requests
import json

def test_new_dataset_system():
    """Test hệ thống với dataset mới"""
    
    base_url = "http://localhost:8080"
    
    # Test 1: Health check
    print("🧪 Test 1: Health check")
    response = requests.get(f"{base_url}/")
    print(f"   Status: {response.status_code}")
    
    # Test 2: Switch to new dataset
    print("\n🧪 Test 2: Switch to new dataset")
    response = requests.post(f"{base_url}/api/switch_dataset")
    result = response.json()
    print(f"   Result: {result}")
    
    # Test 3: Search with new dataset
    print("\n🧪 Test 3: Search test")
    search_queries = [
        "presentation",
        "coding tutorial", 
        "nature scene",
        "interview",
        "person talking"
    ]
    
    for query in search_queries:
        print(f"\n   🔍 Testing query: '{query}'")
        response = requests.post(f"{base_url}/api/search", json={
            "query": query,
            "top_k": 3,
            "model": "clip_vit_base"
        })
        
        if response.status_code == 200:
            results = response.json()
            print(f"      ✅ Found {len(results.get('results', []))} results")
            
            for i, result in enumerate(results.get('results', [])[:2]):
                video_name = result.get('video_name', 'Unknown')
                similarity = result.get('similarity', 0)
                timestamp = result.get('timestamp', '00:00')
                print(f"      {i+1}. {video_name} - {timestamp} ({similarity:.1%})")
        else:
            print(f"      ❌ Search failed: {response.status_code}")

if __name__ == "__main__":
    test_new_dataset_system()
```

### **🚀 Chạy Test**

```bash
# Start web interface
python web_interface.py

# Mở terminal mới và test
python test_new_system.py

# Test manual qua browser
curl -X POST "http://localhost:8080/api/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "presentation", "top_k": 5}'
```

---

## 📊 **Bước 8: Complete Reset Workflow**

### **🔄 All-in-One Reset Script**

```bash
# Tạo file: reset_and_rebuild.sh
#!/bin/bash

echo "🚀 Starting complete system reset with new videos..."

# Step 1: Backup old data
echo "📦 Backing up old data..."
mkdir -p backup/$(date +%Y%m%d_%H%M%S)
mv frames backup/$(date +%Y%m%d_%H%M%S)/frames_old 2>/dev/null || true
mv index backup/$(date +%Y%m%d_%H%M%S)/index_old 2>/dev/null || true
mv embeddings backup/$(date +%Y%m%d_%H%M%S)/embeddings_old 2>/dev/null || true

# Step 2: Create new directories
echo "📁 Creating new directories..."
mkdir -p frames_new index_new embeddings_new

# Step 3: Extract frames
echo "🎬 Extracting frames from new videos..."
python extract_new_videos.py

# Step 4: Build embeddings
echo "🧠 Building embeddings..."
python build_new_embeddings.py

# Step 5: Update symlinks to point to new data
echo "🔗 Updating symlinks..."
rm -f frames index embeddings
ln -s frames_new frames
ln -s index_new index  
ln -s embeddings_new embeddings

echo "✅ Complete reset finished!"
echo "📊 New dataset ready for use"
echo "🌐 Start web interface: python web_interface.py"
```

### **🎯 Kết Quả Cuối Cùng**

Sau khi hoàn thành tất cả bước:

```
📊 New Dataset Status:
├── 🎬 4 videos processed
├── 📁 ~4,000 frames extracted (ước tính 1 frame/second)
├── 🧠 Embeddings built with CLIP ViT Base
├── 🔍 FAISS index ready for search
├── 🌐 Web interface updated
└── ✅ System ready for use!

🔍 Search Performance:
- Query processing: ~0.1-0.3 seconds
- 4 models available: CLIP Base/Large, Chinese CLIP, Sentence Transformers  
- GPU accelerated on RTX 3060
- Real-time model switching enabled
```

### **🎯 Next Steps**

1. **Test thoroughly**: Thử các query khác nhau
2. **Model comparison**: So sánh 4 models với dataset mới
3. **Performance tuning**: Tối ưu batch size, GPU memory
4. **Add more videos**: Expand dataset theo thời gian

Bạn giờ có complete workflow để bắt đầu lại từ con số 0 với 4 video mới!
