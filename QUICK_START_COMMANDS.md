# 🚀 Quick Start Commands - Complete Guide

## 📋 **TL;DR - One Command Setup**

```bash
# Workflow hoàn chỉnh từ đầu đến cuối
python quick_start.py

# Hoặc với batch file (Windows)
quick_start.bat

# Hoặc với PowerShell (Windows)
.\quick_start.ps1
```

---

## 🔧 **Manual Commands - Step by Step**

### **1. Setup Environment**
```bash
# Cài đặt dependencies
pip install -r requirements_simplified.txt

# Kiểm tra cài đặt
python -c "import torch, transformers, PIL, cv2, numpy, fastapi, uvicorn; print('✅ All imports OK')"
```

### **2. Check GPU (Optional)**
```bash
# Kiểm tra GPU có available không
python -c "import torch; print('GPU Available:', torch.cuda.is_available())"
```

### **3. Prepare Image Frames**
```bash
# Đảm bảo có images trong một trong các thư mục này:
# frames/
# datasets/
# data/
# images/

# Kiểm tra số lượng images
python -c "
import os
from pathlib import Path
for dir_name in ['frames', 'datasets', 'data', 'images']:
    if os.path.exists(dir_name):
        count = len(list(Path(dir_name).rglob('*.jpg'))) + len(list(Path(dir_name).rglob('*.png')))
        print(f'{dir_name}: {count} images')
"
```

### **4. Build Embeddings Index**
```bash
# Build index từ frames directory
python -c "
from simplified_search_engine import SimplifiedSearchEngine
engine = SimplifiedSearchEngine()
engine.build_index('frames', 'index')  # Thay 'frames' bằng directory của bạn
print('✅ Index built successfully')
"

# Hoặc với command line
python quick_start.py --build-index frames
```

### **5. Start Web Interface**
```bash
# Khởi động web server
python simplified_web_interface.py

# Hoặc
python quick_start.py --start
```

### **6. Open Browser**
```
http://localhost:8000
```

---

## ⚡ **Quick Commands**

### **System Status**
```bash
# Check trạng thái hệ thống
python quick_start.py --status

# Với batch file
quick_start.bat --status

# Với PowerShell
.\quick_start.ps1 -Status
```

### **Setup Only**
```bash
# Chỉ setup dependencies, không build index
python quick_start.py --setup-only

# Với batch file
quick_start.bat --setup
```

### **Build Index Only**
```bash
# Build index từ directory cụ thể
python quick_start.py --build-index frames
python quick_start.py --build-index datasets/nature_collection

# Với batch file
quick_start.bat --build frames
```

### **Start Web Only**
```bash
# Chỉ khởi động web interface (cần có index sẵn)
python quick_start.py --start

# Với batch file
quick_start.bat --start
```

### **Show Manual Commands**
```bash
# Hiển thị commands chi tiết
python quick_start.py --manual

# Với batch file
quick_start.bat --manual
```

---

## 🐛 **Troubleshooting Commands**

### **Check Dependencies**
```bash
# Test imports
python -c "
try:
    import torch
    print('✅ PyTorch:', torch.__version__)
except ImportError:
    print('❌ PyTorch not installed')

try:
    import transformers
    print('✅ Transformers:', transformers.__version__)
except ImportError:
    print('❌ Transformers not installed')

try:
    import vietocr
    print('✅ VietOCR available')
except ImportError:
    print('❌ VietOCR not installed')
"
```

### **Check GPU**
```bash
# Detailed GPU info
python -c "
import torch
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Count:', torch.cuda.device_count())
    print('Current GPU:', torch.cuda.current_device())
    print('GPU Name:', torch.cuda.get_device_name())
    print('GPU Memory:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
"
```

### **Check Index Status**
```bash
# Check nếu index đã được build
python -c "
import os
import pickle
from pathlib import Path

index_dir = Path('index')
if index_dir.exists():
    faiss_file = index_dir / 'visual_index.faiss'
    metadata_file = index_dir / 'metadata.pkl'
    
    print('Index directory exists:', index_dir.exists())
    print('FAISS file exists:', faiss_file.exists())
    print('Metadata file exists:', metadata_file.exists())
    
    if metadata_file.exists():
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        print('Index contains:', len(metadata), 'items')
    else:
        print('No index found')
else:
    print('Index directory does not exist')
"
```

### **Clean Start**
```bash
# Xóa index cũ và build lại
rmdir /s index  # Windows
# rm -rf index  # Linux/Mac

python quick_start.py --build-index frames
```

---

## 🔄 **Complete Workflow Examples**

### **First Time Setup**
```bash
# 1. Clone/download project
cd Project

# 2. Complete setup
python quick_start.py
# Hoặc: quick_start.bat
# Hoặc: .\quick_start.ps1

# 3. Truy cập http://localhost:8000
```

### **Daily Usage**
```bash
# Check status
python quick_start.py --status

# Start if ready
python quick_start.py --start

# Rebuild index if new images added
python quick_start.py --build-index frames
```

### **Development Mode**
```bash
# Setup only
python quick_start.py --setup-only

# Build index manually
python quick_start.py --build-index frames

# Start with auto-reload
python simplified_web_interface.py
```

---

## 📊 **Performance Tips**

### **GPU Optimization**
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Force GPU usage
python -c "
from simplified_search_engine import SimplifiedSearchEngine
engine = SimplifiedSearchEngine(device='cuda')
"
```

### **Large Dataset Handling**
```bash
# Process in batches for large datasets
python -c "
import os
from pathlib import Path

# Count images first
frames_dir = Path('frames')
image_files = list(frames_dir.rglob('*.jpg')) + list(frames_dir.rglob('*.png'))
print(f'Total images: {len(image_files)}')

# Estimate processing time
print(f'Estimated time: {len(image_files) / 100 * 2} minutes')
"
```

---

## 💡 **Pro Tips**

1. **Always check status first**: `python quick_start.py --status`
2. **Use GPU if available**: Significantly faster processing
3. **Organize images**: Put frames in `frames/` directory for best results
4. **Monitor logs**: Watch console output for errors
5. **Backup index**: Save `index/` directory after building
6. **Use virtual environment**: Avoid dependency conflicts

---

**🎉 Happy Searching! 🔍✨**
