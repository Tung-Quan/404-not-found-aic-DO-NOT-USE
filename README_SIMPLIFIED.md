# 🔍 Simplified AI Search System

## 📖 Tổng quan

Hệ thống tìm kiếm AI đơn giản hóa sử dụng **1 mô hình chính (CLIP) + OCR** để tìm kiếm multimodal trên video frames. Đây là phiên bản tối ưu hóa từ hệ thống phức tạp trước đó, tập trung vào hiệu suất và đơn giản.

### 🎯 Tính năng chính
- **Single Model Architecture**: Chỉ sử dụng CLIP cho encoding
- **OCR Integration**: VietOCR để nhận dạng text trong ảnh
- **Hybrid Search**: Kết hợp tìm kiếm visual và text
- **Web Interface**: Giao diện web đơn giản và trực quan
- **Fast Setup**: Khởi động nhanh với script tự động

## 🚀 Quick Start

### 1. Cài đặt và khởi động
```bash
# Clone hoặc download project
cd Project

# Chạy script setup tự động
python quick_start.py
```

### 2. Truy cập Web Interface
Mở browser và truy cập: **http://localhost:8000**

### 3. Setup hệ thống
1. **Initialize Engine**: Chọn model và device
2. **Load/Build Index**: Load index có sẵn hoặc build từ frames
3. **Search**: Bắt đầu tìm kiếm!

## 🏗️ Kiến trúc hệ thống

### Core Components

#### 1. `simplified_search_engine.py`
**SimplifiedSearchEngine Class**
- **Model Loading**: CLIP (OpenAI hoặc HuggingFace)
- **OCR Integration**: VietOCR cho text extraction
- **Multimodal Processing**: Kết hợp visual + text embeddings
- **FAISS Indexing**: Vector search với cosine similarity

```python
# Usage example
engine = SimplifiedSearchEngine(model_name="ViT-B/32", device="auto")
engine.build_index("frames/", "index/")
results = engine.search("tìm người đàn ông", top_k=20, search_mode="hybrid")
```

#### 2. `simplified_web_interface.py`
**FastAPI Web Server**
- **REST API**: Endpoints cho search, setup, index management
- **File Upload**: Xử lý ảnh upload
- **Real-time Status**: Monitoring hệ thống
- **Multi-dataset Support**: Quản lý nhiều dataset

#### 3. `templates/simplified_index.html`
**Modern Web UI**
- **Responsive Design**: Tương thích mobile
- **Real-time Feedback**: Status updates và loading states
- **Tabs Interface**: Search, Setup, Datasets
- **Visual Results**: Hiển thị kết quả với thumbnails

## 🔧 Cấu hình

### Model Options
- **ViT-B/32**: Nhanh, nhẹ (khuyến nghị cho CPU)
- **ViT-L/14**: Chậm hơn nhưng chính xác hơn (khuyến nghị cho GPU)

### Search Modes
- **Hybrid**: Kết hợp visual + text (mặc định)
- **Visual Only**: Chỉ dựa vào nội dung visual
- **Text Only**: Chỉ tìm trên text được OCR

### Device Selection
- **Auto**: Tự động chọn GPU nếu có, ngược lại CPU
- **CUDA**: Force sử dụng GPU
- **CPU**: Force sử dụng CPU

## 📁 Cấu trúc thư mục

```
Project/
├── simplified_search_engine.py     # Core search engine
├── simplified_web_interface.py     # Web server
├── quick_start.py                  # Setup script
├── requirements_simplified.txt     # Dependencies
├── templates/
│   └── simplified_index.html       # Web UI
├── frames/                         # Video frames
├── datasets/                       # Additional datasets
├── index/                          # Search indexes
└── static/                         # Static web assets
```

## 🔍 API Endpoints

### Status & Setup
- `GET /api/status` - Kiểm tra trạng thái hệ thống
- `POST /api/initialize` - Khởi tạo engine
- `POST /api/build_index` - Build search index
- `POST /api/load_index` - Load index có sẵn

### Search
- `POST /api/search` - Tìm kiếm với query
- `GET /api/datasets` - List datasets có sẵn
- `GET /api/frame/{path}` - Lấy frame image

### File Processing
- `POST /api/process_image` - Xử lý ảnh upload với OCR

## 🎨 Search Query Examples

### Vietnamese Queries
```
"người đàn ông mặc áo xanh"
"xe hơi màu đỏ"
"văn bản tiếng Việt"
"biển báo giao thông"
```

### English Queries
```
"man wearing blue shirt"
"red car"
"text document"
"traffic sign"
```

### Mixed Queries
```
"person holding iPhone"
"computer screen with code"
"restaurant menu"
```

## ⚡ Performance Tips

### Tối ưu hóa GPU
```bash
# Cài đặt PyTorch với CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Tối ưu hóa Memory
- **Batch Processing**: Process nhiều ảnh cùng lúc khi build index
- **Index Caching**: Save/load index để tránh rebuild
- **Model Caching**: Models được cache sau lần load đầu

### Tối ưu hóa Search
- **Pre-built Index**: Build index trước cho datasets lớn
- **FAISS Optimization**: Sử dụng IndexIVFFlat cho datasets rất lớn
- **Top-K Limitation**: Giới hạn kết quả để tăng tốc

## 🔧 Troubleshooting

### Common Issues

#### 1. OCR không hoạt động
```bash
# Cài đặt VietOCR
pip install vietocr
# Hoặc dùng EasyOCR thay thế
pip install easyocr
```

#### 2. GPU không được nhận dạng
```bash
# Kiểm tra CUDA
python -c "import torch; print(torch.cuda.is_available())"
# Cài đặt CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Memory Error khi build index
- Giảm batch size trong quá trình xử lý
- Sử dụng CPU thay vì GPU cho datasets lớn
- Chia nhỏ dataset thành các phần

#### 4. Web interface không load
```bash
# Kiểm tra port 8000
netstat -an | findstr :8000
# Thay đổi port nếu cần
uvicorn simplified_web_interface:app --port 8001
```

## 📊 System Requirements

### Minimum
- **Python**: 3.8+
- **RAM**: 4GB
- **Storage**: 2GB for models + dataset size
- **CPU**: 2 cores

### Recommended
- **Python**: 3.9+
- **RAM**: 8GB+
- **GPU**: NVIDIA GPU with 4GB+ VRAM
- **Storage**: SSD preferred
- **CPU**: 4+ cores

## 🔄 Migration từ hệ thống cũ

Nếu bạn đang sử dụng hệ thống phức tạp trước đó:

1. **Backup dữ liệu**: Sao lưu index và embeddings hiện tại
2. **Export frames**: Đảm bảo frames được organize trong thư mục `frames/`
3. **Run new system**: Khởi động hệ thống mới với `python quick_start.py`
4. **Rebuild index**: Build lại index với engine mới
5. **Test search**: Kiểm tra kết quả tìm kiếm

## 🆘 Support

### Logs và Debug
- **System logs**: Kiểm tra console output
- **FastAPI logs**: Automatic logging cho API calls
- **Error handling**: Comprehensive error messages

### Community
- **Issues**: Report bugs hoặc feature requests
- **Documentation**: Chi tiết trong source code comments
- **Examples**: Sample queries và use cases

## 📈 Future Enhancements

### Planned Features
- **Multiple OCR engines**: Support cho EasyOCR, PaddleOCR
- **Advanced search**: Semantic search improvements
- **Batch upload**: Process multiple images
- **Export results**: CSV/JSON export
- **Performance monitoring**: Real-time metrics

### Performance Improvements
- **Model quantization**: Faster inference
- **Index optimization**: Better FAISS configurations
- **Caching strategies**: Enhanced response times
- **Distributed search**: Multi-node support

---

## 🎉 Kết luận

Hệ thống Simplified AI Search đã được tối ưu hóa để:
- ✅ **Đơn giản**: 1 mô hình chính thay vì nhiều models phức tạp
- ✅ **Nhanh**: Optimized performance với CLIP + OCR
- ✅ **Ổn định**: Fewer dependencies, better error handling
- ✅ **User-friendly**: Modern web interface
- ✅ **Scalable**: Có thể mở rộng cho production

**Happy Searching! 🔍✨**
