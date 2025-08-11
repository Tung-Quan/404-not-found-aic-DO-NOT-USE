# 🎬 Quick Start Guide - AI Video Search

## 🚀 Two Server Options Available

### 1. ⚡ Simple Server (RECOMMENDED)
- **File**: `start_server_simple.bat`
- **Port**: 8001
- **Memory**: Low usage (~1GB)
- **Features**: Individual frame search, basic video search
- **Best for**: Production use, systems with limited memory

### 2. 🔥 Advanced Server  
- **File**: `start_server_advanced.bat`
- **Port**: 8000
- **Memory**: High usage (~4GB+)
- **Features**: Full video search, top 5 frames per video, TF-IDF
- **Best for**: Development, powerful systems

## 📋 File Structure

```
📁 Project/
├── 🚀 start_server_simple.bat    # Memory optimized server
├── 🔥 start_server_advanced.bat  # Full featured server  
├── 🎛️ start_server.bat           # Server selection menu
├── 🛠️ setup_and_run.bat          # Complete setup from scratch
├── 📂 api/
│   ├── app_simple.py             # Simple server code
│   └── app.py                    # Advanced server code
└── 📚 README.md                  # Full documentation
```

## ⚡ Quick Commands

```bash
# Start simple server (recommended)
start_server_simple.bat

# Start advanced server 
start_server_advanced.bat

# Choose server interactively
start_server.bat

# Complete setup from videos
setup_and_run.bat
```

## 🌐 Access URLs

### Simple Server (Port 8001)
- API Docs: http://localhost:8001/docs
- Health: http://localhost:8001/health  
- Frame Search: http://localhost:8001/search_frames?q=your_query

### Advanced Server (Port 8000)
- API Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health
- Video Search: http://localhost:8000/search?q=your_query

## 🆘 Troubleshooting

- **Memory Error**: Use simple server instead
- **Port Conflict**: Change port in .bat files
- **No Videos Found**: Place .mp4 files in `videos/` folder
- **Missing Index**: Run `setup_and_run.bat` first
