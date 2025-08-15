#!/usr/bin/env python3
"""
Enhanced Video Search - Modern Simple Launcher
==============================================
Khởi chạy bản hiện đại không cần TensorFlow phức tạp
"""

import os
import sys
import subprocess
from pathlib import Path
import platform

class ModernSimpleLauncher:
    def __init__(self):
        self.project_root = Path(__file__).parent
        os.chdir(self.project_root)
        
    def print_banner(self):
        print("🚀" + "=" * 58 + "🚀")
        print("    Enhanced Video Search - Modern Simple Mode")
        print("🚀" + "=" * 58 + "🚀")
        print(f"📍 Platform: {platform.system()} {platform.release()}")
        print(f"🐍 Python: {sys.version.split()[0]}")
        
        # Check virtual environment
        venv_path = os.environ.get('VIRTUAL_ENV')
        if venv_path:
            print(f"📦 Environment: {Path(venv_path).name}")
        else:
            print("📦 Environment: System Python")
        print()
        
    def check_dependencies(self):
        """Kiểm tra dependencies cần thiết"""
        required = ['fastapi', 'uvicorn', 'pandas']
        missing = []
        
        for pkg in required:
            try:
                __import__(pkg)
                print(f"✅ {pkg}")
            except ImportError:
                print(f"❌ {pkg} - chưa cài đặt")
                missing.append(pkg)
        
        if missing:
            print(f"\n🔧 Cần cài đặt: {' '.join(missing)}")
            install = input("Cài đặt ngay? (y/n): ").lower()
            if install == 'y':
                cmd = [sys.executable, '-m', 'pip', 'install'] + missing
                subprocess.run(cmd)
                print("✅ Đã cài đặt xong!")
            return False
        return True
    
    def show_menu(self):
        """Hiển thị menu chính"""
        print("🎯 CHỌN CHỨC NĂNG:")
        print("1. 🔍 Tìm kiếm video (Simple Mode)")
        print("2. 📊 Xem thông tin hệ thống") 
        print("3. 🗂️  Quản lý index")
        print("4. 🌐 Khởi chạy API Server")
        print("5. 🛠️  Cài đặt dependencies")
        print("6. 🚪 Thoát")
        print()
        
    def start_api_server(self):
        """Khởi chạy API server"""
        print("🌐 Đang khởi chạy API Server...")
        print("📡 Truy cập tại: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print()
        
        # Sử dụng app_unified.py nếu có, không thì tạo server đơn giản
        if Path('src/api/app_unified.py').exists():
            subprocess.run([sys.executable, 'src/api/app_unified.py'])
        else:
            print("❌ Không tìm thấy app_unified.py")
            
    def install_dependencies(self):
        """Cài đặt tất cả dependencies"""
        print("🔧 Đang cài đặt dependencies...")
        packages = [
            'fastapi', 'uvicorn[standard]', 'pandas', 
            'numpy', 'pillow', 'python-multipart'
        ]
        
        cmd = [sys.executable, '-m', 'pip', 'install'] + packages
        subprocess.run(cmd)
        print("✅ Đã cài đặt xong tất cả dependencies!")
        
    def system_info(self):
        """Hiển thị thông tin hệ thống"""
        print("📊 THÔNG TIN HỆ THỐNG:")
        print(f"🔹 OS: {platform.system()} {platform.release()}")
        print(f"🔹 Architecture: {platform.machine()}")
        print(f"🔹 Python: {sys.version}")
        print(f"🔹 Working Directory: {os.getcwd()}")
        
        # Kiểm tra project structure
        dirs = ['src', 'index', 'videos', 'frames']
        print(f"\n📁 PROJECT STRUCTURE:")
        for d in dirs:
            if Path(d).exists():
                print(f"✅ {d}/")
            else:
                print(f"❌ {d}/ - chưa tạo")
        print()
        
    def simple_search(self):
        """Tìm kiếm đơn giản"""
        print("🔍 TÌM KIẾM VIDEO - SIMPLE MODE")
        print("=" * 40)
        
        query = input("Nhập từ khóa tìm kiếm: ").strip()
        if not query:
            print("❌ Vui lòng nhập từ khóa!")
            return
            
        # Kiểm tra index file
        index_file = Path('index/meta.parquet')
        if not index_file.exists():
            print("❌ Chưa có index file. Vui lòng tạo index trước!")
            return
            
        try:
            import pandas as pd
            df = pd.read_parquet(index_file)
            print(f"📋 Tìm thấy {len(df)} frame trong index")
            
            # Simple keyword search
            if 'description' in df.columns:
                results = df[df['description'].str.contains(query, case=False, na=False)]
            else:
                print("⚠️  Không có mô tả, hiển thị kết quả ngẫu nhiên...")
                results = df.sample(min(5, len(df)))
                
            if len(results) > 0:
                print(f"🎯 Tìm thấy {len(results)} kết quả:")
                for i, (_, row) in enumerate(results.head(5).iterrows()):
                    print(f"{i+1}. {row.get('video_file', 'unknown')} - Frame {row.get('frame_number', 0)}")
            else:
                print("❌ Không tìm thấy kết quả nào!")
                
        except Exception as e:
            print(f"❌ Lỗi tìm kiếm: {e}")
        
        input("\nNhấn Enter để tiếp tục...")
        
    def run(self):
        """Chạy launcher"""
        self.print_banner()
        
        if not self.check_dependencies():
            return
            
        while True:
            self.show_menu()
            choice = input("Chọn (1-6): ").strip()
            
            if choice == '1':
                self.simple_search()
            elif choice == '2':
                self.system_info()
                input("\nNhấn Enter để tiếp tục...")
            elif choice == '3':
                print("🗂️  Index management - Tính năng đang phát triển")
                input("\nNhấn Enter để tiếp tục...")
            elif choice == '4':
                self.start_api_server()
                break
            elif choice == '5':
                self.install_dependencies()
                input("\nNhấn Enter để tiếp tục...")
            elif choice == '6':
                print("👋 Tạm biệt!")
                break
            else:
                print("❌ Lựa chọn không hợp lệ!")
            
            print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    launcher = ModernSimpleLauncher()
    launcher.run()
