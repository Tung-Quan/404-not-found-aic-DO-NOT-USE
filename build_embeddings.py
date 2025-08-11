#!/usr/bin/env python3
"""
Build AI embeddings and search index
Wrapper script that calls the embedding generation and index building scripts
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_path: str, description: str) -> bool:
    """Run a Python script and return success status"""
    
    print(f"🔧 {description}...")
    print(f"   Running: python {script_path}")
    print("-" * 50)
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_path], check=True)
        print(f"✅ {description} completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"   Exit code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Script not found: {script_path}")
        return False

def main():
    """Main function to build embeddings and index"""
    
    print("🧠 AI Embeddings & Index Builder")
    print("=" * 50)
    
    # Check if required directories exist
    scripts_dir = Path('scripts')
    if not scripts_dir.exists():
        print("❌ Error: 'scripts' directory not found!")
        return False
    
    # Check if frames exist
    frames_dir = Path('frames')
    if not frames_dir.exists() or not any(frames_dir.iterdir()):
        print("❌ Error: No frames found!")
        print("   Please run 'python extract_frames.py' first")
        return False
    
    # Check if metadata exists
    meta_file = Path('index/meta.parquet')
    if not meta_file.exists():
        print("❌ Error: Metadata file not found!")
        print("   Please run 'python build_meta.py' first")
        return False
    
    print("📊 Prerequisites check passed!")
    print()
    
    # Step 1: Generate AI embeddings
    success1 = run_script('scripts/encode_siglip.py', 'Generating AI embeddings')
    print()
    
    if not success1:
        print("❌ Cannot proceed without embeddings!")
        return False
    
    # Step 2: Build search index
    success2 = run_script('scripts/build_faiss.py', 'Building search index')
    print()
    
    # Summary
    print("=" * 50)
    if success1 and success2:
        print("🎉 All steps completed successfully!")
        print()
        print("📁 Generated files:")
        print("   ├── index/embeddings/frames.f16.mmap")
        print("   └── index/faiss/ip_flat.index")
        print()
        print("✅ Ready to start the server!")
        print("   Next: Run 'start_server_simple.bat'")
        return True
    else:
        print("❌ Some steps failed!")
        print("   Please check the errors above and try again")
        return False

if __name__ == "__main__":
    success = main()
    
    if not success:
        sys.exit(1)
