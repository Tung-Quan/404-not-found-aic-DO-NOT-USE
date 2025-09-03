"""
🔍 Monitor Embedding Progress - Real-time tracking
"""

import time
import os
from pathlib import Path
import json

def monitor_progress():
    """Monitor embedding progress in real-time"""
    print("🔍 MONITORING EMBEDDING PROGRESS")
    print("=" * 50)
    print("Ctrl+C to stop monitoring")
    
    start_time = time.time()
    
    while True:
        try:
            current_time = time.strftime("%H:%M:%S")
            elapsed = time.time() - start_time
            
            print(f"\n⏰ {current_time} (Elapsed: {elapsed/60:.1f}m)")
            
            # Check index files
            index_dir = Path("index")
            
            if index_dir.exists():
                faiss_file = index_dir / "visual_index.faiss"
                metadata_file = index_dir / "metadata.pkl"
                
                if faiss_file.exists():
                    size_mb = faiss_file.stat().st_size / (1024 * 1024)
                    print(f"📊 FAISS Index: {size_mb:.2f} MB")
                    
                if metadata_file.exists():
                    try:
                        import pickle
                        with open(metadata_file, 'rb') as f:
                            metadata = pickle.load(f)
                        
                        processed = len(metadata)
                        total = 170540
                        progress = (processed / total) * 100
                        
                        print(f"📷 Processed: {processed:,}/{total:,} ({progress:.2f}%)")
                        
                        if processed > 0 and elapsed > 30:  # After 30 seconds
                            speed = processed / elapsed
                            remaining = (total - processed) / speed if speed > 0 else 0
                            
                            print(f"⚡ Speed: {speed:.1f} images/sec")
                            print(f"⏱️ ETA: {remaining/3600:.1f} hours")
                            
                            # Progress bar
                            bar_length = 40
                            filled = int(bar_length * progress / 100)
                            bar = "█" * filled + "░" * (bar_length - filled)
                            print(f"📊 [{bar}] {progress:.1f}%")
                            
                    except Exception as e:
                        print(f"📋 Metadata: Error reading - {e}")
                else:
                    print("📋 Metadata: Not found yet")
            else:
                print("📂 Index directory: Not found")
            
            time.sleep(15)  # Check every 15 seconds
            
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")
            break
        except Exception as e:
            print(f"❌ Monitor error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_progress()
