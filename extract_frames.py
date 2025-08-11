#!/usr/bin/env python3
"""
Extract frames from videos using FFmpeg
Extracts 1 frame per second from each video file
"""

import os
import subprocess
import sys
from pathlib import Path

def extract_frames_from_video(video_path: Path, output_dir: Path, fps: int = 1):
    """Extract frames from a single video file"""
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # FFmpeg command to extract frames
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output files
        '-i', str(video_path),  # Input video
        '-vf', f'fps={fps}',  # Video filter: extract 1 frame per second
        str(output_dir / 'frame_%06d.jpg')  # Output pattern
    ]
    
    print(f"Extracting frames from: {video_path.name}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Run FFmpeg command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Count extracted frames
        frame_count = len(list(output_dir.glob('frame_*.jpg')))
        print(f"✅ Successfully extracted {frame_count} frames")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error extracting frames from {video_path.name}:")
        print(f"   {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ Error: FFmpeg not found!")
        print("   Please install FFmpeg and add it to your PATH")
        print("   Windows: choco install ffmpeg")
        print("   Or download from: https://ffmpeg.org/download.html")
        return False

def main():
    """Main function to extract frames from all videos"""
    
    # Define directories
    videos_dir = Path('videos')
    frames_dir = Path('frames')
    
    # Check if videos directory exists
    if not videos_dir.exists():
        print("❌ Error: 'videos' directory not found!")
        print("   Please create 'videos' folder and add your video files")
        return False
    
    # Get all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(videos_dir.glob(f'*{ext}'))
        video_files.extend(videos_dir.glob(f'*{ext.upper()}'))
    
    if not video_files:
        print("❌ Error: No video files found in 'videos' directory!")
        print(f"   Supported formats: {', '.join(video_extensions)}")
        return False
    
    print(f"📹 Found {len(video_files)} video file(s)")
    print("=" * 50)
    
    # Extract frames from each video
    success_count = 0
    for video_file in video_files:
        video_name = video_file.stem  # Filename without extension
        output_dir = frames_dir / video_name
        
        if extract_frames_from_video(video_file, output_dir):
            success_count += 1
        
        print("-" * 30)
    
    # Summary
    print("=" * 50)
    print(f"📊 Extraction Summary:")
    print(f"   Total videos: {len(video_files)}")
    print(f"   Successful: {success_count}")
    print(f"   Failed: {len(video_files) - success_count}")
    
    if success_count == len(video_files):
        print("🎉 All videos processed successfully!")
        return True
    else:
        print("⚠️  Some videos failed to process")
        return False

if __name__ == "__main__":
    print("🎬 Video Frame Extraction Tool")
    print("=" * 50)
    
    success = main()
    
    if success:
        print("\n✅ Frame extraction completed!")
        print("   Next step: Run 'python build_meta.py'")
    else:
        print("\n❌ Frame extraction failed!")
        print("   Please check the errors above and try again")
        sys.exit(1)
