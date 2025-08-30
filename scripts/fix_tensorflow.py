#!/usr/bin/env python3
"""
Fix TensorFlow/Keras compatibility issues
"""

import subprocess
import sys
import os

def fix_tensorflow_issues():
    """Fix common TensorFlow compatibility issues"""
    print("Fixing TensorFlow compatibility issues...")
    print("=" * 50)
    
    fixes = [
        # Downgrade protobuf to compatible version
        ("protobuf", "protobuf==4.25.4"),
        
        # Ensure compatible TensorFlow version
        ("tensorflow", "tensorflow==2.15.0"),
        
        # Ensure compatible tf-keras
        ("tf-keras", "tf-keras==2.15.0"),
        
        # Fix TensorFlow Hub
        ("tensorflow-hub", "tensorflow-hub==0.15.0"),
    ]
    
    for package_name, package_spec in fixes:
        print(f"\n[INFO] Fixing {package_name}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "--upgrade", "--force-reinstall", package_spec
            ])
            print(f"[OK] {package_name} fixed successfully")
        except subprocess.CalledProcessError as e:
            print(f"[WARNING] Failed to fix {package_name}: {e}")
    
    print("\n" + "=" * 50)
    print("TensorFlow compatibility fixes completed!")
    print("Please restart the application.")

if __name__ == "__main__":
    fix_tensorflow_issues()
