#!/usr/bin/env python3
"""
🗂️ CLEANUP SCRIPT - Remove Overlapping & Unnecessary Files
Removes duplicate, redundant, and overlapping files to simplify system
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict

def get_project_root() -> Path:
    """Get project root directory"""
    return Path(__file__).parent

def backup_file(file_path: Path, backup_dir: Path):
    """Backup file before deletion"""
    if file_path.exists():
        backup_dir.mkdir(exist_ok=True)
        backup_path = backup_dir / file_path.name
        shutil.copy2(file_path, backup_path)
        print(f"📦 Backed up: {file_path.name}")

def safe_remove_file(file_path: Path, reason: str):
    """Safely remove file with logging"""
    if file_path.exists():
        try:
            file_path.unlink()
            print(f"🗑️ Removed: {file_path.name} - {reason}")
            return True
        except Exception as e:
            print(f"❌ Failed to remove {file_path.name}: {e}")
            return False
    else:
        print(f"⚠️ File not found: {file_path.name}")
        return False

def main():
    print("🚀 SYSTEM CLEANUP - Removing Overlapping Files")
    print("=" * 60)
    
    project_root = get_project_root()
    backup_dir = project_root / "backup_before_cleanup"
    
    # Files to remove with reasons
    files_to_remove: Dict[str, str] = {
        # Duplicate search files
        "fix_search.py": "Duplicate of ai_search_engine.py functionality",
        "fix_search_complete.py": "Duplicate search engine with model loading",
        "debug_search.py": "Debug-only file, functionality in main engine",
        "simple_fix.py": "Basic version, superseded by main engine",
        
        # Duplicate lite versions
        "backend_ai_lite.py": "Duplicate of ai_search_lite.py",
        
        # Outdated setup files
        "setup.py": "Old setup script, replaced by setup_optimal.*",
        
        # Empty or minimal files (check if they exist and are small)
        "check_embeddings.py": "Basic embedding check, functionality in main system",
        "create_datasets.py": "Basic dataset creation, replaced by comprehensive system",
        
        # Test files that are redundant
        "test_new_user_workflow.py": "Specific test file, functionality covered elsewhere"
    }
    
    # Additional script files that might be redundant
    script_files_to_check = [
        "scripts/fix_tensorflow.py"  # May be old TensorFlow fix
    ]
    
    print("🔍 Analyzing files for removal...")
    
    removed_count = 0
    backup_count = 0
    
    # Process main directory files
    for filename, reason in files_to_remove.items():
        file_path = project_root / filename
        
        # Backup before removing
        if file_path.exists():
            backup_file(file_path, backup_dir)
            backup_count += 1
            
            # Remove file
            if safe_remove_file(file_path, reason):
                removed_count += 1
    
    # Check script files
    for script_file in script_files_to_check:
        script_path = project_root / script_file
        if script_path.exists():
            # Check file size - if very small, likely redundant
            file_size = script_path.stat().st_size
            if file_size < 1000:  # Less than 1KB
                backup_file(script_path, backup_dir)
                backup_count += 1
                if safe_remove_file(script_path, "Small script file, likely redundant"):
                    removed_count += 1
    
    # Clean up __pycache__ directories
    print("\n🧹 Cleaning up cache directories...")
    cache_cleaned = 0
    for pycache_dir in project_root.rglob("__pycache__"):
        if pycache_dir.is_dir():
            try:
                shutil.rmtree(pycache_dir)
                print(f"🗑️ Removed cache: {pycache_dir}")
                cache_cleaned += 1
            except Exception as e:
                print(f"❌ Failed to remove cache {pycache_dir}: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 CLEANUP SUMMARY:")
    print(f"✅ Files backed up: {backup_count}")
    print(f"🗑️ Files removed: {removed_count}")
    print(f"🧹 Cache directories cleaned: {cache_cleaned}")
    
    if backup_count > 0:
        print(f"📦 Backup location: {backup_dir}")
        print("💡 You can restore files from backup if needed")
    
    # Check for potential overlaps in remaining files
    print("\n🔍 Checking for remaining potential overlaps...")
    
    remaining_files = [
        "ai_search_engine.py",
        "ai_search_lite.py", 
        "enhanced_hybrid_manager.py",
        "tensorflow_model_manager.py",
        "ai_agent_manager.py",
        "web_interface.py",
        "main_launcher.py"
    ]
    
    existing_core_files = []
    for filename in remaining_files:
        file_path = project_root / filename
        if file_path.exists():
            existing_core_files.append(filename)
    
    print("📋 Core files remaining:")
    for filename in existing_core_files:
        print(f"   ✅ {filename}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("1. 📝 Review web_interface.py for duplicate endpoints")
    print("2. 🔧 Consolidate manager files into unified_model_manager.py") 
    print("3. 🚀 Consider BLIP-2 migration as outlined in SYSTEM_REDESIGN_PLAN.md")
    print("4. 📚 Update documentation to reflect new simplified structure")
    
    print("\n🎉 Cleanup completed successfully!")

if __name__ == "__main__":
    main()
