#!/usr/bin/env python3
"""
Phân tích files trùng lặp và không cần thiết trong hệ thống AI Video Search.
Tạo backup plan và đề xuất files cần xóa an toàn.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Set
import ast

def get_file_hash(filepath: str, sample_size: int = 8192) -> str:
    """Tính hash của file để detect duplicates"""
    try:
        with open(filepath, 'rb') as f:
            content = f.read(sample_size)
            return hashlib.md5(content).hexdigest()
    except:
        return ""

def analyze_python_imports(filepath: str) -> Set[str]:
    """Phân tích imports trong Python file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        tree = ast.parse(content)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
        return imports
    except:
        return set()

def analyze_functions(filepath: str) -> Set[str]:
    """Phân tích functions/classes trong Python file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        tree = ast.parse(content)
        functions = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.add(node.name)
            elif isinstance(node, ast.ClassDef):
                functions.add(node.name)
        return functions
    except:
        return set()

def main():
    root = Path(".")
    
    # Files cần phân tích
    analysis_targets = [
        # Core engine files
        "ai_search_engine.py",
        "ai_search_lite.py", 
        "backend_ai_lite.py",
        "backend_ai.py",
        
        # Fix files (likely duplicates)
        "fix_search.py",
        "fix_search_complete.py", 
        "simple_fix.py",
        "debug_search.py",
        
        # Manager files
        "enhanced_hybrid_manager.py",
        "ai_agent_manager.py",
        "tensorflow_model_manager.py",
        
        # Web interface
        "web_interface.py",
        "main_launcher.py",
        
        # Scripts
        "scripts/encode_clip.py",
        "scripts/encode_siglip.py", 
        "scripts/build_faiss.py",
        "scripts/text_embed.py"
    ]
    
    report = {
        "duplicate_groups": [],
        "obsolete_files": [],
        "safe_to_remove": [],
        "keep_files": [],
        "file_analysis": {}
    }
    
    # Phân tích từng file
    for target in analysis_targets:
        filepath = root / target
        if not filepath.exists():
            continue
            
        file_info = {
            "path": str(filepath),
            "size": filepath.stat().st_size,
            "hash": get_file_hash(str(filepath)),
            "imports": list(analyze_python_imports(str(filepath))),
            "functions": list(analyze_functions(str(filepath)))
        }
        report["file_analysis"][target] = file_info
    
    # Detect potential duplicates based on function overlap
    def calculate_similarity(file1_info, file2_info):
        funcs1 = set(file1_info["functions"])
        funcs2 = set(file2_info["functions"])
        if not funcs1 or not funcs2:
            return 0.0
        overlap = len(funcs1.intersection(funcs2))
        total = len(funcs1.union(funcs2))
        return overlap / total if total > 0 else 0.0
    
    # Tìm files có chức năng tương tự (>50% functions overlap)
    duplicates = []
    files = list(report["file_analysis"].items())
    
    for i, (path1, info1) in enumerate(files):
        for path2, info2 in files[i+1:]:
            similarity = calculate_similarity(info1, info2)
            if similarity > 0.5:
                duplicates.append({
                    "file1": path1,
                    "file2": path2, 
                    "similarity": similarity,
                    "recommendation": "investigate_merge"
                })
    
    report["duplicate_groups"] = duplicates
    
    # Đánh dấu files cần xóa dựa trên pattern analysis
    
    # 1. Fix files - có thể obsolete
    fix_files = ["fix_search.py", "fix_search_complete.py", "simple_fix.py", "debug_search.py"]
    report["safe_to_remove"].extend([f for f in fix_files if (root / f).exists()])
    
    # 2. Lite versions if full version exists
    if (root / "ai_search_engine.py").exists() and (root / "ai_search_lite.py").exists():
        report["safe_to_remove"].append("ai_search_lite.py")
    
    if (root / "backend_ai.py").exists() and (root / "backend_ai_lite.py").exists():
        report["safe_to_remove"].append("backend_ai_lite.py")
    
    # 3. Core files cần giữ
    keep_files = [
        "web_interface.py",  # Main web server
        "enhanced_hybrid_manager.py",  # TensorFlow rerank (user wants to keep)
        "tensorflow_model_manager.py",  # TensorFlow models
        "ai_search_engine.py",  # Main search logic
        "scripts/build_faiss.py",  # Core indexing
    ]
    report["keep_files"] = keep_files
    
    # Write report
    with open("_duplicate_analysis.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("📊 DUPLICATE ANALYSIS COMPLETE")
    print(f"Found {len(duplicates)} potential duplicate pairs")
    print(f"Safe to remove: {len(report['safe_to_remove'])} files")
    print("\n🗑️  RECOMMENDED DELETIONS:")
    for f in report["safe_to_remove"]:
        print(f"  - {f}")
    
    print("\n✅ CORE FILES TO KEEP:")
    for f in report["keep_files"]:
        print(f"  - {f}")
    
    print(f"\n📄 Full report: _duplicate_analysis.json")

if __name__ == "__main__":
    main()
