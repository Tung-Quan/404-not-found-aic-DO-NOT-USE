# parsers/json_parser.py
import json, os

def load_object_labels(json_path: str):
    """Đọc object detection từ file JSON"""
    if not os.path.exists(json_path):
        return []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("detection_class_entities", [])
    except Exception as e:
        print(f"[WARN] Failed to load {json_path}: {e}")
        return []