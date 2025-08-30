# parsers/metadata_linker.py
import os
from parsers.json_parser import load_object_labels
from parsers.csv_parser import load_keyframe_metadata

def link_metadata(video_name, frame_idx, ocr_text):
    # JSON path
    json_path = f"data/objects/{video_name}/{frame_idx:03d}.json"
    objects = load_object_labels(json_path)

    # CSV path
    csv_path = f"data/map-keyframes/{video_name}.csv"
    keyframes = load_keyframe_metadata(csv_path)
    time_info = keyframes.get(frame_idx, {})

    return {
        "video": video_name,
        "frame_idx": frame_idx,
        "ocr_text": ocr_text,
        "objects": objects,
        "timestamp": time_info.get("pts_time"),
        "fps": time_info.get("fps")
    }