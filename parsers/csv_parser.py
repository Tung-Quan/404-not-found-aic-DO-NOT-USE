# parsers/csv_parser.py
import csv, os

def load_keyframe_metadata(csv_path):
    csv_path = f"map-keyframes/{video_name}.csv"  # đã bỏ 'data/'

    metadata = {}
    if not os.path.exists(csv_path):
        return metadata
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = int(row["frame_idx"])
            metadata[frame_idx] = {
                "pts_time": float(row["pts_time"]),
                "fps": float(row["fps"]),
                "n": int(row["n"])
            }
    return metadata