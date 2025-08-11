import os, glob, pandas as pd
from pathlib import Path
os.makedirs('index', exist_ok=True)
rows = []
found_any = False
for vid_dir in sorted(glob.glob('frames/*')):
    print(f"Processing {vid_dir}...")
    if not os.path.isdir(vid_dir):
        print(f"Skipping {vid_dir}: not a directory.")
        continue
    vid = os.path.basename(vid_dir)
    all_files = os.listdir(vid_dir)
    jpg_files = sorted([str(p) for p in Path(vid_dir).glob('*.jpg')])
    print(f"Checking {vid_dir}: found {len(jpg_files)} .jpg files.")
    if not jpg_files:
        print(f"Warning: No .jpg files found in {vid_dir}")
    else:
        found_any = True
    for p in jpg_files:
        ts_name = os.path.splitext(os.path.basename(p))[0]
        try:
            ts = int(ts_name.split('_')[-1])
        except ValueError:
            ts = -1
        rows.append((p, vid, ts))

if not found_any:
    raise SystemExit('No frames found. Make sure frames/<video_slug>/*.jpg exist.')
pd.DataFrame(rows, columns=['frame_path','video_id','ts']).to_parquet('index/meta.parquet')
print('Saved index/meta.parquet, rows =', len(rows))