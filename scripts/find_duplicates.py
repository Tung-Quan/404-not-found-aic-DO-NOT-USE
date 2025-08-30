"""
Find duplicate and unnecessary files in the repo.
- Uses the previously-created _workspace_report.json for the file list.
- Computes SHA1 for groups with equal sizes. For files <=100MB it computes full SHA1; for larger files it computes a sample hash (first+last 4MB) to avoid long reads.
- Skips virtualenv internals (paths starting with .venv) and common cache dirs.
- Writes report to _dup_report.json in repo root.

Run: python scripts/find_duplicates.py
"""
import os, json, hashlib
ROOT = r'e:\\Disk_D\\BK_LEARNING\\LEARNING\\react\\Project'
REPORT_IN = os.path.join(ROOT, '_workspace_report.json')
REPORT_OUT = os.path.join(ROOT, '_dup_report.json')
try:
    all_info = json.load(open(REPORT_IN, 'r', encoding='utf-8'))
except Exception as e:
    print('Failed to load', REPORT_IN, '->', e)
    raise
files = all_info.get('files', [])
# Filter
def skip_path(p):
    if p.startswith('.venv') or p.startswith('.cache'):
        return True
    return False
items = [f for f in files if not skip_path(f['path'])]
# group by size
from collections import defaultdict
size_map = defaultdict(list)
for f in items:
    size = f.get('size')
    if size is None:
        continue
    size_map[size].append(f['path'])

# helpers
BUF = 4 * 1024 * 1024  # 4MB sample for big files
SMALL_LIMIT = 100 * 1024 * 1024  # 100MB

def sha1_full(path):
    h = hashlib.sha1()
    fp = os.path.join(ROOT, path)
    try:
        with open(fp, 'rb') as fh:
            while True:
                data = fh.read(1<<20)
                if not data:
                    break
                h.update(data)
        return h.hexdigest()
    except Exception as e:
        return None

def sha1_sample(path):
    h = hashlib.sha1()
    fp = os.path.join(ROOT, path)
    try:
        sz = os.path.getsize(fp)
        with open(fp, 'rb') as fh:
            head = fh.read(BUF)
            if sz > BUF:
                fh.seek(max(0, sz-BUF))
                tail = fh.read(BUF)
            else:
                tail = b''
        h.update(head)
        h.update(tail)
        h.update(str(sz).encode())
        return h.hexdigest()
    except Exception:
        return None

duplicate_groups = []
zero_byte_files = []
large_probable_duplicates = []

for size, paths in size_map.items():
    if len(paths) <= 1:
        continue
    if size == 0:
        zero_byte_files.extend(paths)
        continue
    # for groups with multiple files, compute hash strategy
    hash_map = defaultdict(list)
    if size <= SMALL_LIMIT:
        for p in paths:
            h = sha1_full(p)
            hash_map[h].append(p)
    else:
        for p in paths:
            h = sha1_sample(p)
            hash_map[h].append(p)
    for h, group in hash_map.items():
        if not h:
            continue
        if len(group) >= 2:
            if size <= SMALL_LIMIT:
                duplicate_groups.append({'size': size, 'hash': h, 'paths': group, 'type': 'exact'})
            else:
                large_probable_duplicates.append({'size': size, 'hash': h, 'paths': group, 'type': 'probable'})

# build candidate removal list: keep first path, mark others
candidates = []
for g in duplicate_groups + large_probable_duplicates:
    keep = g['paths'][0]
    remove = g['paths'][1:]
    if remove:
        candidates.append({'keep': keep, 'remove': remove, 'size': g['size'], 'type': g['type']})

out = {
    'total_files_scan': len(files),
    'zero_byte_files': zero_byte_files,
    'duplicate_groups_exact': duplicate_groups,
    'duplicate_groups_probable_large': large_probable_duplicates,
    'candidates_to_remove': candidates,
    'empty_dirs': all_info.get('empty_dirs', [])
}
json.dump(out, open(REPORT_OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
print('WROTE', REPORT_OUT)
print('zero_byte_files:', len(zero_byte_files))
print('exact duplicate groups:', len(duplicate_groups))
print('probable large duplicate groups:', len(large_probable_duplicates))
print('candidates_to_remove:', len(candidates))
