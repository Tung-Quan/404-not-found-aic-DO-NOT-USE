#!/usr/bin/env python3
"""
Inspect index / embedding files in the repository.
- Finds files with extensions: .index, .faiss, .npy, .npz
- For .npy: open with numpy (mmap) to read shape and dtype without loading full array.
- For FAISS (.index): try to import faiss and read index metadata (ntotal, dimension if available).
- Writes JSON report to _index_report.json in repo root.

Run: python scripts/check_index.py
"""
import os
import json
import sys
import traceback
from pathlib import Path

report = {'found': [], 'errors': []}

patterns = ['.index','.faiss','.npy','.npz']

for dirpath, dirnames, filenames in os.walk(root):
    # skip .git
    if '.git' in dirpath:
        continue
    for fn in filenames:
        p = Path(dirpath) / fn
        if any(fn.lower().endswith(ext) for ext in patterns):
            item = {'path': str(p.relative_to(root)), 'size': None}
            try:
                item['size'] = p.stat().st_size
            except Exception as e:
                item['size'] = None
                report['errors'].append({'file': str(p), 'error': str(e)})
            # inspect specific types
            lower = fn.lower()
            try:
                if lower.endswith('.npy') or lower.endswith('.npz'):
                    try:
                        import numpy as _np
                        arr = _np.load(str(p), mmap_mode='r')
                        # for npz, arr may be NpzFile
                        if hasattr(arr, 'files'):
                            item['type'] = 'npz'
                            item['arrays'] = {name: {'shape': arr[name].shape, 'dtype': str(arr[name].dtype)} for name in arr.files}
                        else:
                            item['type'] = 'npy'
                            item['shape'] = getattr(arr, 'shape', None)
                            item['dtype'] = str(getattr(arr, 'dtype', None))
                        try:
                            arr._mmap is None
                        except Exception:
                            pass
                    except Exception as e:
                        item['inspect_error'] = repr(e)
                elif lower.endswith('.index') or lower.endswith('.faiss'):
                    # attempt to inspect with faiss if available
                    try:
                        import faiss
                        idx = faiss.read_index(str(p))
                        item['type'] = 'faiss'
                        item['ntotal'] = getattr(idx, 'ntotal', None)
                        # try to get dimension
                        d = getattr(idx, 'd', None)
                        if d is None:
                            # try to read from index.reconstruct if possible
                            try:
                                if idx.ntotal>0:
                                    v = faiss.vector_to_array(idx.reconstruct(0))
                                    item['dimension_guess'] = int(len(v))
                            except Exception:
                                item['dimension_guess'] = None
                        else:
                            item['dimension'] = int(d)
                    except Exception as e:
                        item['inspect_error'] = 'faiss not available or failed to read: '+repr(e)
                else:
                    item['type'] = 'other'
            except Exception as e:
                report['errors'].append({'file': str(p), 'error': traceback.format_exc()})
            report['found'].append(item)

out_path = root / '_index_report.json'
with open(out_path, 'w', encoding='utf-8') as fh:
    json.dump(report, fh, indent=2, ensure_ascii=False)

print('Wrote', out_path)
print('Found', len(report['found']), 'index-like files. Errors:', len(report['errors']))
# print short summary
for i in report['found'][:40]:
    print(i.get('path')[:200].ljust(80), i.get('type', ''), 'size=', i.get('size'), 'ntotal=', i.get('ntotal', ''))

sys.exit(0)
