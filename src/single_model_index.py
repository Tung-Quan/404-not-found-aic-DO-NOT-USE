"""
Minimal reusable module for single-model CLIP indexing and rerank.
Exports:
- CLIPEncoder
- build_embeddings_and_index
- load_index_and_metadata
- retrieve_and_rerank

This is a lightweight extraction of the notebook functions so they can be run from a script.
"""
from pathlib import Path
import os
import json
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

# optional deps
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    from transformers import CLIPProcessor, CLIPModel
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

from PIL import Image

ROOT = Path('.').resolve()
CANDIDATE_FRAME_DIRS = [Path('frames'), ROOT / 'frames', Path.cwd() / 'frames']
MAP_KEYFRAMES_DIRS = [Path('map-keyframes'), ROOT / 'map-keyframes']
OBJECTS_DIRS = [Path('objects'), ROOT / 'objects']
OUTPUT_DIR = Path('index_single_model')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME_DEFAULT = 'openai/clip-vit-large-patch14'
BATCH_SIZE_DEFAULT = 16

EMBEDDINGS_FILE = OUTPUT_DIR / 'embeddings.npy'
PATHS_FILE = OUTPUT_DIR / 'paths.txt'
METADATA_FILE = OUTPUT_DIR / 'paths_metadata.json'
FAISS_INDEX_FILE = OUTPUT_DIR / 'faiss.index'

RERANK_WEIGHTS = {'image': 1.0, 'object_text': 0.6}


class CLIPEncoder:
    def __init__(self, model_name: str = MODEL_NAME_DEFAULT, device: Optional[str] = None):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError('transformers is required')
        self.model_name = model_name
        self.device = device or ('cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name)
        if self.device == 'cuda' and TORCH_AVAILABLE:
            try:
                self.model.to('cuda')
            except Exception:
                self.device = 'cpu'
        self.model.eval()

    def encode_image(self, image_path_or_pil) -> Optional[np.ndarray]:
        try:
            if isinstance(image_path_or_pil, (str, Path)):
                img = Image.open(str(image_path_or_pil)).convert('RGB')
            else:
                img = image_path_or_pil
        except Exception:
            return None
        inputs = self.processor(images=[img], return_tensors='pt', padding=True)
        if TORCH_AVAILABLE:
            if self.device == 'cuda':
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            with torch.no_grad():
                feats = self.model.get_image_features(**inputs)
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                vec = feats.cpu().numpy().reshape(-1)
                return vec.astype('float32')
        return None

    def encode_text(self, text: str) -> Optional[np.ndarray]:
        inputs = self.processor(text=[text], return_tensors='pt', padding=True)
        if TORCH_AVAILABLE:
            if self.device == 'cuda':
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            with torch.no_grad():
                feats = self.model.get_text_features(**inputs)
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                vec = feats.cpu().numpy().reshape(-1)
                return vec.astype('float32')
        return None


# utilities

def find_frames(candidate_dirs=CANDIDATE_FRAME_DIRS, exts=('jpg', 'jpeg', 'png')) -> List[Path]:
    found = []
    for d in candidate_dirs:
        d = Path(d)
        if not d.exists():
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith(exts):
                    found.append(Path(root) / f)
        if found:
            break
    return sorted(found)


def _csv_matches_frame(row: dict, frame_stem: str) -> bool:
    for key in ('frame_idx', 'frame', 'n'):
        if key in row and row[key]:
            try:
                if str(int(float(row[key]))) == frame_stem:
                    return True
            except Exception:
                pass
    if 'file' in row and Path(row['file']).stem == frame_stem:
        return True
    return False


def get_frame_pts(frame_path: Path, map_dirs=MAP_KEYFRAMES_DIRS) -> Optional[float]:
    frame_stem = frame_path.stem
    video_name = frame_path.parent.name
    for base in map_dirs:
        base = Path(base)
        if not base.exists():
            continue
        for csvf in base.rglob('*.csv'):
            try:
                with open(csvf, newline='', encoding='utf-8') as fh:
                    import csv
                    reader = csv.DictReader(fh)
                    for row in reader:
                        if _csv_matches_frame(row, frame_stem):
                            for k in ('pts_time', 'timestamp', 'time', 'pts'):
                                if k in row and row[k]:
                                    try:
                                        return float(row[k])
                                    except Exception:
                                        pass
            except Exception:
                continue
    return None


def yield_batches(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i+batch_size]


def l2norm_np(v: np.ndarray) -> np.ndarray:
    v = v.astype('float32')
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


# main index build

def build_embeddings_and_index(encoder: CLIPEncoder, output_dir: Path = OUTPUT_DIR, rebuild: bool = False, batch_size: int = BATCH_SIZE_DEFAULT) -> Tuple[np.ndarray, List[str]]:
    frames = find_frames()
    if not frames:
        return np.empty((0, 512), dtype='float32'), []
    embeddings_file = output_dir / 'embeddings.npy'
    paths_file = output_dir / 'paths.txt'
    metadata_file = output_dir / 'paths_metadata.json'
    faiss_index_file = output_dir / 'faiss.index'

    if embeddings_file.exists() and paths_file.exists() and not rebuild:
        E = np.load(str(embeddings_file))
        with open(paths_file, 'r', encoding='utf-8') as fh:
            paths = [l.strip() for l in fh if l.strip()]
        return E, paths

    vectors = []
    paths = []
    for batch in yield_batches(frames, batch_size):
        for p in batch:
            vec = encoder.encode_image(p)
            if vec is None:
                continue
            vectors.append(vec)
            paths.append(str(Path(p).resolve()))
    if not vectors:
        return np.empty((0, 512), dtype='float32'), []
    E = np.vstack(vectors).astype('float32')
    E = l2norm_np(E)
    np.save(str(embeddings_file), E)
    with open(paths_file, 'w', encoding='utf-8') as fh:
        for p in paths:
            fh.write(p + '\n')
    meta = {}
    for p in paths:
        pt = get_frame_pts(Path(p))
        if pt is not None:
            meta[p] = {'pts_time': pt}
    if meta:
        with open(metadata_file, 'w', encoding='utf-8') as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

    if FAISS_AVAILABLE:
        try:
            dim = E.shape[1]
            index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(E)
            index.add(E)
            faiss.write_index(index, str(faiss_index_file))
        except Exception:
            pass
    return E, paths


def load_index_and_metadata(output_dir: Path = OUTPUT_DIR) -> Tuple[Optional[np.ndarray], List[str], Dict[str, Any], Any]:
    embeddings_file = output_dir / 'embeddings.npy'
    paths_file = output_dir / 'paths.txt'
    metadata_file = output_dir / 'paths_metadata.json'
    faiss_index_file = output_dir / 'faiss.index'
    E = None
    paths = []
    meta = {}
    if embeddings_file.exists():
        E = np.load(str(embeddings_file))
    if paths_file.exists():
        with open(paths_file, 'r', encoding='utf-8') as fh:
            paths = [l.strip() for l in fh if l.strip()]
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r', encoding='utf-8') as fh:
                meta = json.load(fh)
        except Exception:
            meta = {}
    index = None
    if FAISS_AVAILABLE and faiss_index_file.exists():
        try:
            index = faiss.read_index(str(faiss_index_file))
        except Exception:
            index = None
    return E, paths, meta, index


def retrieve_and_rerank(query: str, encoder: CLIPEncoder, top_k: int = 20, rerank_k: int = 8, output_dir: Path = OUTPUT_DIR) -> List[Dict[str, Any]]:
    E, paths, meta, index = load_index_and_metadata(output_dir)
    if E is None or not paths:
        return []
    qv = encoder.encode_text(query)
    if qv is None:
        return []
    qv = qv.astype('float32')
    if index is not None:
        q = qv.reshape(1, -1).astype('float32')
        faiss.normalize_L2(q)
        D, I = index.search(q, top_k)
        cand_idx = [int(x) for x in I[0] if x >= 0]
    else:
        sims = (E @ qv).astype('float32')
        cand_idx = list(np.argsort(sims)[-top_k:][::-1])
    items = []
    for idx in cand_idx[:rerank_k]:
        path = Path(paths[idx])
        img_vec = E[idx].astype('float32')
        precise_sim = float(np.dot(img_vec, qv))
        score = RERANK_WEIGHTS.get('image', 1.0) * precise_sim
        obj_sim = 0.0
        for od in OBJECTS_DIRS:
            of = Path(od) / path.parent.name / (path.stem + '.json')
            if of.exists():
                try:
                    with open(of, 'r', encoding='utf-8') as fh:
                        obj = json.load(fh)
                        texts = []
                        if isinstance(obj, dict):
                            if 'labels' in obj and isinstance(obj['labels'], list):
                                texts.extend([str(x) for x in obj['labels']])
                            if 'text' in obj and obj['text']:
                                texts.append(str(obj['text']))
                        if texts and encoder is not None:
                            ot = ' '.join(texts)
                            otv = encoder.encode_text(ot)
                            if otv is not None:
                                obj_sim = float(np.dot(otv.astype('float32'), qv))
                except Exception:
                    pass
                break
        score += RERANK_WEIGHTS.get('object_text', 0.0) * obj_sim
        pts = None
        if str(paths[idx]) in meta:
            pts = meta[str(paths[idx])].get('pts_time')
        else:
            pts = get_frame_pts(path)
        items.append({'path': str(path), 'score': score, 'image_sim': precise_sim, 'obj_sim': obj_sim, 'pts_time': pts})
    items = sorted(items, key=lambda x: x['score'], reverse=True)
    return items
