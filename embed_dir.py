#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
embed_dir.py — Build a FAISS index from images using BLIP-2 caption + OCR + (objects) -> E5 text embeddings.

Pipeline (matches web_interface_v2.py):
  image --> caption (BLIP-2) + ocr (engine OCR) + objects (json)
        --> text --> E5 ("passage:" prefix, normalized)
        --> cosine similarity via FAISS IndexFlatIP (inner product on normalized vectors)

Features
- Accepts a single file or a directory (recursive).
- Resume-safe: skips files already listed in meta.json in --index-dir.
- Progress logging: processed/total, %, embedded/skipped/failed.
- Periodic flush of faiss.index + meta.json (--flush-every).
- object_weight controls influence of detector labels in the fused vector.

Tip about captions drifting languages: fix the BLIP-2 prompt inside your engine if needed.
This script calls engine._blip2_caption(PIL.Image or path) and engine._ocr_text(PIL.Image).
"""

import os
import sys
import json
import time
import glob
import inspect
from typing import List, Tuple, Set, Dict, Any, Optional

# Mitigate common OpenMP conflicts on Windows stacks (optional but helpful)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import faiss
from PIL import Image

# Import backend engine & helper
from web_interface_v2 import Blip2OCREngine, extract_video_and_frame

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(root: str) -> List[str]:
    if os.path.isfile(root):
        return [root] if os.path.splitext(root)[1].lower() in IMG_EXTS else []
    files: List[str] = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
    return sorted(set(files))

def load_existing_meta(index_dir: str) -> Tuple[List[Dict[str, Any]], Set[str]]:
    meta_path = os.path.join(index_dir, "meta.json")
    if not os.path.exists(meta_path):
        return [], set()
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        existing = {m.get("path") for m in meta if m.get("path")}
        return meta, existing
    except Exception:
        return [], set()

def ensure_index(engine: Blip2OCREngine, index_dir: str):
    os.makedirs(index_dir, exist_ok=True)
    # Try to load existing index via engine API
    try:
        info = engine.load_index()
        if isinstance(info, dict):
            print(f"[BUILD] Loaded existing index from {index_dir} (num_images={info.get('num_images')})")
        else:
            print(f"[BUILD] Loaded existing index from {index_dir}")
        return
    except Exception:
        pass
    # Create a new FAISS index with IP (cosine on normalized vectors)
    try:
        dim = engine.text_encoder.get_sentence_embedding_dimension()
    except Exception:
        probe = engine.text_encoder.encode(["passage: probe"], normalize_embeddings=True, convert_to_numpy=True)
        dim = int(probe.shape[1])
    engine.faiss_index = faiss.IndexFlatIP(dim)
    engine.meta, engine.ids = [], []
    print(f"[BUILD] Created new FAISS index (dim={dim}) in {index_dir}")

def safe_caption(engine: Blip2OCREngine, pil_img: Image.Image, path: str) -> str:
    # Prefer PIL-based API; fall back to path-based if the engine expects it
    try:
        return (engine._blip2_caption(pil_img) or "").strip()
    except TypeError:
        try:
            return (engine._blip2_caption(path) or "").strip()
        except Exception:
            return ""
    except Exception:
        return ""

def safe_ocr(engine: Blip2OCREngine, pil_img: Image.Image) -> str:
    try:
        txt = engine._ocr_text(pil_img)
        return (txt or "").strip()
    except Exception:
        return ""

def read_objects(objects_root: str, video: str, frame: Any) -> Tuple[List[str], str]:
    jpath = os.path.join(objects_root, video, f"{frame}.json")
    if not os.path.exists(jpath):
        return [], ""
    try:
        with open(jpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            objs = sorted(data, key=lambda k: -float(data[k]))
        elif isinstance(data, list):
            objs = [str(x) for x in data]
        else:
            objs = []
        return objs, ", ".join(objs)
    except Exception:
        return [], ""

def embed_texts(engine: Blip2OCREngine, texts: List[str]) -> np.ndarray:
    """Use engine._embed_texts if available (adds 'passage:'), else fall back to text_encoder directly."""
    if hasattr(engine, "_embed_texts") and callable(getattr(engine, "_embed_texts")):
        return engine._embed_texts(texts).astype("float32")
    # Fallback — prepend 'passage:' to match E5 training prompt
    prefixed = [f"passage: {t}" for t in texts]
    return engine.text_encoder.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True).astype("float32")

def fuse_embeddings(engine: Blip2OCREngine, main_text: str, obj_text: str, object_weight: float) -> np.ndarray:
    main = embed_texts(engine, [main_text])
    if not obj_text or object_weight == 0:
        return main[0]
    obj = embed_texts(engine, [obj_text])
    fused = main + float(object_weight) * obj
    fused = fused / (np.linalg.norm(fused, axis=1, keepdims=True) + 1e-12)
    return fused[0].astype("float32")

def save_state(engine: Blip2OCREngine, index_dir: str):
    faiss.write_index(engine.faiss_index, os.path.join(index_dir, "faiss.index"))
    with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(engine.meta, f, ensure_ascii=False)

def human_pct(n: int, d: int) -> str:
    return f"{(100.0 * n / d):.2f}%" if d > 0 else "0%"

def embed_path(
    path: str,
    index_dir: str = "index_blip2_ocr",
    objects_root: str = "objects",
    blip2_model: str = "Salesforce/blip2-flan-t5-xl",
    text_model: str = "intfloat/multilingual-e5-base",
    object_weight: float = 1.3,
    flush_every: int = 1000,
    caption_lang: Optional[str] = None,
    caption_prompt: Optional[str] = None,
    use_objects: bool = True,
):
    t0 = time.time()
    files = list_images(path)
    total = len(files)
    print(f"[BUILD] Scanning path = {path}")
    print(f"[BUILD] Found {total} image files.")

    # Prepare engine kwargs and include optional caption args only if supported
    engine_kwargs = dict(
        blip2_model=blip2_model,
        text_embed_model=text_model,
        index_dir=index_dir,
        objects_root=objects_root,
        use_objects=use_objects,
        object_topk=10,
        object_weight=object_weight,
        load_blip2=True,
        load_ocr=True,
    )
    sig = inspect.signature(Blip2OCREngine.__init__)
    if "caption_lang" in sig.parameters and caption_lang is not None:
        engine_kwargs["caption_lang"] = caption_lang
    if "caption_prompt" in sig.parameters and caption_prompt is not None:
        engine_kwargs["caption_prompt"] = caption_prompt

    engine = Blip2OCREngine(**engine_kwargs)
    ensure_index(engine, index_dir)

    # Load existing meta for resume AFTER ensure_index
    engine.meta, existing = load_existing_meta(index_dir)
    if existing:
        print(f"[BUILD] Resume mode ON. Existing vectors: {len(existing)}")

    counters = {"processed": 0, "embedded": 0, "skipped": 0, "failed": 0}

    for i, img_path in enumerate(files, 1):
        counters["processed"] += 1

        # Skip if already in meta.json
        if img_path in existing:
            counters["skipped"] += 1
            if i % 100 == 0 or i == total:
                print(f"[SKIP ] {i}/{total} ({human_pct(i, total)}). Skipped so far: {counters['skipped']}")
            continue

        try:
            im = Image.open(img_path).convert("RGB")
            cap = safe_caption(engine, im, img_path)
            ocr = safe_ocr(engine, im)
            video, frame = extract_video_and_frame(img_path)

            objs: List[str] = []
            obj_text = ""
            if use_objects:
                objs, obj_text = read_objects(objects_root, video, frame)

            # Build main text
            if cap and ocr:
                main_text = f"{cap} [SEP] {ocr}"
            else:
                main_text = cap or ocr or ""

            vec = fuse_embeddings(engine, main_text, obj_text, object_weight)

            engine.faiss_index.add(vec.reshape(1, -1))
            # frame might be a string; try to coerce to int when possible
            try:
                frame_val: Any = int(frame)
            except Exception:
                frame_val = frame
            engine.meta.append({
                "path": img_path,
                "video": video,
                "frame": frame_val,
                "caption": cap,
                "ocr": ocr,
                "objects": objs
            })
            existing.add(img_path)
            counters["embedded"] += 1

            if (i % 100 == 0) or (i == total) or (counters["embedded"] % 50 == 0):
                print(f"[OK   ] {i}/{total} ({human_pct(i, total)}) | embedded={counters['embedded']} skipped={counters['skipped']} failed={counters['failed']}")

            if (counters["embedded"] > 0) and (counters["embedded"] % flush_every == 0):
                save_state(engine, index_dir)
                print(f"[SAVE ] Flushed at embedded={counters['embedded']}")

        except KeyboardInterrupt:
            print("\n[INT  ] Keyboard interrupt. Saving and exit...")
            save_state(engine, index_dir)
            raise
        except Exception as e:
            counters["failed"] += 1
            if (i % 50 == 0) or (i == total):
                print(f"[FAIL ] {i}/{total} ({human_pct(i, total)}) | {e}")

    save_state(engine, index_dir)
    dt = time.time() - t0
    speed = (total / dt) if dt > 0 else 0.0
    print("\n===== BUILD SUMMARY =====")
    print(f"Total files     : {total}")
    print(f"Processed       : {counters['processed']}")
    print(f"Embedded        : {counters['embedded']}")
    print(f"Skipped(existing): {counters['skipped']}")
    print(f"Failed          : {counters['failed']}")
    print(f"Elapsed         : {dt:.1f}s ({speed:.2f} img/s)")

def main():
    import argparse
    p = argparse.ArgumentParser(description="Embed all images under a path using BLIP-2 + OCR + E5 (cosine via FAISS IP).")
    p.add_argument("path", help="File or directory to scan for images")
    p.add_argument("--index-dir", default="index_blip2_ocr")
    p.add_argument("--objects-root", default="objects")
    p.add_argument("--blip2-model", default="Salesforce/blip2-opt-2.7b")
    p.add_argument("--text-model", default="intfloat/multilingual-e5-base")
    p.add_argument("--object-weight", type=float, default=1.3, help="Weight for object labels when fusing embeddings (0 disables objects influence).")
    p.add_argument("--flush-every", type=int, default=1000, help="Flush FAISS/meta to disk every N embeddings.")
    p.add_argument("--caption-lang", default=None, help="(Optional) Force caption language, passed to engine if supported.")
    p.add_argument("--caption-prompt", default=None, help="(Optional) Custom caption prompt, passed to engine if supported.")
    p.add_argument("--no-objects", action="store_true", help="Disable using object labels (equivalent to object_weight=0).")
    args = p.parse_args()

    use_objects = not args.no_objects
    if args.no_objects:
        # keep the flag but also set weight to 0 to ensure no effect in fusion
        args.object_weight = 0.0

    embed_path(
        args.path,
        index_dir=args.index_dir,
        objects_root=args.objects_root,
        blip2_model=args.blip2_model,
        text_model=args.text_model,
        object_weight=args.object_weight,
        flush_every=args.flush_every,
        caption_lang=args.caption_lang,
        caption_prompt=args.caption_prompt,
        use_objects=use_objects,
    )

if __name__ == "__main__":
    main()
