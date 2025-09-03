"""
BLIP-2 + OCR + OBJECTS Retrieval API (async indexing + embedded % + custom CSV name)
- Index frames lớn theo batch để tiết kiệm RAM.
- Nội dung index = BLIP-2 caption + OCR text + OBJECT tags (objects/<video>/<frame>.json).
- Embedding = Sentence-Transformers (E5 multilingual) trên văn bản đại diện cho ảnh.
- Lập chỉ mục FAISS (inner-product ~ cosine vì vector đã normalize).
- Hỗ trợ: KIS / Q&A (BLIP-2 VQA) / TRAKE + Export CSV (đặt tên file theo input).
- Build index chạy nền (thread), có /api/status trả % embedded vào FAISS + ETA, /api/cancel để dừng.
"""
import os
# os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")
# os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import threading, time
from fastapi.staticfiles import StaticFiles
import re
import json
import time
import threading
from typing import List, Dict, Any, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

import numpy as np
# --- LOGGING TO TERMINAL ---
import logging, sys, os

logger = logging.getLogger("blip2_ocr")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("[%(levelname)s %(asctime)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(_h)
# Mặc định INFO; tăng DEBUG nếu muốn chi tiết hơn
logger.setLevel(logging.INFO)
# ---------------------------
logger.setLevel(logging.DEBUG)

# --------- Optional deps checks ----------
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

_HAS_EASYOCR = False
_HAS_PYTESSERACT = False
try:
    import easyocr
    _HAS_EASYOCR = True
except Exception:
    try:
        import pytesseract  # system tesseract
        _HAS_PYTESSERACT = True
    except Exception:
        pass

try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    _HAS_BLIP2 = True
except Exception:
    _HAS_BLIP2 = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False


def _safe_int(s, default=0):
    try:
        return int(s)
    except Exception:
        return default


import re

def extract_video_and_frame(path: str) -> Tuple[str, int]:
    """
    Từ đường dẫn .../frames/<video>/<frame_idx>.jpg -> ("<video>", <frame_idx:int>)
    Nếu tên file có chữ (vd: frame_000123.jpg) vẫn bắt được số cuối.
    """
    p = os.path.normpath(path)
    parts = p.split(os.sep)
    video = parts[-2] if len(parts) >= 2 else ""
    stem = os.path.splitext(parts[-1])[0]
    m = re.search(r"(\d+)$", stem)  # lấy cụm số ở CUỐI tên file
    frame = int(m.group(1)) if m else 0
    return video, frame

def _maybe_read_objects(self, objects_root: str, img_path: str) -> List[str]:
    video, frame = extract_video_and_frame(img_path)
    jpath = os.path.join(objects_root, video, f"{frame}.json")
    if not os.path.exists(jpath):
        return []
    try:
        with open(jpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        # chấp nhận nhiều định dạng: list[str] hoặc dict{name:score}
        if isinstance(data, dict):
            return sorted(data, key=lambda k: -float(data[k]))  # theo score nếu có
        if isinstance(data, list):
            return [str(x) for x in data]
    except Exception:
        return []
    return []


class Blip2OCREngine:
    def __init__(
        self,
        blip2_model: str = "Salesforce/blip2-flan-t5-xl",  # model hợp lệ
        text_embed_model: str = "intfloat/multilingual-e5-base",
        device: Optional[str] = None,
        ocr_langs: Optional[List[str]] = None,
        index_dir: str = "index_blip2_ocr",
        # objects
        objects_root: str = "objects",
        use_objects: bool = True,
        object_topk: int = 10,
        object_weight: float = 1.3,
        # optional
        load_blip2: bool = True,
        load_ocr: bool = True,
        hf_token: Optional[str] = None,
    ):
        # deps tối thiểu cho search
        missing = []
        if not _HAS_FAISS: missing.append("faiss")
        if not _HAS_ST: missing.append("sentence-transformers")
        if load_blip2:
            if not _HAS_TORCH: missing.append("torch")
            if not _HAS_BLIP2: missing.append("transformers (BLIP-2)")
            if not _HAS_PIL: missing.append("Pillow")
        if missing:
            raise RuntimeError(f"Missing deps: {', '.join(missing)}")

        self.device = device or ("cuda" if _HAS_TORCH and torch.cuda.is_available() else "cpu")
        self.index_dir = index_dir
        self.ocr_langs = ocr_langs or ["vi", "en"]


        # ==== STATE luôn có (fix AttributeError) ====
        self.ids: List[str] = []
        self.meta: List[Dict[str, Any]] = []
        self.faiss_index = None

        self._lock = threading.Lock()
        self.total_images = 0
        self.processed_images = 0
        self.embedded_images = 0
        self.current_path = ""
        self.start_time = 0.0
        self.cancel_flag = False
        self.building = False
        self.build_error: Optional[str] = None
        # ============================================

        # Objects
        self.objects_root = objects_root
        self.use_objects = bool(use_objects)
        self.object_topk = int(object_topk)
        self.object_weight = float(object_weight)

        # Text encoder (bắt buộc)
        self.text_encoder = SentenceTransformer(text_embed_model, device=self.device)

        # OCR (tuỳ chọn)
        self.reader = None
        if load_ocr and _HAS_EASYOCR:
            try:
                self.reader = easyocr.Reader(self.ocr_langs, gpu=(self.device.startswith("cuda")))
            except Exception:
                self.reader = None
        elif load_ocr and _HAS_PYTESSERACT:
            self.reader = "pytesseract"  # marker để dùng pytesseract trong _ocr_text

        # BLIP-2 (tuỳ chọn)
        self.processor = None
        self.blip2 = None
        if load_blip2:
            # self.processor = Blip2Processor.from_pretrained(blip2_model, token=hf_token)
            # self.blip2 = Blip2ForConditionalGeneration.from_pretrained(
            #     blip2_model,
            #     token=hf_token,
            #     torch_dtype=(torch.float16 if (_HAS_TORCH and torch.cuda.is_available()) else torch.float32)
            # ).to(self.device).eval()
            # Processor giữ nguyên (nhẹ, không tốn nhiều RAM)
            self.processor = Blip2Processor.from_pretrained(blip2_model)

            # ===== Robust loader cho BLIP-2, ưu tiên ít RAM =====
            bnb_ok = False
            try:
                # Thử quantization nếu bitsandbytes sẵn có (tiết kiệm RAM đáng kể)
                from transformers import BitsAndBytesConfig
                bnb_conf = BitsAndBytesConfig(
                    load_in_4bit=True,                      # thử 4-bit
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16 if self.device=="cuda" else torch.float32,
                )
                bnb_ok = True
            except Exception:
                bnb_ok = False

            try:
                if bnb_ok:
                    # 4-bit quant + auto offload
                    self.blip2 = Blip2ForConditionalGeneration.from_pretrained(
                        blip2_model,
                        quantization_config=bnb_conf,
                        device_map="auto",           # tự chia GPU/CPU
                        low_cpu_mem_usage=True,
                    )
                else:
                    # Không có bitsandbytes: dùng auto-offload sang disk để giảm RAM
                    offload_dir = os.path.join(self.index_dir, "_offload_blip2")
                    os.makedirs(offload_dir, exist_ok=True)
                    self.blip2 = Blip2ForConditionalGeneration.from_pretrained(
                        blip2_model,
                        device_map="auto",           # có GPU sẽ dùng, phần dư offload
                        torch_dtype=(torch.float16 if self.device=="cuda" else torch.float32),
                        low_cpu_mem_usage=True,      # nạp từng phần
                        offload_folder=offload_dir,  # thư mục offload trên đĩa
                    )
            except OSError as e:
                # Bắt lỗi Windows 1455 (paging file)
                msg = str(e)
                if ("1455" in msg) or ("paging file" in msg.lower()):
                    raise RuntimeError(
                        "Windows error 1455: Hết Virtual Memory khi load BLIP-2. "
                        "Hãy tăng Pagefile (System Properties → Advanced → Performance → Settings → Advanced → "
                        "Virtual memory → Change → bỏ tick Automatically → Custom size → Initial/Maximum ≥ 32768 MB), "
                        "hoặc cài bitsandbytes để load 4-bit (pip install bitsandbytes), "
                        "hoặc đổi sang offload_folder ổ đĩa còn nhiều dung lượng."
                    )
                else:
                    raise
            # ===== Hết phần robust loader =====

        
        self.log_every_n = int(os.getenv("LOG_EVERY_N", "100"))   # in mỗi 100 ảnh (mặc định)
        self.list_preview_n = int(os.getenv("LIST_PREVIEW_N", "20"))  # in trước 20 file đầu khi scan

    
    def reset_build_state(self):
        with self._lock:
            self.building = False
            self.build_error = None
            self.cancel_flag = False
            self.current_path = ""
            self.start_time = 0.0
            self.total_images = 0
            self.processed_images = 0
            self.embedded_images = 0

    # ---------- utils ----------
    def _gather_frames(self, frames_root: str) -> list:
        """Quét frames/<video>/<frame>.jpg (và .jpeg/.png/.bmp)"""
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = []
        root = os.path.normpath(frames_root)
        logger.info(f"Scanning frames_root = {root} ...")
        for r, _, fnames in os.walk(root):
            for fn in fnames:
                if os.path.splitext(fn)[1].lower() in exts:
                    files.append(os.path.join(r, fn))
        files.sort()
        logger.info(f"Found {len(files)} frame images under {root}.")
        if files:
            n = min(self.list_preview_n, len(files))
            logger.info("First %d files:\n%s", n, "\n".join(files[:n]))
        return files



    # ---------- OCR ----------
    def _ocr_text(self, image: "Image.Image") -> str:
        text = ""
        if self.reader is not None:
            try:
                import numpy as _np
                res = self.reader.readtext(_np.array(image))
                text = " ".join([t[1] for t in res]) if res else ""
            except Exception:
                text = ""
        elif _HAS_PYTESSERACT:
            try:
                import pytesseract
                text = pytesseract.image_to_string(image, lang="vie+eng")
            except Exception:
                text = ""
        return text.strip()

    # ---------- BLIP-2 ----------
    @torch.inference_mode()
    def _blip2_caption(self, img_path: str) -> str:
        if (self.processor is None) or (self.blip2 is None):
            raise RuntimeError("BLIP-2 not loaded. Initialize with load_blip2=True.")
        from PIL import Image
        image = Image.open(img_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        #Nếu vẫn nặng, giảm tiếp max_new_tokens xuống 16.
        gen_cfg = dict(
            max_new_tokens=25,   # ngắn gọn để nhanh
            num_beams=3,         # beam nhỏ cho tốc độ
            do_sample=False
        )
        with torch.inference_mode():
            if self.device == "cuda":
                with torch.cuda.amp.autocast():
                    out = self.blip2.generate(**inputs, **gen_cfg)
            else:
                out = self.blip2.generate(**inputs, **gen_cfg)
        text = self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        return text


    @torch.inference_mode()
    def _blip2_vqa(self, image: "Image.Image", question: str, max_new_tokens: int = 20) -> str:
        if (self.processor is None) or (self.blip2 is None):
            raise RuntimeError("BLIP-2 not loaded. Initialize with load_blip2=True.")

        prompt = f"Question: {question}\nAnswer in Vietnamese:"
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device, dtype=self.blip2.dtype)
        out = self.blip2.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()

    # ---------- Objects ----------
    def _objects_path(self, video: str, frame_idx: int) -> str:
        return os.path.join(self.objects_root, video, f"{frame_idx}.json")

    def _read_objects_top(self, video: str, frame_idx: int):
        p = self._objects_path(video, frame_idx)
        if not os.path.exists(p):
            return []
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return []
        labels = data.get("detection_class_entities") or data.get("detection_class_names") or []
        scores = data.get("detection_scores") or []
        try:
            scores = [float(x) for x in scores]
        except Exception:
            scores = [0.0] * len(labels)
        n = min(len(labels), len(scores))
        pairs = [(str(labels[i]).strip(), float(scores[i])) for i in range(n)]
        pairs.sort(key=lambda x: x[1], reverse=True)
        seen, out = set(), []
        for label, sc in pairs:
            if not label or label in seen:
                continue
            seen.add(label)
            out.append((label, sc))
            if len(out) >= self.object_topk:
                break
        return out

    def _objects_text_and_names(self, objs):
        names = [l for (l, _) in objs]
        weighted_tokens = []
        w = max(0.0, self.object_weight)
        for name, sc in objs:
            repeats = max(1, int(round(sc * w * 3)))
            weighted_tokens.extend([name] * repeats)
        text = " ".join(weighted_tokens) if weighted_tokens else ""
        return text, names

    # ---------- Embedding ----------
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        texts_proc = [f"passage: {t}" for t in texts]
        embs = self.text_encoder.encode(
            texts_proc,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embs.astype("float32")

    def _embed_query(self, query: str) -> np.ndarray:
        emb = self.text_encoder.encode([f"query: {query}"], convert_to_numpy=True, normalize_embeddings=True)
        return emb.astype("float32")

    # ---------- Build/Load ----------
    def build_index_stream(self, frames_root: str, batch_size: int = 32, save_embeddings_npy: bool = False):
        import json, time, numpy as np, faiss, os, re
        files = self._gather_frames(frames_root)

        with self._lock:
            self.building = True
            self.build_error = None
            self.cancel_flag = False
            self.start_time = time.time()
            self.total_images = len(files)
            self.processed_images = 0
            self.embedded_images = 0
            self.current_path = ""

        if len(files) == 0:
            logger.warning("No frame images found. Abort building.")
            with self._lock:
                self.building = False
            return

        # Chuẩn bị FAISS (cosine qua IP + normalize)
        os.makedirs(self.index_dir, exist_ok=True)
        dim = getattr(self.text_encoder, "get_sentence_embedding_dimension", lambda: None)()
        if dim is None:
            dim = int(self.text_encoder.encode(["probe"], normalize_embeddings=True).shape[1])
        if self.faiss_index is None:
            self.faiss_index = faiss.IndexFlatIP(dim)

        # embeddings memmap (tuỳ chọn)
        mmap = None
        if save_embeddings_npy:
            emb_path = os.path.join(self.index_dir, "embeddings.npy")
            mmap = np.memmap(emb_path, dtype="float32", mode="w+", shape=(self.total_images, dim))

        # reset meta/ids
        self.meta, self.ids = [], []

        logger.info(f"[BUILD] Streaming index with BLIP-2 + OCR | total={self.total_images} | dim={dim}")

        def extract_video_and_frame(path: str):
            p = os.path.normpath(path)
            parts = p.split(os.sep)
            video = parts[-2] if len(parts) >= 2 else ""
            stem = os.path.splitext(parts[-1])[0]
            m = re.search(r"(\d+)$", stem)
            frame = int(m.group(1)) if m else 0
            return video, frame

        def read_objects(img_path: str):
            if not getattr(self, "use_objects", False):
                return []
            try:
                video, frame = extract_video_and_frame(img_path)
                jpath = os.path.join(getattr(self, "objects_root", "objects"), video, f"{frame}.json")
                if not os.path.exists(jpath):
                    return []
                with open(jpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    objs = sorted(data.items(), key=lambda kv: -float(kv[1]))
                    objs = [k for k, _ in objs]
                elif isinstance(data, list):
                    objs = [str(x) for x in data]
                else:
                    objs = []
                k = int(getattr(self, "object_topk", 10))
                return objs[:k] if k > 0 else objs
            except Exception:
                return []

        log_every = max(1, int(getattr(self, "log_every_n", 100)))
        for idx, p in enumerate(files):
            if self.cancel_flag:
                logger.warning("Cancel flag detected. Stopping build loop.")
                break

            with self._lock:
                self.current_path = p
            if (idx % log_every) == 0:
                logger.info("[Build %d/%d] %s", idx + 1, self.total_images, p)

            # 1) Caption & OCR & Objects
            try:
                cap = self._blip2_caption(p) or ""
            except Exception as e:
                logger.debug("Caption error on %s: %s", p, e)
                cap = ""
            try:
                ocr = self._ocr_text(p) or ""
            except Exception as e:
                logger.debug("OCR error on %s: %s", p, e)
                ocr = ""
            objs = read_objects(p)
            obj_text = ", ".join(objs) if objs else ""

            # 2) Hợp văn bản chính
            if cap and ocr:
                main_text = f"{cap} [SEP] {ocr}"
            else:
                main_text = cap or ocr

            # 3) Nhúng văn bản (E5) + fuse objects có trọng số
            main_emb = self.text_encoder.encode([main_text], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
            if obj_text:
                obj_emb = self.text_encoder.encode([obj_text], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
                fused = main_emb + float(getattr(self, "object_weight", 1.3)) * obj_emb
                fused = fused / (np.linalg.norm(fused, axis=1, keepdims=True) + 1e-12)
                vec = fused[0]
            else:
                vec = main_emb[0]  # đã normalize

            # 4) Add vào FAISS ngay (streaming)
            self.faiss_index.add(vec.reshape(1, -1))
            if mmap is not None:
                mmap[idx, :] = vec

            # 5) Cập nhật tiến độ & meta
            v, fr = extract_video_and_frame(p)
            self.meta.append({
                "path": p, "video": v, "frame": int(fr),
                "caption": cap, "ocr": ocr,
                "objects": objs
            })
            self.ids.append(p)

            with self._lock:
                self.embedded_images += 1
                self.processed_images += 1

        # 6) Lưu index + meta
        try:
            faiss.write_index(self.faiss_index, os.path.join(self.index_dir, "faiss.index"))
            with open(os.path.join(self.index_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(self.meta, f, ensure_ascii=False)
            if mmap is not None:
                mmap.flush()
                del mmap
        except Exception as e:
            with self._lock:
                self.build_error = f"Write error: {e}"
            logger.error("Write error: %s", e)

        elapsed = time.time() - self.start_time
        with self._lock:
            pct = (self.embedded_images * 100.0 / self.total_images) if self.total_images else 0.0
            self.current_path = ""
            self.building = False
        logger.info("[BUILD DONE] embedded=%d / total=%d (%.2f%%) in %.1fs",
                    self.embedded_images, self.total_images, pct, elapsed)


    def load_index(self) -> Dict[str, Any]:
        idx_path = os.path.join(self.index_dir, "faiss.index")
        meta_path = os.path.join(self.index_dir, "meta.json")
        if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
            raise RuntimeError(f"Index not found in {self.index_dir}. Build index first.")
        self.faiss_index = faiss.read_index(idx_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.ids = [m["path"] for m in self.meta]
        return {"num_images": len(self.meta), "index_dir": self.index_dir}

    # ---------- Search primitives ----------
    def _search(self, query: str, topk: int = 50, restrict_video: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.faiss_index is None:
            raise RuntimeError("Index not loaded. Call load_index or wait for build to finish.")
        q_emb = self._embed_query(query)
        D, I = self.faiss_index.search(q_emb, topk)
        I = I[0].tolist()
        D = D[0].tolist()
        results = []
        for score, idx in zip(D, I):
            if idx < 0:
                continue
            m = self.meta[idx]
            if (restrict_video is None) or (m["video"] == restrict_video):
                r = dict(m)
                r["score"] = float(score)
                results.append(r)
        return results

    # ---------- Public tasks ----------
    def textual_kis(self, query: str, topk: int = 10) -> List[Dict[str, Any]]:
        return self._search(query, topk=topk)

    def qa(self, question: str, topk: int = 10) -> Dict[str, Any]:
        from PIL import Image
        cands = self._search(question, topk=topk)
        if not cands:
            return {"video": "", "frame": 0, "answer": ""}
        best = cands[0]
        try:
            im = Image.open(best["path"]).convert("RGB")
            ans = self._blip2_vqa(im, question)
        except Exception:
            ans = ""
        return {"video": best["video"], "frame": int(best["frame"]), "answer": ans}

    def trake(self, query: str, topk: int = 50) -> Dict[str, Any]:
        parts = re.split(r"\s*(?:->|,|;| và rồi | rồi | sau đó | tiếp theo | then | next | and then )\s*", query, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) < 2:
            cands = self._search(query, topk=topk)
            if not cands:
                return {"video": "", "frames": []}
            video = cands[0]["video"]
            sel = [c for c in cands if c["video"] == video][:4]
            return {"video": video, "frames": [int(c["frame"]) for c in sel]}

        cands = self._search(parts[0], topk=topk)
        if not cands:
            return {"video": "", "frames": []}
        video = cands[0]["video"]

        frames = []
        last_frame = -1
        for p in parts:
            rcands = self._search(p, topk=topk, restrict_video=video)
            if not rcands:
                continue
            choice = rcands[0]
            for c in rcands:
                if c["frame"] > last_frame:
                    choice = c
                    break
            frames.append(int(choice["frame"]))
            last_frame = int(choice["frame"])

        frames_sorted = []
        prev = -1
        for f in frames:
            if f > prev:
                frames_sorted.append(f)
                prev = f
        return {"video": video, "frames": frames_sorted}

    # ---------- CSV helpers ----------
    def export_csv_kis_lines(self, queries: List[str]) -> List[str]:
        lines = []
        for q in queries[:100]:
            res = self.textual_kis(q, topk=1)
            if not res:
                continue
            r = res[0]
            lines.append(f"{r['video']},{int(r['frame'])}")
        return lines

    def export_csv_qa_lines(self, queries: List[str]) -> List[str]:
        lines = []
        for q in queries[:100]:
            out = self.qa(q, topk=10)
            if not out.get("video"):
                continue
            ans = out.get("answer", "")
            if ("," in ans) or ('"' in ans) or ("\n" in ans):
                ans = '"' + ans.replace('"', '""') + '"'
            lines.append(f"{out['video']},{int(out['frame'])},{ans if ans else ''}")
        return lines

    def export_csv_trake_lines(self, queries: List[str]) -> List[str]:
        lines = []
        for q in queries[:100]:
            out = self.trake(q, topk=50)
            if not out.get("video") or not out.get("frames"):
                continue
            parts = ",".join(str(int(f)) for f in out["frames"])
            lines.append(f"{out['video']},{parts}")
        return lines

    # ---------- Status ----------
    def status(self) -> Dict[str, Any]:
        with self._lock:
            done = self.processed_images
            total = self.total_images
            emb = self.embedded_images
            pct_proc = (100.0 * done / total) if total > 0 else 0.0
            pct_emb = (100.0 * emb / total) if total > 0 else 0.0
            eta = None
            if total > 0 and emb > 0:
                elapsed = max(1e-6, time.time() - self.start_time)
                rate = emb / elapsed
                remain = max(0, total - emb)
                eta = remain / max(1e-6, rate)
            return {
                "engine_ready": True,
                "building": self.building,
                "processed": done,
                "embedded": emb,
                "total": total,
                "pct": pct_proc,             # % processed (tham khảo)
                "pct_embedded": pct_emb,     # % đã add vào FAISS (quan trọng)
                "current_path": self.current_path,
                "eta_sec": eta,
                "error": self.build_error,
                "index_dir": self.index_dir
            }

    def cancel(self):
        if hasattr(self, "_lock"):
            with self._lock:
                self.cancel_flag = True
        else:
            self.cancel_flag = True
            
    #---- LOAD EXTERNAL INDEX ----------
    def load_external_index(self, faiss_path: str, metadata_path: Optional[str] = None) -> Dict[str, Any]:
        """Load FAISS index & metadata từ đường dẫn tùy ý (faiss/json/pkl)."""
        if not os.path.exists(faiss_path):
            raise RuntimeError(f"FAISS file not found: {faiss_path}")
        self.faiss_index = faiss.read_index(faiss_path)

        meta: List[Dict[str, Any]] = []
        if metadata_path and os.path.exists(metadata_path):
            # Hỗ trợ .json hoặc .pkl
            if metadata_path.lower().endswith(".json"):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                # kỳ vọng list[dict]
                for m in raw:
                    p = m.get("path") or m.get("image_path") or ""
                    video, frame = extract_video_and_frame(p)
                    meta.append({
                        "path": p,
                        "video": m.get("video", video),
                        "frame": int(m.get("frame", frame)),
                        "caption": m.get("caption", ""),
                        "ocr": m.get("ocr", ""),
                        "objects": m.get("objects", []),
                    })
            else:
                import pickle
                with open(metadata_path, "rb") as f:
                    raw = pickle.load(f)
                # Chuẩn hóa từ metadata.pkl của pipeline cũ
                # Cố gắng map các field phổ biến: path / image_path, video, frame, extracted_text...
                for m in raw:
                    p = m.get("path") or m.get("image_path") or m.get("img_path") or ""
                    video, frame = extract_video_and_frame(p)
                    caption = m.get("caption", "")
                    # Nhiều pipeline cũ chỉ có OCR/extracted_text
                    ocr = m.get("ocr") or m.get("extracted_text") or ""
                    objs = m.get("objects") or m.get("object_names") or []
                    if isinstance(objs, dict):
                        objs = list(objs.keys())
                    meta.append({
                        "path": p,
                        "video": m.get("video", video),
                        "frame": int(m.get("frame", frame)),
                        "caption": caption or "",
                        "ocr": (ocr or "").strip(),
                        "objects": objs if isinstance(objs, list) else [],
                    })
        else:
            # Nếu không có metadata → suy ra tối thiểu từ path
            raise RuntimeError("Metadata file is required to map frames. Provide metadata_path (.pkl or .json).")

        if not meta:
            raise RuntimeError("Empty or unsupported metadata format.")

        self.meta = meta
        self.ids = [m["path"] for m in self.meta]
        # Không đổi self.index_dir; nhưng trả về để log
        return {"num_images": len(self.meta), "faiss_path": faiss_path, "metadata_path": metadata_path}


# ---------------- FastAPI ----------------
app = FastAPI(title="BLIP-2 + OCR + OBJECTS Retrieval API", version="2.3")
# phục vụ thư mục hiện tại (chứa index.html) tại đường dẫn /ui
app.mount("/ui", StaticFiles(directory=".", html=True), name="ui")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: Optional[Blip2OCREngine] = None
_build_thread: Optional[threading.Thread] = None

class InitBody(BaseModel):
    frames_root: str
    index_dir: str = "index_blip2_ocr"
    blip2_model: str = "Salesforce/blip2-flan-t5-base"
    text_embed_model: str = "intfloat/multilingual-e5-base"
    # Objects
    objects_root: str = "objects"
    use_objects: bool = True
    object_topk: int = 10
    object_weight: float = 1.3
    # Build options
    batch_size: int = 32
    save_embeddings_npy: bool = False

class SearchBody(BaseModel):
    query: str
    topk: int = 10

@app.post("/api/initialize")
def initialize(body: InitBody):
    global engine
    engine = Blip2OCREngine(
        blip2_model=body.blip2_model,
        text_embed_model=body.text_embed_model,
        index_dir=body.index_dir,
        objects_root=body.objects_root,
        use_objects=body.use_objects,
        object_topk=body.object_topk,
        object_weight=body.object_weight
    )
    return {"status": "ok", "device": engine.device, "index_dir": body.index_dir}

@app.post("/api/load_index")
def load_index(
    index_dir: Optional[str] = Form(None),
    faiss_path: Optional[str] = Form(None),
    metadata_path: Optional[str] = Form(None),
):
    global engine
    # Engine "nhẹ" cho kịch bản chỉ load index sẵn có
    if engine is None:
        engine = Blip2OCREngine(
            index_dir=index_dir or "index_blip2_ocr",
            load_blip2=False,
            load_ocr=False,
        )

    # Ưu tiên đường dẫn tuyệt đối do bạn nhập ở UI
    if faiss_path:
        try:
            out = engine.load_external_index(faiss_path=faiss_path, metadata_path=metadata_path)
            return {"status": "loaded_external", **out}
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Load external index failed: {e}")

    # Fallback: theo index_dir (yêu cầu tồn tại faiss.index + meta.json)
    idx_dir = index_dir or engine.index_dir
    idx_path = os.path.join(idx_dir, "faiss.index")
    meta_path = os.path.join(idx_dir, "meta.json")
    if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
        raise HTTPException(
            status_code=400,
            detail=f"Index not found in '{idx_dir}'. Hãy nhập 'FAISS path' + 'Metadata path' ở UI, hoặc build index."
        )
    try:
        engine.index_dir = idx_dir
        out = engine.load_index()
        return {"status": "loaded_dir", **out}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Load dir index failed: {e}")

_build_thread = None  # đặt ở mức module nếu chưa có

@app.post("/api/build_index")
def build_index(body: InitBody):
    global engine, _build_thread

    # Build cần BLIP-2 đã khởi tạo
    if engine is None:
        engine = Blip2OCREngine(
            blip2_model=body.blip2_model,
            text_embed_model=body.text_embed_model,
            index_dir=body.index_dir,
            objects_root=body.objects_root,
            use_objects=body.use_objects,
            object_topk=body.object_topk,
            object_weight=body.object_weight,
        )

    # Dọn thread cũ (nếu đã kết thúc)
    if _build_thread and not _build_thread.is_alive():
        _build_thread = None
        engine.building = False

    # Đang build thật sự?
    if engine.building or (_build_thread and _build_thread.is_alive()):
        return {"status": "already_building", **engine.status()}

    # Validate frames_root
    if not os.path.isdir(body.frames_root):
        raise HTTPException(status_code=400, detail=f"frames_root không tồn tại: {body.frames_root}")

    files = engine._gather_frames(body.frames_root)
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="Không tìm thấy ảnh (.jpg/.jpeg/.png/.bmp) trong frames_root")

    # ===== BẬT CỜ & SET COUNTERS TRƯỚC KHI START THREAD =====
    with engine._lock:
        engine.building = True
        engine.build_error = None
        engine.cancel_flag = False
        engine.current_path = ""
        engine.start_time = time.time()
        engine.total_images = len(files)
        engine.processed_images = 0
        engine.embedded_images = 0
    # =========================================================

    def _runner():
        try:
            engine.build_index_stream(
                frames_root=body.frames_root,
                batch_size=int(body.batch_size),
                save_embeddings_npy=bool(body.save_embeddings_npy),
            )
        except Exception as e:
            with engine._lock:
                engine.build_error = str(e)
            # đảm bảo hạ cờ nếu lỗi xảy ra sớm
            with engine._lock:
                engine.building = False
                engine.current_path = ""

    _build_thread = threading.Thread(target=_runner, daemon=True)
    _build_thread.start()

    # lúc này building đã True → client sẽ thấy đúng
    return {"status": "started", **engine.status()}

@app.get("/api/status")
def get_status():
    if engine is None:
        return {"engine_ready": False}
    return engine.status()

@app.post("/api/cancel")
def cancel_build():
    if engine is None:
        return {"engine_ready": False}
    engine.cancel()
    return {"status": "cancelling", **engine.status()}

@app.post("/api/load_index")
def load_index(index_dir: str = Form("index_blip2_ocr")):
    global engine
    if engine is None:
        engine = Blip2OCREngine(index_dir=index_dir)
    engine.index_dir = index_dir
    out = engine.load_index()
    return out

@app.post("/api/kis")
def search_kis(body: SearchBody):
    if engine is None:
        return JSONResponse({"error": "engine not initialized"}, status_code=400)
    res = engine.textual_kis(body.query, topk=body.topk)
    return res

@app.post("/api/qa")
def search_qa(body: SearchBody):
    if engine is None:
        return JSONResponse({"error": "engine not initialized"}, status_code=400)
    out = engine.qa(body.query, topk=body.topk)
    return out

@app.post("/api/trake")
def search_trake(body: SearchBody):
    if engine is None:
        return JSONResponse({"error": "engine not initialized"}, status_code=400)
    out = engine.trake(body.query, topk=body.topk)
    return out

def _read_txt_lines(file_content: bytes) -> List[str]:
    s = file_content.decode("utf-8").splitlines()
    return [line.strip() for line in s if line.strip()]

def _sanitize_filename(name: str, default_base: str) -> str:
    base = re.sub(r"[^A-Za-z0-9_\-]+", "_", (name or "")).strip("_")
    if not base:
        base = default_base
    return base + ".csv"

@app.post("/api/export_csv")
async def export_csv(
    task: str = Form(...),                  # "kis" | "qa" | "trake"
    query_file: UploadFile = File(...),
    output_name: Optional[str] = Form(None) # tên mong muốn (không đuôi)
):
    if engine is None:
        return JSONResponse({"error": "engine not initialized"}, status_code=400)
    data = await query_file.read()
    queries = _read_txt_lines(data)

    if task == "kis":
        lines = engine.export_csv_kis_lines(queries)
        default = "query-1-kis"
    elif task == "qa":
        lines = engine.export_csv_qa_lines(queries)
        default = "query-3-qa"
    elif task == "trake":
        lines = engine.export_csv_trake_lines(queries)
        default = "query-4-trake"
    else:
        return JSONResponse({"error": "invalid task"}, status_code=400)

    # Chọn tên file theo output_name (nếu có), sanitize
    if output_name:
        fname = _sanitize_filename(output_name, default)
    else:
        # fallback: nếu người dùng không nhập tên → đặt theo task mặc định
        fname = default + ".csv"

    path = os.path.join(engine.index_dir, fname)
    os.makedirs(engine.index_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")
    return FileResponse(path, media_type="text/csv", filename=fname)
@app.get("/api/image")
def get_image(path: str):
    # Cảnh báo: demo local. Nếu deploy, hãy ràng buộc path vào frames_root an toàn hơn.
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)

@app.get("/api/health")
def health():
    return {"status": "ok", "engine_ready": engine is not None}
