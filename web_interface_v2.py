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
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


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
# Cho phép đổi mức log bằng ENV: LOG_LEVEL=DEBUG/INFO/WARNING/ERROR
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))


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
    from transformers import Blip2Processor, Blip2ForConditionalGeneration,AutoProcessor
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

if _HAS_BLIP2 and _HAS_TORCH:
    def load_blip2_safely(model_name: str, device: str):
        use_cuda = torch.cuda.is_available() and (device == "cuda")
        dtype = torch.float16 if use_cuda else torch.float32

        # model lớn (xl/xxl/6.7b) -> cho phép accelerate chia GPU/CPU (device_map='auto')
        needs_auto = any(s in model_name for s in ["-xl", "-xxl", "6.7b"])

        kwargs = dict(dtype=dtype, low_cpu_mem_usage=True)
        if needs_auto:
            kwargs["device_map"] = "auto"   # KHÔNG gọi .to() sau khi load!
        else:
            kwargs["device_map"] = None

        processor = AutoProcessor.from_pretrained(model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(model_name, **kwargs)

        used_auto = bool(kwargs["device_map"] == "auto")
        if not used_auto:
            model = model.to("cuda" if use_cuda else "cpu")

        return processor, model.eval(), used_auto
else:
    logger.warning("[FATAL ERROR] BLIP-2 or torch not available, BLIP-2 functions will fail.")
    exit(1)


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
        blip2_model: str = "Salesforce/blip2-opt-2.7b",  # model hợp lệ
        text_embed_model: str = "intfloat/multilingual-e5-base",
        device: Optional[str] = None,
        ocr_langs: Optional[List[str]] = None,
        index_dir: str = "index_blip2_ocr",
        # objects
        objects_root: str = "objects",
        use_objects: bool = True,
        object_topk: int = 10,
        object_weight: float = 1.3,
        #option for language
        caption_lang: str = "vi",
        caption_prompt: Optional[str] = None,
        # optional
        load_blip2: bool = True,
        load_ocr: bool = True,
        hf_token: Optional[str] = None,
        text_batch_size: int = 512,
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

        # self.device = device or ("cuda" if _HAS_TORCH and torch.cuda.is_available() else "cpu")
        if device in ("cpu", "cuda"):
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

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
        self.text_batch_size = int(text_batch_size or 512)
        
        # ============================================

        # Objects
        self.objects_root = objects_root
        self.use_objects = bool(use_objects)
        self.object_topk = int(object_topk)
        self.object_weight = float(object_weight if object_weight is not None else 1.3)
        logger.info("[ENGINE INIT] device=%s | index_dir=%s", device or "auto", index_dir)
        logger.info("[ENGINE INIT] models: blip2=%s | text=%s | ocr=%s | use_objects=%s (topk=%d, w=%.2f)",
            blip2_model if load_blip2 else "OFF",
            text_embed_model,
            ("easyocr" if (load_ocr and _HAS_EASYOCR) else ("pytesseract" if (load_ocr and _HAS_PYTESSERACT) else "OFF")),
            bool(use_objects), int(object_topk), float(object_weight))

        # Text encoder (bắt buộc)
        self.text_encoder = SentenceTransformer(text_embed_model, device=self.device)
        logger.info("[ENGINE] Text encoder loaded: %s", text_embed_model)

        self.caption_lang = (caption_lang or "vi").lower()
        _default_prompts = {
            "vi": "Mô tả ngắn gọn bức ảnh bằng tiếng Việt, súc tích và trung lập.",
            "en": "Briefly describe the image in English, concise and neutral."
        }
        self.caption_prompt = caption_prompt or _default_prompts.get(self.caption_lang, _default_prompts[self.caption_lang])
        # OCR (tuỳ chọn)
        self.reader = None
        if load_ocr and _HAS_EASYOCR:
            try:
                self.reader = easyocr.Reader(self.ocr_langs, gpu=(self.device.startswith("cuda")))
            except Exception:
                self.reader = None
        elif load_ocr and _HAS_PYTESSERACT:
            self.reader = "pytesseract"  # marker để dùng pytesseract trong _ocr_text
        if self.reader and _HAS_EASYOCR:
            logger.info("[ENGINE] OCR: easyocr(langs=%s, gpu=%s)", self.ocr_langs, self.device.startswith("cuda"))
        elif self.reader == "pytesseract":
            logger.info("[ENGINE] OCR: pytesseract(vie+eng)")
        else:
            logger.info("[ENGINE] OCR: OFF")

        # BLIP-2 (tuỳ chọn)
        self.processor = None
        self.blip2 = None
        if load_blip2:
            t0 = time.time()
            logger.info("[ENGINE] Loading BLIP-2: %s ...", blip2_model)
            self.processor, self.blip2, self._used_device_map_auto = load_blip2_safely(blip2_model, self.device)
            logger.info("[ENGINE] BLIP-2 ready (device_map=%s, device=%s)", 
            "auto" if self._used_device_map_auto else "none", self.device)
            # self.processor = Blip2Processor.from_pretrained(blip2_model, token=hf_token)
            # self.blip2 = Blip2ForConditionalGeneration.from_pretrained(
            #     blip2_model,
            #     token=hf_token,
            #     torch_dtype=(torch.float16 if (_HAS_TORCH and torch.cuda.is_available()) else torch.float32)
            # ).to(self.device).eval()
            # Processor giữ nguyên (nhẹ, không tốn nhiều RAM)
            # self.processor = Blip2Processor.from_pretrained(blip2_model)

            # # ===== Robust loader cho BLIP-2, ưu tiên ít RAM =====
            # bnb_ok = False
            # try:
            #     # Thử quantization nếu bitsandbytes sẵn có (tiết kiệm RAM đáng kể)
            #     from transformers import BitsAndBytesConfig
            #     bnb_conf = BitsAndBytesConfig(
            #         # load_in_4bit=True,                      # thử 4-bit
            #         bnb_4bit_use_double_quant=True,
            #         bnb_4bit_compute_dtype=torch.float16 if self.device=="cuda" else torch.float32,
            #     )
            #     bnb_ok = True
            # except Exception:
            #     bnb_ok = False

            # try:
            #     if bnb_ok:
            #         # 4-bit quant + auto offload
            #         self.blip2 = Blip2ForConditionalGeneration.from_pretrained(
            #             blip2_model,
            #             # quantization_config=bnb_conf,
            #             device_map="auto",           # tự chia GPU/CPU
            #             low_cpu_mem_usage=True,
            #         )
            #     else:
            #         # Không có bitsandbytes: dùng auto-offload sang disk để giảm RAM
            #         offload_dir = os.path.join(self.index_dir, "_offload_blip2")
            #         os.makedirs(offload_dir, exist_ok=True)
            #         self.blip2 = Blip2ForConditionalGeneration.from_pretrained(
            #             blip2_model,
            #             device_map="auto",           # có GPU sẽ dùng, phần dư offload
            #             torch_dtype=(torch.float16 if self.device=="cuda" else torch.float32),
            #             low_cpu_mem_usage=True,      # nạp từng phần
            #             offload_folder=offload_dir,  # thư mục offload trên đĩa
            #         )
            # except OSError as e:
            #     # Bắt lỗi Windows 1455 (paging file)
            #     msg = str(e)
            #     if ("1455" in msg) or ("paging file" in msg.lower()):
            #         raise RuntimeError(
            #             "Windows error 1455: Hết Virtual Memory khi load BLIP-2. "
            #             "Hãy tăng Pagefile (System Properties → Advanced → Performance → Settings → Advanced → "
            #             "Virtual memory → Change → bỏ tick Automatically → Custom size → Initial/Maximum ≥ 32768 MB), "
            #             "hoặc cài bitsandbytes để load 4-bit (pip install bitsandbytes), "
            #             "hoặc đổi sang offload_folder ổ đĩa còn nhiều dung lượng."
            #         )
            #     else:
            #         raise
            # # ===== Hết phần robust loader =====
            resolved_device = "cuda" if torch.cuda.is_available() and _HAS_TORCH else "cpu"
            self.device = resolved_device

            # self.blip2 = Blip2ForConditionalGeneration.from_pretrained(
            #     blip2_model,
            #     # token=hf_token,
            #     # device_map="auto",
            #     device_map=None,
            #     low_cpu_mem_usage=True,
            #     torch_dtype=(torch.float16 if (_HAS_TORCH and torch.cuda.is_available()) else torch.float32)
            # ).to(self.device).eval()
            # self.blip2 = Blip2ForConditionalGeneration.from_pretrained(
            #     blip2_model,
            #     token=hf_token,
            #     dtype=(torch.float16 if resolved_device == "cuda" else torch.float32),
            #     low_cpu_mem_usage=True,        # dùng loader tiết kiệm RAM, KHÔNG offload
            #     device_map=None                # ép không dùng accelerate hooks/offload
            # )
            # self.blip2 = self.blip2.to(resolved_device).eval()
            try:
                import torch as _t
                _dtype = str(next(self.blip2.parameters()).dtype).replace("torch.", "")
            except Exception:
                _dtype = "unknown"
            logger.info("[ENGINE] BLIP-2 ready (dtype=%s, device=%s) in %.1fs", _dtype, self.device, time.time()-t0)
        else:
            logger.info("[ENGINE] BLIP-2: OFF")

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
    def _blip2_caption(self, img, max_new_tokens: int = 40) -> str:
        """
        img có thể là path, PIL.Image hoặc np.ndarray
        Trả về caption (str). Thất bại -> "".
        """
        try:
            import numpy as np
            from PIL import Image as PILImage

            # Chuẩn hóa ảnh
            if isinstance(img, str):
                pil = PILImage.open(img).convert("RGB")
            elif isinstance(img, PILImage.Image):
                pil = img.convert("RGB")
            elif isinstance(img, np.ndarray):
                pil = PILImage.fromarray(img)
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

            # ⚠️ QUAN TRỌNG: ép về NumPy trước khi đưa vào processor để tránh .read()
            arr = np.asarray(pil)

            inputs = self.processor(images=arr, text="", return_tensors="pt")
            # đẩy tensor lên đúng device nếu có .to
            for k, v in list(inputs.items()):
                if hasattr(v, "to"):
                    inputs[k] = v.to(self.device)

            out_ids = self.blip2.generate(**inputs, max_new_tokens=max_new_tokens)
            cap = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
            return cap
        except Exception as e:
            logger.debug("[BLIP-2 CAPTION ERROR] Caption error on %s: %s", getattr(self, "current_path", "?"), e)
            return ""



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
    # def _embed_texts(self, texts: List[str]) -> np.ndarray:
    #     texts_proc = [f"passage: {t}" for t in texts]
    #     embs = self.text_encoder.encode(
    #         texts_proc,
    #         batch_size=self.text_batch_size,
    #         show_progress_bar=False,
    #         convert_to_numpy=True,
    #         normalize_embeddings=True
    #     )
    #     return embs.astype("float32")
    def _safe_caption(self, pil_img, path):
        try:
            return (self._blip2_caption(pil_img) or "").strip()
        except TypeError:
            try:
                return (self._blip2_caption(path) or "").strip()
            except Exception:
                logger.debug("[BLIP-2 CAPTION ERROR, _safe_caption] Caption error on %s: TypeError fallback also failed", path)
                return ""
        except Exception as e:
            logger.debug("[BLIP-2 CAPTION ERROR, _safe_caption] Caption error on %s: %s", path, e)
            return ""

    def _safe_ocr(self, pil_img):
        try:
            txt = self._ocr_text(pil_img)
            return (txt or "").strip()
        except Exception as e:
            logger.debug("[OCR ERROR, _safe_ocr] OCR error on %s as exception %s", getattr(self, "current_path", "?"), e)
            return ""

    def _make_main_texts(self, captions, ocrs):
        out = []
        for cap, ocr in zip(captions, ocrs):
            # if cap and ocr:
            #     out.append(f"{cap} [SEP] {ocr}")
            # else:
            #     out.append(cap or ocr or "")
            appended = ""
            if cap and ocr:
                appended = f"{cap} [SEP] {ocr}"
            else:
                appended = cap or ocr or ""
            out.append(appended)
            logger.debug("[_make_main_text]Caption: %s | OCR: %s => Main text: %s", cap, ocr, appended)
        return out

    def _embed_texts(self, texts):
        if not texts:
            dim = self.text_encoder.get_sentence_embedding_dimension()
            return np.zeros((0, dim), dtype="float32")
        prefixed = [f"passage: {t}" for t in texts]   # đúng chuẩn E5
        embs = self.text_encoder.encode(
            prefixed,
            batch_size=self.text_batch_size,          # dùng batch lớn
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,                # chuẩn hoá cho cosine
        ).astype("float32")
        return embs

    def _fuse_main_and_obj(self, main_emb, obj_emb=None):
        if obj_emb is None:
            return main_emb
        w = float(self.object_weight)
        fused = main_emb + (w * obj_emb)
        norms = np.linalg.norm(fused, axis=1, keepdims=True) + 1e-12
        return (fused / norms).astype("float32")      # L2-normalize sau khi cộng

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

        logger.info("[BUILD] Start indexing with BLIP-2 + OCR | total=%d | dim=%d | batch=%d | use_objects=%s(topk=%d, w=%.2f)",
            self.total_images, dim, batch_size, getattr(self, "use_objects", False),
            int(getattr(self, "object_topk", 10)), float(getattr(self, "object_weight", 1.3)))

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

        # ===== Vòng lặp THEO LÔ ẢNH =====
        for start in range(0, len(files), batch_size):
            if self.cancel_flag:
                logger.warning("Cancel flag detected. Stopping build loop.")
                break

            batch_paths = files[start:start + batch_size]

            captions, ocrs, obj_texts = [], [], []
            videos, frames = [], []
            opened_imgs = []

            # ---- B1: Đọc & trích xuất từng ảnh (caption/ocr/objects) ----
            for bi, p in enumerate(batch_paths):
                gi = start + bi  # global index
                with self._lock:
                    self.current_path = p
                if (gi % log_every) == 0:
                    logger.info("[Build %d/%d] %s", gi + 1, self.total_images, p)

                img = None
                try:
                    img = Image.open(p).convert("RGB")
                except Exception as e:
                    logger.debug("[BUILD INDEX STREAM ERROR]Open image error on %s: %s", p, e)

                # Caption
                try:
                    cap = self._blip2_caption(img if img is not None else p, max_new_tokens=40) or ""
                except Exception as e:
                    logger.debug("[BUILD INDEX STREAM ERROR]Caption error on %s: %s", p, e)
                    cap = ""

                # OCR
                try:
                    ocr = self._ocr_text(img) if img is not None else ""
                except Exception as e:
                    logger.debug("[BUILD INDEX STREAM ERROR]OCR error on %s: %s", p, e)
                    ocr = ""

                # Objects
                objs = read_objects(p)
                obj_text = ", ".join(objs) if objs else ""

                # Lưu lại
                captions.append(cap.strip())
                ocrs.append(ocr.strip())
                obj_texts.append(obj_text.strip())
                v, fr = extract_video_and_frame(p)
                videos.append(v)
                frames.append(fr)
                opened_imgs.append(img)  # giữ tham chiếu để GC sau batch

            # ---- B2: Ghép main_texts theo embed_dir.py ----
            main_texts = []
            for cap, ocr in zip(captions, ocrs):
                if cap and ocr:
                    main_texts.append(f"{cap} [SEP] {ocr}")
                else:
                    main_texts.append((cap or ocr or "").strip())

            # Đảm bảo không có chuỗi rỗng (tránh edge-case encoder)
            main_texts = [t if t.strip() else " " for t in main_texts]

            # ---- B3: Encode THEO LÔ LỚN cho main_texts ----
            main_emb = self._embed_texts(main_texts)  # (B, d) float32, normalized

            # Nếu vì lý do gì đó trả về rỗng: fallback encode toàn batch bằng " "
            if main_emb is None or getattr(main_emb, "shape", (0, 0))[0] == 0:
                logger.warning("[EMB FALLBACK] empty batch embeddings -> re-encode blanks")
                main_emb = self._embed_texts([" "] * len(batch_paths))

            # ---- B4: Encode objects nếu có VÀ fuse + L2-normalize ----
            fused = main_emb
            if any(len(t) > 0 for t in obj_texts):
                obj_texts_safe = [t if t.strip() else " " for t in obj_texts]
                obj_emb = self._embed_texts(obj_texts_safe)  # normalized
                w = float(getattr(self, "object_weight", 1.3))
                fused = main_emb + (w * obj_emb)
                fused = fused / (np.linalg.norm(fused, axis=1, keepdims=True) + 1e-12)
                fused = fused.astype("float32")

            # ---- B5: Add cả LÔ vào FAISS + ghi memmap (nếu bật) ----
            self.faiss_index.add(fused)
            if mmap is not None:
                mmap[start:start + fused.shape[0], :] = fused

            # ---- B6: Cập nhật meta/ids & tiến độ ----
            for bi, p in enumerate(batch_paths):
                self.meta.append({
                    "path": p,
                    "video": videos[bi],
                    "frame": int(frames[bi]),
                    "caption": captions[bi],
                    "ocr": ocrs[bi],
                    "objects": [s.strip() for s in obj_texts[bi].split(",")] if obj_texts[bi] else []
                })
                self.ids.append(p)

            with self._lock:
                self.embedded_images += fused.shape[0]
                self.processed_images += len(batch_paths)

            # Giải phóng ảnh đã mở
            for img in opened_imgs:
                try:
                    if hasattr(img, "close"):
                        img.close()
                except Exception:
                    pass

            if self.cancel_flag:
                logger.warning("Cancel flag detected after batch; breaking.")
                break

        # ---- B7: Lưu index + meta ----
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



    # def load_index(self) -> Dict[str, Any]:
    #     idx_path = os.path.join(self.index_dir, "faiss.index")
    #     meta_path = os.path.join(self.index_dir, "meta.json")
    #     if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
    #         raise RuntimeError(f"Index not found in {self.index_dir}. Build index first.")
    #     self.faiss_index = faiss.read_index(idx_path)
    #     with open(meta_path, "r", encoding="utf-8") as f:
    #         self.meta = json.load(f)
    #     self.ids = [m["path"] for m in self.meta]
    #     return {"num_images": len(self.meta), "index_dir": self.index_dir}
    def load_index(self) -> Dict[str, Any]:
        idx_path = os.path.join(self.index_dir, "faiss.index")
        meta_path = os.path.join(self.index_dir, "meta.json")
        logger.info("[LOAD] index_dir=%s\n        faiss=%s\n        meta=%s", self.index_dir, idx_path, meta_path)
        if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
            raise RuntimeError(f"Index not found in {self.index_dir}. Build index first.")
        self.faiss_index = faiss.read_index(idx_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.ids = [m["path"] for m in self.meta]
        ntotal = int(self.faiss_index.ntotal)
        logger.info("[LOAD DONE] num_images(meta)=%d | faiss.ntotal=%d", len(self.meta), ntotal)
        if ntotal != len(self.meta):
            logger.warning("[LOAD WARN] ntotal != meta size (FAISS=%d vs META=%d)", ntotal, len(self.meta))
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
    text_batch_size: int = 512

class SearchBody(BaseModel):
    query: str
    topk: int = 10

# @app.post("/api/initialize")
# def initialize(body: InitBody):
#     global engine
#     engine = Blip2OCREngine(
#         blip2_model=body.blip2_model,
#         text_embed_model=body.text_embed_model,
#         index_dir=body.index_dir,
#         objects_root=body.objects_root,
#         use_objects=body.use_objects,
#         object_topk=body.object_topk,
#         object_weight=body.object_weight
#     )
#     return {"status": "ok", "device": engine.device, "index_dir": body.index_dir}
@app.post("/api/initialize")
def initialize(body: InitBody):
    global engine
    logger.info("[INIT] blip2=%s | text=%s | index_dir=%s | use_objects=%s(topk=%d, w=%.2f)",
                body.blip2_model, body.text_embed_model, body.index_dir,
                body.use_objects, body.object_topk, body.object_weight)
    t0 = time.time()
    try:
        engine = Blip2OCREngine(
            blip2_model=body.blip2_model,
            text_embed_model=body.text_embed_model,
            index_dir=body.index_dir,
            objects_root=body.objects_root,
            use_objects=body.use_objects,
            object_topk=body.object_topk,
            object_weight=body.object_weight
        )
        logger.info("[INIT DONE] device=%s in %.1fs", engine.device, time.time()-t0)
        return {"status": "ok", "device": engine.device, "index_dir": body.index_dir}
    except Exception as e:
        logger.exception("[INIT ERROR] %s", e)
        # trả 500 có message thay vì stacktrace thô
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Initialize failed: {e}")
    



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
            logger.info("[LOAD REQ] faiss_path=%s | metadata_path=%s | index_dir=%s", faiss_path, metadata_path, index_dir)
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

# @app.post("/api/build_index")
# def build_index(body: InitBody):
#     logger.info("[BUILD REQ] frames_root=%s | batch=%s | save_npy=%s",
#             body.frames_root, body.batch_size, body.save_embeddings_npy)

#     global engine, _build_thread

#     # Build cần BLIP-2 đã khởi tạo
#     if engine is None:
#         engine = Blip2OCREngine(
#             blip2_model=body.blip2_model,
#             text_embed_model=body.text_embed_model,
#             index_dir=body.index_dir,
#             objects_root=body.objects_root,
#             use_objects=body.use_objects,
#             object_topk=body.object_topk,
#             object_weight=body.object_weight,
#         )

#     # Dọn thread cũ (nếu đã kết thúc)
#     if _build_thread and not _build_thread.is_alive():
#         _build_thread = None
#         engine.building = False

#     # Đang build thật sự?
#     if engine.building or (_build_thread and _build_thread.is_alive()):
#         return {"status": "already_building", **engine.status()}

#     # Validate frames_root
#     if not os.path.isdir(body.frames_root):
#         raise HTTPException(status_code=400, detail=f"frames_root không tồn tại: {body.frames_root}")

#     files = engine._gather_frames(body.frames_root)
#     if len(files) == 0:
#         raise HTTPException(status_code=400, detail="Không tìm thấy ảnh (.jpg/.jpeg/.png/.bmp) trong frames_root")

#     # ===== BẬT CỜ & SET COUNTERS TRƯỚC KHI START THREAD =====
#     with engine._lock:
#         engine.building = True
#         engine.build_error = None
#         engine.cancel_flag = False
#         engine.current_path = ""
#         engine.start_time = time.time()
#         engine.total_images = len(files)
#         engine.processed_images = 0
#         engine.embedded_images = 0
#     # =========================================================

#     def _runner():
#         try:
#             engine.build_index_stream(
#                 frames_root=body.frames_root,
#                 batch_size=int(body.batch_size),
#                 save_embeddings_npy=bool(body.save_embeddings_npy),
#             )
#         except Exception as e:
#             logger.exception("[BUILD ERROR] %s", e)
#             with engine._lock:
#                 engine.build_error = str(e)
#             # đảm bảo hạ cờ nếu lỗi xảy ra sớm
#             with engine._lock:
#                 engine.building = False
#                 engine.current_path = ""

#     _build_thread = threading.Thread(target=_runner, daemon=True)
#     _build_thread.start()

#     logger.info("[BUILD STARTED] total=%d | index_dir=%s", engine.total_images, engine.index_dir)
#     # lúc này building đã True → client sẽ thấy đúng
#     return {"status": "started", **engine.status()}
@app.post("/api/build_index")
def build_index(body: InitBody):
    logger.info("[BUILD REQ] frames_root=%s | batch=%s | text_batch=%s | save_npy=%s",
                body.frames_root, body.batch_size, getattr(body, "text_batch_size", None), body.save_embeddings_npy)

    global engine, _build_thread

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
        # Nếu __init__ không nhận text_batch_size, gán trực tiếp:
        if hasattr(body, "text_batch_size") and body.text_batch_size:
            engine.text_batch_size = int(body.text_batch_size)
    else:
        # (TÙY CHỌN) recreate nếu model đổi
        # need_recreate = ...
        # if need_recreate: engine = Blip2OCREngine(...)

        # Cập nhật cấu hình động
        if body.index_dir and body.index_dir != engine.index_dir:
            engine.index_dir = body.index_dir
            engine.faiss_index = None
            engine.meta, engine.ids = [], []
        if hasattr(body, "objects_root") and body.objects_root:
            engine.objects_root = body.objects_root
        if hasattr(body, "use_objects"):
            engine.use_objects = bool(body.use_objects)
        if hasattr(body, "object_topk") and body.object_topk is not None:
            engine.object_topk = int(body.object_topk)
        if hasattr(body, "object_weight") and body.object_weight is not None:
            engine.object_weight = float(body.object_weight)
        if hasattr(body, "text_batch_size") and body.text_batch_size:
            engine.text_batch_size = int(body.text_batch_size)

    if _build_thread and not _build_thread.is_alive():
        _build_thread = None
        engine.building = False

    if engine.building or (_build_thread and _build_thread.is_alive()):
        return {"status": "already_building", **engine.status()}

    if not os.path.isdir(body.frames_root):
        raise HTTPException(status_code=400, detail=f"frames_root không tồn tại: {body.frames_root}")

    files = engine._gather_frames(body.frames_root)
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="Không tìm thấy ảnh (.jpg/.jpeg/.png/.bmp) trong frames_root")

    with engine._lock:
        engine.building = True
        engine.build_error = None
        engine.cancel_flag = False
        engine.current_path = ""
        engine.start_time = time.time()
        engine.total_images = len(files)
        engine.processed_images = 0
        engine.embedded_images = 0

    def _runner():
        try:
            engine.build_index_stream(
                frames_root=body.frames_root,
                batch_size=int(body.batch_size),
                save_embeddings_npy=bool(body.save_embeddings_npy),
            )
        except Exception as e:
            logger.exception("[BUILD ERROR] %s", e)
            with engine._lock:
                engine.build_error = str(e)
                engine.building = False
                engine.current_path = ""

    _build_thread = threading.Thread(target=_runner, daemon=True)
    _build_thread.start()

    logger.info("[BUILD STARTED] total=%d | index_dir=%s", engine.total_images, engine.index_dir)
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

# @app.post("/api/load_index")
# def load_index(index_dir: str = Form("index_blip2_ocr")):
#     global engine
#     if engine is None:
#         engine = Blip2OCREngine(index_dir=index_dir)
#     engine.index_dir = index_dir
#     out = engine.load_index()
#     return out

@app.post("/api/kis")
def search_kis(body: SearchBody):
    logger.info("[KIS] q=%s | topk=%d", body.query, body.topk)
    if engine is None:
        return JSONResponse({"error": "engine not initialized"}, status_code=400)
    res = engine.textual_kis(body.query, topk=body.topk)
    return res

@app.post("/api/qa")
def search_qa(body: SearchBody):
    logger.info("[QA] q=%s | topk=%d", body.query, body.topk)
    if engine is None:
        return JSONResponse({"error": "engine not initialized"}, status_code=400)
    out = engine.qa(body.query, topk=body.topk)
    return out

@app.post("/api/trake")
def search_trake(body: SearchBody):
    logger.info("[TRAKE] q=%s | topk=%d", body.query, body.topk)
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
