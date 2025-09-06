# check_macos.py
import os, sys, platform
import numpy as np

print("=== macOS ENV CHECK START ===")
print("Python:", sys.version.split()[0], "| Platform:", platform.platform(), "| Machine:", platform.machine())

# Torch + MPS
try:
    import torch
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    dev = "mps" if has_mps else ("cuda" if has_cuda else "cpu")
    print(f"[OK] torch {torch.__version__} | CUDA: {has_cuda} | MPS: {has_mps} | device={dev}")
except Exception as e:
    print("[ERR] torch import failed:", e); sys.exit(1)

# Transformers + BLIP-2 processor (nhẹ)
try:
    import transformers
    from transformers import Blip2Processor
    print("[OK] transformers:", transformers.__version__)
    _proc = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-base")
    print("[OK] BLIP-2 processor downloaded")
except Exception as e:
    print("[ERR] transformers/processor failed:", e); sys.exit(1)

# Sentence-Transformers (E5 encode test)
try:
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("intfloat/multilingual-e5-base")
    embs = sbert.encode(
        ["query: xin chào", "passage: Xin chào, đây là kiểm tra embedding."],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    print("[OK] sentence-transformers: shape", embs.shape, "norms", np.linalg.norm(embs, axis=1))
except Exception as e:
    print("[ERR] sentence-transformers failed:", e); sys.exit(1)

# FAISS CPU
try:
    import faiss
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    D, I = index.search(embs[:1], 2)
    print("[OK] faiss-cpu: search scores", [float(x) for x in D[0]], "idx", I[0].tolist())
except Exception as e:
    print("[ERR] faiss failed:", e); sys.exit(1)

# EasyOCR (nhanh gọn – không bắt buộc nếu bạn dùng pytesseract)
try:
    import easyocr
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (320, 100), "white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), "HELLO", fill="black")
    import numpy as _np
    reader = easyocr.Reader(['vi','en'], gpu=False)  # MPS không dùng ở đây; test CPU
    res = reader.readtext(_np.array(img))
    print("[OK] easyocr:", res[:1])
except Exception as e:
    print("[WARN] easyocr not fully tested:", e)

# (tuỳ chọn) Full BLIP-2 model test (nặng). Bật bằng env BLIP2_CHECK=1
if os.getenv("BLIP2_CHECK") == "1":
    try:
        from transformers import Blip2ForConditionalGeneration
        model_name = os.getenv("BLIP2_MODEL", "Salesforce/blip2-flan-t5-base")
        device = "mps" if (has_mps) else ("cuda" if has_cuda else "cpu")
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype).to(device)
        from PIL import Image
        img = Image.new("RGB", (224, 224), "gray")
        prompt = "Describe this image in Vietnamese."
        inputs = _proc(images=img, text=prompt, return_tensors="pt").to(device, dtype=dtype)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=10)
        txt = _proc.batch_decode(out, skip_special_tokens=True)[0]
        print("[OK] BLIP-2 generate:", txt)
    except Exception as e:
        print("[ERR] BLIP-2 model check failed:", e)
else:
    print("[INFO] Skip heavy BLIP-2 model check. Set BLIP2_CHECK=1 to run it.")

print("=== macOS ENV CHECK DONE ===")
