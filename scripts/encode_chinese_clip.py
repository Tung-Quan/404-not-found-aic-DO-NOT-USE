import os, sys, time, json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from sys import stdout
from transformers import AutoModel, AutoProcessor

# ------------------ Config ------------------
MODEL_ID   = os.environ.get("ZH_CLIP_MODEL_ID", "OFA-Sys/chinese-clip-vit-base-patch16")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE      = torch.float16 if DEVICE == "cuda" else torch.float32
BATCH      = int(os.environ.get("ZH_CLIP_BATCH", "64"))
FRAMES_DIR = Path(os.environ.get("FRAMES_DIR", "frames"))
OUT_DIR    = Path(os.environ.get("EMB_DIR", "index/embeddings"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMB_PATH   = OUT_DIR / "frames_chinese_clip.f16.mmap"
MANIFEST   = OUT_DIR / "frames_chinese_clip.manifest.json"

print("🇨🇳 CHINESE-CLIP ENCODING")
print("=" * 60)
print(f"🎯 Model: {MODEL_ID}")
print(f"🖥️  Device: {DEVICE}")
print(f"📦 Batch Size: {BATCH}")
print(f"🎨 Data Type: {DTYPE}\n")

# ------------------ Helpers ------------------
def iter_images(root: Path):
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in root.rglob(ext):
            yield p

def norm_path(p: Path) -> str:
    try:
        pp = p if p.is_absolute() else (Path.cwd() / p).resolve()
        return str(pp).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")

def load_chinese_clip():
    """
    Prefer safetensors. If repo lacks safetensors, user must upgrade torch>=2.6
    or use a model that provides safetensors.
    """
    try:
        model = AutoModel.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            local_files_only=False,
        ).to(DEVICE).eval()
        processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
        return model, processor
    except Exception as e:
        raise RuntimeError(
            f"{MODEL_ID} lacks safetensors or was blocked by safety checks. "
            f"Upgrade torch>=2.6 or switch to a safetensors variant. Root error: {e}"
        )

def infer_dim(model, processor):
    dim = int(getattr(getattr(model, "config", object()), "projection_dim", 0) or 0)
    if dim > 0:
        return dim
    tmp = Image.new("RGB", (224, 224), color=(128, 128, 128))
    with torch.inference_mode():
        ins = processor(images=[tmp], return_tensors="pt")
        ins = {k: v.to(DEVICE) for k, v in ins.items()}
        if hasattr(model, "get_image_features"):
            out = model.get_image_features(**ins)
        else:
            # fallback: vision forward + mean-pool
            outputs = model.vision_model(**ins)
            out = outputs.last_hidden_state.mean(dim=1)
        dim = int(out.shape[-1])
    return dim

def save_manifest(frame_paths, dim):
    data = {
        "model_id": MODEL_ID,
        "count": len(frame_paths),
        "dim": dim,
        "created_ts": time.time(),
        "frame_paths": [norm_path(Path(p)) for p in frame_paths],
        "dtype": "float16",
        "device": DEVICE,
    }
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ------------------ Main ------------------
def main():
    print("🔄 Loading Chinese-CLIP (safetensors preferred, use_fast=False)...")
    model, processor = load_chinese_clip()
    print("✅ Chinese-CLIP model loaded.\n")

    files = [str(p) for p in iter_images(FRAMES_DIR)]
    files.sort()
    N = len(files)
    print(f"🖼️ Frames: {N:,}")
    if N == 0:
        print(f"❌ No images found in {FRAMES_DIR}")
        sys.exit(1)

    print("🔍 Inferring embedding dimension...")
    DIM = infer_dim(model, processor)
    print(f"📏 Embedding Dimension (image): {DIM}\n")

    print(f"💾 Output (memmap): {EMB_PATH}")
    mem = np.memmap(EMB_PATH, dtype="float16", mode="w+", shape=(N, DIM))

    # Progress setup
    use_bar = stdout.isatty()
    try:
        from tqdm import tqdm
        bar = tqdm(total=N, desc="Encoding frames", unit="img",
                   dynamic_ncols=True, disable=not use_bar)
    except Exception:
        bar = None
        use_bar = False
    last_log = time.time()

    images, idx_buf = [], []

    def flush_progress():
        nonlocal last_log
        bsz = len(images)
        if bsz == 0:
            return
        s, e = idx_buf[0], idx_buf[-1] + 1
        with torch.inference_mode():
            ins = processor(images=images, return_tensors="pt")
            ins = {k: v.to(DEVICE) for k, v in ins.items()}
            if hasattr(model, "get_image_features"):
                feats = model.get_image_features(**ins)
            else:
                outputs = model.vision_model(**ins)
                feats = outputs.last_hidden_state.mean(dim=1)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            arr = feats.detach().to(torch.float16).cpu().numpy()
        if arr.shape != (e - s, DIM):
            raise RuntimeError(f"Shape mismatch: arr={arr.shape}, slice={(e - s, DIM)}")
        mem[s:e] = arr
        for im in images:
            try: im.close()
            except: pass
        images.clear()
        idx_buf.clear()

        if use_bar and bar is not None:
            bar.update(bsz)
        else:
            now = time.time()
            if now - last_log >= 2:
                print(f"Processed {e}/{N} ({e/N:.1%})", flush=True)
                last_log = now

    print(f"🔄 Encoding {N:,} frames with Chinese-CLIP...\n")
    for i, fp in enumerate(files):
        img = Image.open(fp).convert("RGB")
        images.append(img)
        idx_buf.append(i)
        if len(images) >= BATCH:
            flush_progress()
    flush_progress()
    if use_bar and bar is not None:
        bar.close()
    mem.flush()

    save_manifest(files, DIM)
    size_mb = os.path.getsize(EMB_PATH) / (1024 * 1024)
    print("\n🎉 DONE! Chinese-CLIP embeddings saved.")
    print(f"📁 {EMB_PATH}  |  shape=({N},{DIM}) float16  |  ~{size_mb:.1f} MB")
    print(f"📝 Manifest: {MANIFEST}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⛔ Stopped by user")
        sys.exit(130)
