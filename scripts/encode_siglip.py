import os, numpy as np, torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

MODEL_ID = 'google/siglip-base-patch16-256-multilingual'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
BATCH = int(os.getenv('BATCH', '64'))
DIM = 512

print("🌐 SIGLIP ENCODING - MULTILINGUAL")
print("=" * 60)
print(f"🎯 Model: {MODEL_ID}")
print(f"🖥️  Device: {DEVICE}")
print(f"📦 Batch Size: {BATCH}")
print(f"📏 Embedding Dimension: {DIM}")
print(f"🎨 Data Type: {DTYPE}")

print("\n🔄 Loading SigLIP model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
print("✅ SigLIP model loaded successfully!")

print("\n📋 Quét toàn bộ file .jpg trong frames bằng os.walk...")
frame_files = []
for root, dirs, files in os.walk('frames'):
    for file in files:
        if file.lower().endswith('.jpg'):
            frame_files.append(os.path.join(root, file))
N = len(frame_files)
print(f"📊 Tổng số frame .jpg cần encode: {N:,}")

os.makedirs('index/embeddings', exist_ok=True)
mem_path = 'index/embeddings/frames_siglip.f16.mmap'
print(f"💾 Output path: {mem_path}")
mem = np.memmap(mem_path, dtype='float16', mode='w+', shape=(N, DIM))

images, idx_buf = [], []

def flush():
    if not images:
        return
    with torch.no_grad():
        ins = processor(images=images, return_tensors='pt').to(DEVICE)
        for k in ins:
            ins[k] = ins[k].to(DTYPE)
        out = model.get_image_features(**ins)
        out = torch.nn.functional.normalize(out, dim=-1)
        arr = out.detach().cpu().to(torch.float16).numpy()
    s, e = idx_buf[0], idx_buf[-1] + 1
    mem[s:e] = arr
    images.clear()
    idx_buf.clear()

print(f"\n🔄 Đang encode {N:,} frames với SigLIP...")
print("⏱️  Quá trình này có thể lâu tuỳ vào cấu hình máy...")
for i, frame_path in tqdm(enumerate(frame_files), total=N, desc="Encoding frames"):
    img = Image.open(frame_path).convert('RGB')
    images.append(img)
    idx_buf.append(i)
    if len(images) >= BATCH:
        flush()
flush()
mem.flush()
print(f"\n🎉 SUCCESS! SigLIP embeddings saved!")
print(f"📁 Location: {mem_path}")
print(f"📊 Shape: {mem.shape}")
print(f"💾 Size: {os.path.getsize(mem_path) / 1024 / 1024:.1f} MB")
print(f"\n🎯 Next steps:")
print("1. Build FAISS index: python scripts/build_faiss.py")
print("2. Update API servers to use SigLIP")
print("3. Test performance improvements")
print(f"\n✨ Expected improvements:")
print("- 🌐 Multilingual support")
print("- 🚀 Fast inference")
print("- 🎯 Better quality scores")
