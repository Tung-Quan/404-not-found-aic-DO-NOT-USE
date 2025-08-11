import os, numpy as np, pandas as pd, torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

MODEL_ID = 'openai/clip-vit-base-patch32'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
BATCH = int(os.getenv('BATCH', '64'))
DIM = 512

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
meta = pd.read_parquet('index/meta.parquet')
N = len(meta)
os.makedirs('index/embeddings', exist_ok=True)
mem_path = 'index/embeddings/frames.f16.mmap'
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
    images.clear(); idx_buf.clear()

for i, row in tqdm(meta.iterrows(), total=N):
    img = Image.open(row.frame_path).convert('RGB')
    images.append(img)
    idx_buf.append(i)
    if len(images) >= BATCH:
        flush()
flush()
mem.flush()
print('Wrote embeddings to', mem_path)