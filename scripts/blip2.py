import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
import faiss
import tensorflow_hub as hub
import tensorflow as tf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 16

# 1. Load BLIP-2
model, vis_processors, _ = load_model_and_preprocess(
    "blip2_feature_extractor", "pretrain", device=DEVICE
)
model.eval()

# 2. Load TF-Hub model for color/texture embeddings
tf_model = hub.load("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5")

def tfhub_embed(image: Image.Image):
    img = image.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    feat = tf_model(img)
    feat = tf.linalg.l2_normalize(feat, axis=1)
    return feat.numpy()[0]

# 3. Load frames
frame_dir = "map-keyframes-aic25-b1"
frame_files = [os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.lower().endswith(".jpg")]
N = len(frame_files)
print(f"Tổng frame: {N}")

# 4. Load object + OCR
with open("objects-aic25-b1/objects.json", "r", encoding="utf-8") as f:
    obj_info = json.load(f)

# 5. Encode frames
BLIP_DIM = 768
TF_DIM = 1280  # Mobilenet_v2 output dim
DIM = BLIP_DIM + BLIP_DIM + TF_DIM  # visual + text + TFHub
embeddings = np.zeros((N, DIM), dtype=np.float32)

with torch.no_grad():
    for i in tqdm(range(0, N, BATCH), desc="Encoding frames"):
        batch_files = frame_files[i:i+BATCH]
        images = [Image.open(f).convert("RGB") for f in batch_files]

        # BLIP-2 visual
        processed = torch.stack([vis_processors["eval"](img) for img in images]).to(DEVICE)
        visual_feats = model.encode_image(processed)
        visual_feats = torch.nn.functional.normalize(visual_feats, dim=-1)

        # OCR + object
        text_feats = []
        for f in batch_files:
            key = os.path.basename(f)
            txt = obj_info.get(key, {}).get("text", "")
            txt_feat = model.encode_text([txt]) if txt else torch.zeros(1, BLIP_DIM).to(DEVICE)
            txt_feat = torch.nn.functional.normalize(txt_feat, dim=-1)
            text_feats.append(txt_feat)
        text_feats = torch.cat(text_feats, dim=0)

        # TF-Hub color/texture
        tf_feats = np.array([tfhub_embed(img) for img in images], dtype=np.float32)

        # Concatenate all embeddings
        combined = torch.cat([visual_feats, text_feats], dim=1).cpu().numpy()
        combined = np.concatenate([combined, tf_feats], axis=1)

        embeddings[i:i+len(batch_files)] = combined

# 6. Build FAISS index
index = faiss.IndexFlatIP(DIM)
index.add(embeddings)
print("✅ FAISS index built")

# 7. Search query
def search(query: str, topk=5):
    with torch.no_grad():
        text_feat = model.encode_text([query])
        text_feat = torch.nn.functional.normalize(text_feat, dim=-1).cpu().numpy()
    # expand text_feat to match full dim
    tf_dummy = np.zeros((1, TF_DIM), dtype=np.float32)
    combined_query = np.concatenate([text_feat, text_feat, tf_dummy], axis=1)
    D, I = index.search(combined_query, topk)
    return [(frame_files[i], float(D[0][idx])) for idx, i in enumerate(I[0])]

# 8. Example
query = "bé chúc mừng sinh nhật"
results = search(query)
for path, score in results:
    print(f"{score:.4f}  {path}")
