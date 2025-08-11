import torch, numpy as np
from transformers import AutoProcessor, AutoModel
MODEL_ID = 'google/siglip-base-patch16-256-multilingual'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()

def text_embed(text: str) -> np.ndarray:
    with torch.no_grad():
        ins = processor(text=[text], return_tensors='pt').to(DEVICE)
        out = model.get_text_features(**ins)
        out = torch.nn.functional.normalize(out, dim=-1)
        return out[0].detach().cpu().numpy().astype('float32')