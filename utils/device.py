# utils/device.py
import torch

def get_device():
    """
    Detect GPU (CUDA) nếu có, ngược lại fallback sang CPU
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[INFO] Using GPU: {gpu_name}")
        return "cuda"
    else:
        print("[INFO] No GPU detected. Using CPU.")
        return "cpu"