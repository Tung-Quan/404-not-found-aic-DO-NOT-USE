import torch
print(torch.__version__)
print(torch.version.cuda)           # nên ra "12.1"
print(torch.cuda.is_available())    # True
print(torch.cuda.get_device_name(0))# NVIDIA GeForce RTX 3060
