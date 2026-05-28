import torch
from ..models.cnn import CNN

def get_model(pkl_path: str) -> CNN:
    return torch.load(pkl_path, map_location="cpu", weights_only=False)