import torch

from ..models.cnn import CNN


def get_model(pkl_path: str) -> CNN:
    """
    Loads torch model from pickle path.

    Args:
        pkl_path (str): Path to model.

    Returns:
        CNN: Torch CNN model.
    """
    return torch.load(pkl_path, map_location="cpu", weights_only=False)
