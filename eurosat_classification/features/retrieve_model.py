import torch

from ..models.cnn import CNN


def get_model(pkl_path: str) -> CNN:
<<<<<<< HEAD
    return torch.load(pkl_path, map_location="cpu", weights_only=False)
=======
    """
    Loads torch model from pickle path.

    Args:
        pkl_path (str): Path to model.

    Returns:
        CNN: Torch CNN model.
    """
    return torch.load(pkl_path, map_location="cpu", weights_only=False)
>>>>>>> 0795957fa8c1a387bce5b84759349c7bd9d4ce8b
