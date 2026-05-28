from .retrieve_model import get_model
from ..models.cnn import CNN
import torch.nn as nn
from torch import Tensor
import tifffile
from ..data.preprocessors import normalize_MS_img
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt





def get_last_conv_layer(model: CNN) -> list[nn.Conv2d]:
    last_layer = None
    
    for layer in model.backbone:
        if isinstance(layer, nn.Conv2d):
            last_layer = layer

    if last_layer is None:
        raise ValueError("No last layer found")

    return [last_layer]


def load_rgb(path: str | Path) -> tuple[torch.Tensor, np.ndarray]:
    """Load a 3-band JPG/PNG as a [1, 3, H, W] float tensor in [0, 1] and as np.ndarray.

    tensor used for gradcam
    np array used for visualization

    Returns (tensor, np array).
    """
    tensor = transforms.ToTensor()(Image.open(path).convert("RGB")).unsqueeze(0)
    array = tensor.permute(1, 2, 0).numpy()

    return tensor, array


def _scaled_rgb_colour(raw: torch.Tensor) -> np.ndarray:
    """Build a uint8 scaled-rgb colour composite from raw MS bands (R=B4, G=B3, B=B2).

    Bands B4/B3/B2 map to red/green/blue, giving a natural-looking landscape view
    similar to what the human eye would see from a satellite.
    """
    def scale(band):
        lo, hi = band.min(), band.max()
        return (band - lo) / (hi - lo + 1e-8) 
    # Stack the three bands into a single [H, W, 3] array and scale to [0, 255] uint8 for display
    return np.stack([scale(raw[i].numpy()) for i in (3, 2, 1)], axis=-1)


def load_ms(path: str | Path) -> tuple[torch.Tensor, np.ndarray]:
    """Load a 13-band GeoTIFF as a [13, H, W] float tensor in [0, 1] and as np.ndarray..

    tensor used for gradcam
    np array used for visualization

    preprocessing is clipping and z-score normalisation for the model
    """
    raw = torch.from_numpy(tifffile.imread(str(path)).astype("float32")).permute(2, 0, 1)
    tensor = normalize_MS_img(raw)

    array = _scaled_rgb_colour(tensor)

    return tensor, array


def gradcam(model: CNN, input_tensor: Tensor, input_rgb_image: np.ndarray, target_class: int | None = None):
    with GradCAM(model=model, target_layers=get_last_conv_layer(model)) as cam:
        # Set model to evaluation mode before applying gradcam
        model.eval()

        # targets as none defaults to highest scoring category (per batch)
        grayscale_cam = cam(input_tensor=input_tensor, targets=target_class)
        
        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        
        return show_cam_on_image(input_rgb_image, grayscale_cam, use_rgb=True)
    

def _save_or_show(fig: plt.Figure, output_path: str | Path | None) -> None:
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {Path(output_path).resolve()}")
    else:
        plt.show()
    plt.close(fig)