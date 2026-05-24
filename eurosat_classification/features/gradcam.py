from .retrieve_model import get_model
from ..models.cnn import CNN
import torch.nn as nn
from torch import Tensor
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np


MODEL_PATH = "models/model1.pkl"

model = get_model(MODEL_PATH)

def get_last_conv_layer(model: CNN) -> list[nn.Conv2d]:
    last_layer = None
    
    for layer in model.backbone:
        if isinstance(layer, nn.Conv2d):
            last_layer = layer

    if last_layer is None:
        raise ValueError("No last layer found")

    return [last_layer]

def visualize_gradcam_rgb(model: CNN, input_tensor: Tensor, input_rgb_image: np.ndarray):
    """
    Args:
        model (CNN): CNN model used for prediction
        input_tensor (Tensor): (1, 3, 64, 64) shaped tensor of input image
        input_rgb_image (np.ndarray): (64, 64, 3) array of input image, normalized to [0, 1]
    """
    # Note: We can also make the target layer be all the layers, in which case they are aggregated
    with GradCAM(model=model, target_layers=get_last_conv_layer(model)) as cam:
        # targets as none defaults to highest scoring category (per batch)
        grayscale_cam = cam(input_tensor=input_tensor,targets=None)
        
        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        
        return show_cam_on_image(input_rgb_image, grayscale_cam, use_rgb=True)

def visualize_gradcam_ms(model: CNN, input_tensor: Tensor):
    # Select RGB bands for GradCam visualization -> (3, 64, 64)
    rgb_bands = input_tensor[[3, 2, 1], :, :]
    # Reorder dimensions -> (64, 64, 3)
    rgb_bands = rgb_bands.transpose(1, 2, 0)
    # normalize to [0, 1]
    # TODO: maybe we prefer using the normalization from the preprocessing, but this is just for visualization purposes
    normalized_rgb = (rgb_bands - rgb_bands.min()) / (rgb_bands.max() - rgb_bands.min())

    visualize_gradcam_rgb(model, input_tensor=input_tensor, input_rgb_image=normalized_rgb)
