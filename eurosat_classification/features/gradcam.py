from .retrieve_model import get_model
from ..models.cnn import CNN
import torch.nn as nn


MODEL_PATH = "models/model1.pkl"

model = get_model(MODEL_PATH)

def get_last_conv_layer(model: CNN):
    last_layer = None
    
    for layer in model.backbone:
        if isinstance(layer, nn.Conv2d):
            last_layer = layer

    return last_layer

