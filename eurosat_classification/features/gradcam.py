from .retrieve_model import get_model
from ..models.cnn import CNN
import torch.nn as nn
from torch import Tensor
import tifffile
from ..data.preprocessors import normalize_MS_img
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from ..data.label_map import label_map
import argparse




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
    img_tensor = transforms.ToTensor()(Image.open(path).convert("RGB")).unsqueeze(0)
    img_array = img_tensor.permute(1, 2, 0).numpy()

    return img_tensor, img_array


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
    img_tensor = normalize_MS_img(raw)

    img_array = _scaled_rgb_colour(raw)

    return img_tensor.unsqueeze(0), img_array


def gradcam(model: CNN, input_tensor: Tensor, input_rgb_image: np.ndarray, target_class: int | None = None):
    with GradCAM(model=model, target_layers=get_last_conv_layer(model)) as cam:
        if target_class is not None:
            targets = [ClassifierOutputTarget(target_class)]
        else:
            # when targets is None it defaults to the highest predicted class
            targets = None 
    
        # Set model to evaluation mode before applying gradcam
        model.eval()

        # targets as none defaults to highest scoring category (per batch)
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        
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



def main():
    parser = argparse.ArgumentParser(
        description="GradCAM for EuroSAT CNN models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path",   required=True, help="Path to saved model (.pkl)")
    parser.add_argument("--input_file",   required=True, help=".jpg for RGB or .tif for MS")
    parser.add_argument("--target_class", type=int, default=None,
                        help="Class index to explain. Defaults to the predicted class.")
    parser.add_argument("--output_path",  default=None)
    args = parser.parse_args()

    # Infer image type from file extension
    is_ms = Path(args.input_file).suffix.lower() in {".tif", ".tiff"}

    # Load modle
    model = get_model(args.model_path)

    # Load image for gradcam
    if is_ms:
        img_tensor, img_array = load_ms(args.input_file)
    else:
        img_tensor, img_array = load_rgb(args.input_file)

    # Predict target class for explanation
    with torch.no_grad():
        predicted_class = int(model(img_tensor).argmax(1).item())
    target_class = args.target_class if args.target_class is not None else predicted_class

    # ensure target class is valid
    if target_class not in label_map:
        valid = ", ".join(f"{k} ({v})" for k, v in label_map.items())
        raise ValueError(f"Invalid target class {target_class}. Valid options are: {valid}")

    # beginin gradcam
    print(f"Predicted : {label_map[predicted_class]}")
    print(f"Explaining: {label_map[target_class]}")

    gradcam_visualization = gradcam(model, img_tensor, img_array, target_class)

    fig, ax = plt.subplots()
    ax.imshow(gradcam_visualization)
    ax.set_title(f"GradCAM | Predicted: {label_map[predicted_class]} | Explaining: {label_map[target_class]}")
    ax.axis("off")

    _save_or_show(fig, args.output_path)


if __name__ == "__main__":
    main()
