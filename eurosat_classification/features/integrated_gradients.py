"""
Compute and visualise Integrated Gradients for a trained EuroSAT CNN.

Integrated Gradients attribute a model's prediction
to its input features by accumulating gradients along a straight-line path from
a baseline (all-zeros) to the actual input.  A large positive attribution means
that pixel/channel pushed the model toward the explained class; a large negative
attribution means it pushed against it.

Usage:
    python -m eurosat_classification.features.integrated_gradients \
        --model_path models/model1.pkl \
        --input_file path/to/image.jpg     # 3-band RGB
        --input_file path/to/image.tif     # 13-band MS

Optional arguments:
    --target_class  Class index to explain (default: predicted class)
    --n_steps       Interpolation steps (default: 50, higher = more accurate)
    --output_path   Save figure to this path instead of displaying it
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tifffile
from PIL import Image
from torchvision import transforms

from ..data.band_names import MS_BAND_NAMES
from ..data.label_map import label_map
from ..data.preprocessors import normalize_MS_img

def load_rgb(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a 3-band JPG/PNG image.

    Returns
    -------
    raw          : [3, H, W] float tensor in [0, 1] – used for display
    preprocessed : same as raw (RGB training used no extra normalisation)
    baseline     : all-zeros tensor – represents a blank/dark image
    """
    preprocessed = transforms.ToTensor()(Image.open(path).convert("RGB"))
    baseline = torch.zeros_like(preprocessed) # Use zeros as baseline
    return preprocessed, preprocessed.clone(), baseline

def load_ms(path: str | Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a 13-band GeoTIFF image.

    Returns
    -------
    raw          : [13, H, W] raw digital-number values – used for display
    preprocessed : [13, H, W] clipped and z-score normalised – fed to the model
    baseline     : normalised all-zeros – represents a blank image in model space
    """
    raw = torch.from_numpy(tifffile.imread(str(path)).astype("float32")).permute(2, 0, 1)
    preprocessed = normalize_MS_img(raw)
    baseline = torch.zeros_like(preprocessed) # Use zeros as baseline
    return raw, preprocessed, baseline

def integrated_gradients(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    baseline: torch.Tensor,
    target_class: int,
    n_steps: int = 50,
) -> torch.Tensor:
    """Compute Integrated Gradients attributions.

    Interpolates `n_steps` images between `baseline` and `input_tensor`,
    runs them all through the model in one batch, and averages the gradients
    using the trapezoidal rule.  Multiplying by (input - baseline) gives the
    final attribution, which satisfies the *completeness* axiom:
        sum(attributions) ≈ model(input)[target] - model(baseline)[target]

    Parameters
    ----------
    model        : trained nn.Module (must be in eval mode)
    input_tensor : preprocessed input  [C, H, W]
    baseline     : reference input     [C, H, W]
    target_class : class index whose logit is differentiated
    n_steps      : number of interpolation steps

    Returns
    -------
    attributions : [C, H, W]  signed attribution per pixel and channel
    """
    alphas = torch.linspace(0, 1, n_steps + 1)                              # [n_steps+1]
    path = baseline + alphas.view(-1, 1, 1, 1) * (input_tensor - baseline)  # [n_steps+1, C, H, W]
    path = path.detach().requires_grad_(True) # This tells PyTorch to enable gradient tracking for the path

    model(path)[:, target_class].sum().backward() # Compute all gradients in one backward pass; path.grad will have shape [n_steps+1, C, H, W]

    # Trapezoidal rule: average of gradients at consecutive interpolation points
    avg_grads = ((path.grad[:-1] + path.grad[1:]) / 2).mean(dim=0)  # [C, H, W]

    return ((input_tensor - baseline) * avg_grads).detach()


def _aggregate_attribution(attrs: torch.Tensor) -> np.ndarray:
    """Sum absolute attributions across channels and normalise to [0, 1].

    This gives a single spatial heatmap showing *where* the model looked,
    regardless of which channel drove the attribution.
    """
    agg = attrs.abs().sum(dim=0).numpy()
    # Normalise to [0, 1] for display (add small epsilon to avoid division by zero)
    return (agg - agg.min()) / (agg.max() - agg.min() + 1e-8)


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

def visualise_rgb(
    raw: torch.Tensor,
    attrs: torch.Tensor,
    predicted_class: int,
    target_class: int,
    output_path: str | Path | None,
) -> None:
    """Plot a two-panel figure for RGB images.

    Left  – original image.
    Right – aggregate attribution overlaid in orange/red ("hot" colormap):
            brighter areas contributed most to the predicted class.
    """
    orig = raw.permute(1, 2, 0).numpy()
    agg  = _aggregate_attribution(attrs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(
        f"Predicted: {label_map[predicted_class]}   "
        f"Explained: {label_map[target_class]}"
    )

    ax1.imshow(orig)
    ax1.set_title("Original image")
    ax1.axis("off")

    # Overlay: the heatmap is semi-transparent so the original scene is still visible.
    # Bright spots mark pixels that most strongly drove the classification decision.
    ax2.imshow(orig)
    ax2.imshow(agg, cmap="hot", alpha=0.6)
    ax2.set_title("Attribution heatmap (overlaid)\nBrighter = more influential pixels")
    ax2.axis("off")

    plt.tight_layout()
    _save_or_show(fig, output_path)


def visualise_ms(
    raw: torch.Tensor,
    attrs: torch.Tensor,
    predicted_class: int,
    target_class: int,
    output_path: str | Path | None,
) -> None:
    """Plot a 3×5 grid of attribution heatmaps for 13-band MS images.

    Cell 0    – scaled-rgb colour composite for visual context.
    Cells 1-13 – per-band attribution maps using a red/blue diverging colormap:
                  red  = positive attribution (band pushed model toward the class),
                  blue = negative attribution (band pushed model away from the class).
    Cell 14   – aggregate across all 13 bands (bright = most influential location).
    """
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle(
        f"Predicted: {label_map[predicted_class]}   "
        f"Explained: {label_map[target_class]}"
    )
    flat = axes.flatten()

    flat[0].imshow(_scaled_rgb_colour(raw))
    flat[0].set_title("Scaled RGB\n(R=B4, G=B3, B=B2)", fontsize=8)
    flat[0].axis("off")

    for ch in range(13):
        a    = attrs[ch].numpy()
        vmax = max(abs(a.min()), abs(a.max())) or 1.0  # fall back to 1.0 if band has no attribution
        flat[ch + 1].imshow(a, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        flat[ch + 1].set_title(MS_BAND_NAMES[ch], fontsize=7)
        flat[ch + 1].axis("off")

    flat[14].imshow(_aggregate_attribution(attrs), cmap="hot")
    flat[14].set_title("Aggregate\n(sum |attr| all bands)", fontsize=8)
    flat[14].axis("off")

    plt.tight_layout()
    _save_or_show(fig, output_path)


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
        description="Integrated Gradients for EuroSAT CNN models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path",   required=True, help="Path to saved model (.pkl)")
    parser.add_argument("--input_file",   required=True, help=".jpg for RGB or .tif for MS")
    parser.add_argument("--target_class", type=int, default=None,
                        help="Class index to explain. Defaults to the predicted class.")
    parser.add_argument("--n_steps",      type=int, default=50)
    parser.add_argument("--output_path",  default=None)
    args = parser.parse_args()

    # Infer image type from file extension
    is_ms = Path(args.input_file).suffix.lower() in {".tif", ".tiff"}

    model = torch.load(args.model_path, map_location="cpu", weights_only=False)
    model.eval()

    if is_ms:
        raw, preprocessed, baseline = load_ms(args.input_file)
    else:
        raw, preprocessed, baseline = load_rgb(args.input_file)

    with torch.no_grad():
        predicted_class = int(model(preprocessed.unsqueeze(0)).argmax(1).item())

    target_class = args.target_class if args.target_class is not None else predicted_class
    print(f"Predicted : {label_map[predicted_class]}")
    print(f"Explaining: {label_map[target_class]}")

    attrs = integrated_gradients(model, preprocessed, baseline, target_class, args.n_steps)

    if is_ms:
        visualise_ms(raw, attrs, predicted_class, target_class, args.output_path)
    else:
        visualise_rgb(raw, attrs, predicted_class, target_class, args.output_path)


if __name__ == "__main__":
    main()
