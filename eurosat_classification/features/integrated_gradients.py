"""
Compute and visualise Integrated Gradients for a trained EuroSAT CNN.

Integrated Gradients attribute a model's prediction
to its input features by accumulating gradients along a straight-line path from
a baseline (all-zeros) to the actual input.  A large positive attribution means
that pixel/channel pushed the model toward the explained class;
a large negative attribution means it pushed against it.

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
import tifffile
import torch
from PIL import Image
from torchvision import transforms

from ..data.band_names import MS_BAND_NAMES
from ..data.label_map import label_map
from ..data.preprocessors import normalize_MS_img
from .gradcam import _scaled_rgb_colour


def load_rgb_ig(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load a 3-band JPG/PNG as a [3, H, W] float tensor in [0, 1].

    RGB training used no normalisation, so the
    raw image is also the model input.
    Args:
        path (str | Path): Path to RGB image file.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Normalized RGB image and baseline.
    """
    image = transforms.ToTensor()(Image.open(path).convert("RGB"))
    baseline = torch.zeros_like(image)
    return image, baseline


def load_ms_ig(
    path: str | Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a 13-band GeoTIFF as a [13, H, W] float tensor.

    Returns (raw, preprocessed, baseline) where raw holds the original
    digital-number values for display, preprocessed is clipped and
    z-score normalised for the model, and baseline is all-zeros in
    normalised space.

    Args:
        path (str | Path): Path to MS image file.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Raw, preprocessed, and
                            baseline versions of MS image.
    """
    raw = torch.from_numpy(
        tifffile.imread(str(path)).astype("float32")
    ).permute(2, 0, 1)
    preprocessed = normalize_MS_img(raw)
    baseline = torch.zeros_like(preprocessed)
    return raw, preprocessed, baseline


def integrated_gradients(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    baseline: torch.Tensor,
    target_class: int,
    n_steps: int = 50,
) -> torch.Tensor:
    """
    Compute Integrated Gradients attributions for a single image.

    Interpolates n_steps images between baseline and input_tensor, runs them
    through the model in one batch, and averages the gradients using the
    trapezoidal rule. The result satisfies the completeness axiom:
        sum(attributions) ≈ model(input)[target] - model(baseline)[target]

    Args:
        model: Trained nn.Module in eval mode.
        input_tensor: Preprocessed input [C, H, W].
        baseline: Reference input [C, H, W], typically all-zeros.
        target_class: Class index whose logit is differentiated.
        n_steps: Number of interpolation steps (more = more accurate).

    Returns:
        torch.Tensor: Signed attribution
        tensor [C, H, W] per pixel and channel.
    """
    alphas = torch.linspace(
        0, 1, n_steps + 1, device=input_tensor.device
    )  # [n_steps+1]
    path = baseline + alphas.view(-1, 1, 1, 1) * (
        input_tensor - baseline
    )  # [n_steps+1, C, H, W]
    path = path.detach().requires_grad_(
        True
    )  # This tells PyTorch to enable gradient tracking for the path

    # Compute all gradients in one backward pass;
    # path.grad will have shape [n_steps+1, C, H, W]
    model(path)[:, target_class].sum().backward()

    # Trapezoidal rule: average of gradients
    # at consecutive interpolation points
    avg_grads = ((path.grad[:-1] + path.grad[1:]) / 2).mean(dim=0)  # [C, H, W]

    return ((input_tensor - baseline) * avg_grads).detach()


def _aggregate_attribution(attrs: torch.Tensor) -> np.ndarray:
    """
    Sum absolute attributions across channels and normalise to [0, 1].

    This gives a single spatial heatmap showing where the model looked,
    regardless of which channel drove the attribution.

    Args:
        attrs (torch.tensor): Attribution scores across channels.

    Returns:
        np.ndarray: Summed and normalized attribution scores across channels.
    """
    agg = attrs.abs().sum(dim=0).numpy()
    # Normalise to [0, 1] for display
    # (add small epsilon to avoid division by zero)
    return (agg - agg.min()) / (agg.max() - agg.min() + 1e-8)


def band_attribution_totals(
    model: torch.nn.Module,
    dataset,  # torch Dataset yielding (preprocessed_tensor [C, H, W], label)
    n_steps: int = 50,
    target_class: int | None = None,
    max_samples: int | None = None,
    device: str | torch.device = "cpu",
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """
    Compute class-balanced mean IG attributions per band over a dataset.

    For each image, computes Integrated Gradients and sums pixel-level
    attributions separately for positive (>0) and negative (<0) values
    per channel/band. Sums are accumulated per class, averaged within
    each class, then macro-averaged across classes so that no class
    dominates due to having more samples.

    Args:
        model: Trained nn.Module in eval mode.
        dataset: Dataset yielding (preprocessed_tensor [C, H, W], label) pairs.
        n_steps: IG interpolation steps per image.
        target_class: Class to explain. If None, uses each image's true label.
        max_samples: Cap the number of images processed. None = full dataset.
        device: Torch device for model and tensors.
        verbose: Print progress every 100 images.

    Returns:
        dict[str, np.ndarray]: Dictionary with keys:
            "positive"  - [C] ndarray, class-balanced mean positive attribution
                                                                per band.
            "negative"  - [C] ndarray, class-balanced mean negative attribution
                                                                per band (≤ 0).
            "count"     - number of images processed.
    """
    device = torch.device(device)
    model = model.to(device).eval()

    n_samples = (
        len(dataset) if max_samples is None else min(max_samples, len(dataset))
    )

    pos_by_class: dict[int, np.ndarray] = {}
    neg_by_class: dict[int, np.ndarray] = {}
    count_by_class: dict[int, int] = {}

    for i in range(n_samples):
        img, label = dataset[i]  # [C, H, W]

        if target_class is not None and int(label) != target_class:
            continue

        img = img.to(device)
        baseline = torch.zeros_like(img)

        tc = target_class if target_class is not None else int(label)

        attrs = integrated_gradients(
            model, img, baseline, tc, n_steps
        )  # [C, H, W]
        attrs_np = attrs.cpu().numpy()

        pos = attrs_np.clip(min=0).sum(axis=(1, 2))  # [C]
        neg = attrs_np.clip(max=0).sum(axis=(1, 2))  # [C]

        if tc not in pos_by_class:
            pos_by_class[tc] = pos
            neg_by_class[tc] = neg
            count_by_class[tc] = 1
        else:
            pos_by_class[tc] += pos
            neg_by_class[tc] += neg
            count_by_class[tc] += 1

        if verbose and (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{n_samples}] band attribution totals…")

    if not pos_by_class:
        return {"positive": np.array([]), "negative": np.array([]), "count": 0}

    classes = sorted(pos_by_class)
    pos_means = np.stack(
        [pos_by_class[c] / count_by_class[c] for c in classes]
    )  # [n_classes, C]
    neg_means = np.stack(
        [neg_by_class[c] / count_by_class[c] for c in classes]
    )  # [n_classes, C]

    return {
        "positive": pos_means.mean(axis=0),  # [C]
        "negative": neg_means.mean(axis=0),  # [C]
        "count": n_samples,
    }


def visualise_rgb(
    raw: torch.Tensor,
    attrs: torch.Tensor,
    predicted_class: int,
    target_class: int,
    output_path: str | Path | None,
) -> None | plt.Figure:
    """
    Plot a two-panel attribution figure for RGB images.

    Left panel shows the original image; right panel overlays the aggregate
    attribution heatmap (sum of absolute attributions across channels) on top
    of the original. Brighter areas drove the classification decision most.

    Args:
        raw: [3, H, W] float tensor in [0, 1].
        attrs: [3, H, W] attribution tensor from integrated_gradients().
        predicted_class: Class predicted by the model.
        target_class: Class being explained.
        output_path: Save figure here, or None to display interactively.

        Returns:
            None | plt.Figure: None if figure is saved,
            else returns figure itself.
    """
    orig = raw.permute(1, 2, 0).numpy()
    agg = _aggregate_attribution(attrs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(
        f"Predicted: {label_map[predicted_class]}   "
        f"Explained: {label_map[target_class]}"
    )

    ax1.imshow(orig)
    ax1.set_title("Original image")
    ax1.axis("off")

    ax2.imshow(orig)
    ax2.imshow(agg, cmap="hot", alpha=0.6)
    ax2.set_title(
        "Attribution heatmap (overlaid)\nBrighter = more influential pixels"
    )
    ax2.axis("off")

    plt.tight_layout()
    return _save_or_show(fig, output_path)


def visualise_ms(
    raw: torch.Tensor,
    attrs: torch.Tensor,
    predicted_class: int,
    target_class: int,
    output_path: str | Path | None,
) -> None | plt.Figure:
    """
    Plot a 3x5 attribution grid for 13-band MS images.

    Cell 0 shows a RGB composite (R=B4, G=B3, B=B2) for visual context.
    Cells 1-13 show per-band attribution maps using a diverging red/blue
    colormap: red = pushed model toward the class, blue = pushed model
    away from it.
    Cell 14 shows the aggregate attribution
    (sum of absolute values across all bands).

    Args:
        raw: [13, H, W] raw digital-number tensor, used for the RGB composite.
        attrs: [13, H, W] attribution tensor from integrated_gradients().
        predicted_class: Class predicted by the model.
        target_class: Class being explained.
        output_path: Save figure here, or None to display interactively.

    Returns:
        None | plt.Figure: None if figure is saved, else returns figure itself.
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
        a = attrs[ch].numpy()
        vmax = (
            max(abs(a.min()), abs(a.max())) or 1.0
        )  # fall back to 1.0 if band has no attribution
        flat[ch + 1].imshow(a, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        flat[ch + 1].set_title(MS_BAND_NAMES[ch], fontsize=7)
        flat[ch + 1].axis("off")

    flat[14].imshow(_aggregate_attribution(attrs), cmap="hot")
    flat[14].set_title("Aggregate\n(sum |attr| all bands)", fontsize=8)
    flat[14].axis("off")

    plt.tight_layout()
    return _save_or_show(fig, output_path)


def _save_or_show(
    fig: plt.Figure, output_path: str | Path | None
) -> None | plt.Figure:
    """
    If an output path is specified, it saves the given figure to that path.
    Otherwise, it returns the image.

    Args:
        fig (plt.Figure): Figure to save or show.
        output_path (str | Path | None): Path for saving figure.

    Returns:
        None | plt.Figure: None if figure is saved, else returns figure itself.
    """
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {Path(output_path).resolve()}")
        plt.close(fig)
        return None
    else:
        return fig


def main() -> None:
    """
    Performs Integrated Gradients on given input file using specified model.
    Optionally one can specify the class to explain, otherwise defaults to
    predicted class. Visualization includes original image and Integrated
    Gradients explanation. For RGB, this is one image of attributions
    aggregated across bands. For MS this includes one plot per band,
    and an aggregated plot.

    Argument parser arguments when running in terminal:
            --model_path (float):
            Path of model to use.
        --input_file (jpg or tif):
            Image file to explain.
        --target_class (int):
            Optionally selects a class to explain decision for.
        --n_steps (int):
            Number of interpolation steps.
        --output_path (str):
            Path to save visualiation to.

    Raises:
        ValueError: Invalid target class provided.
    """
    parser = argparse.ArgumentParser(
        description="Integrated Gradients for EuroSAT CNN models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path", required=True, help="Path to saved model (.pkl)"
    )
    parser.add_argument(
        "--input_file", required=True, help=".jpg for RGB or .tif for MS"
    )
    parser.add_argument(
        "--target_class",
        type=int,
        default=None,
        help="Class index to explain. Defaults to the predicted class.",
    )
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--output_path", default=None)
    args = parser.parse_args()

    # Infer image type from file extension
    is_ms = Path(args.input_file).suffix.lower() in {".tif", ".tiff"}

    model = torch.load(args.model_path, map_location="cpu", weights_only=False)
    model.eval()

    if is_ms:
        raw, preprocessed, baseline = load_ms_ig(args.input_file)
    else:
        image, baseline = load_rgb_ig(args.input_file)
        raw = preprocessed = (
            image  # for RGB, the raw image is also the model input
        )

    with torch.no_grad():
        predicted_class = int(
            model(preprocessed.unsqueeze(0)).argmax(1).item()
        )

    target_class = (
        args.target_class if args.target_class is not None else predicted_class
    )

    if target_class not in label_map:
        valid = ", ".join(f"{k} ({v})" for k, v in label_map.items())
        raise ValueError(
            f"Invalid target class {target_class}. Valid options are: {valid}"
        )

    print(f"Predicted : {label_map[predicted_class]}")
    print(f"Explaining: {label_map[target_class]}")

    attrs = integrated_gradients(
        model, preprocessed, baseline, target_class, args.n_steps
    )

    if is_ms:
        fig = visualise_ms(
            raw, attrs, predicted_class, target_class, args.output_path
        )
    else:
        fig = visualise_rgb(
            raw, attrs, predicted_class, target_class, args.output_path
        )

    if fig is not None:
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
