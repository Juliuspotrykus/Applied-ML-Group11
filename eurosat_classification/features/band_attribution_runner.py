"""
Dataset-level Integrated Gradients band attribution.

Iterates over the training set, accumulates total positive and negative IG
attributions per band (using each image's true label as the target class),
saves results to .npz, and writes a bar chart.

Usage (local):
    python -m eurosat_classification.features.band_attribution_runner \
        --model_path models/ms_model_final.pkl \
        --image_type ms \
        --output_dir results/band_attribution \

    # Use --max_samples 200 to do a quick smoke-test before the full run.

Usage (cluster):
    sbatch eurosat_classification/features/band_attribution.sh ms models/ms_model_final.pkl
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..data.band_names import MS_BAND_NAMES, RGB_BAND_NAMES
from ..data.datasets import EuroSATMSDataset, EuroSATRGBDataset
from ..data.download import get_dataset_path
from ..data.preprocessors import normalize_MS_img
from ..data.split import get_train_val_test_splits
from .integrated_gradients import band_attribution_totals


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def plot_band_attribution(
    positive: np.ndarray,
    negative: np.ndarray,
    band_names: list[str],
    output_path: Path,
    title: str,
) -> None:
    x = np.arange(len(band_names))
    fig, ax = plt.subplots(figsize=(max(8, len(band_names) * 0.9), 5))
    ax.bar(x, positive, color="tomato", label="Positive")
    ax.bar(x, negative, color="steelblue", label="Negative")
    ax.set_xticks(x)
    ax.set_xticklabels(band_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean attribution per band (class-balanced)")
    ax.set_title(title)
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Dataset-level IG band attribution totals (train set, true-label target).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path", required=True, help="Path to saved model (.pkl)"
    )
    parser.add_argument("--image_type", required=True, choices=["rgb", "ms"])
    parser.add_argument("--output_dir", default="results/band_attribution")
    parser.add_argument(
        "--n_steps",
        type=int,
        default=50,
        help="IG interpolation steps per image.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Process only this many images (useful for quick smoke-tests).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (cuda/mps/cpu). Auto-detected if not set.",
    )
    parser.add_argument(
        "--target_class",
        type=int,
        default=None,
        help="If set, only process images with this label and explain that class.",
    )
    args = parser.parse_args()

    device = args.device or _auto_device()
    print(f"Device:     {device}")

    model = torch.load(
        args.model_path, map_location=device, weights_only=False
    )
    model.eval()

    data_root = Path(get_dataset_path())
    train_csv, _, _ = get_train_val_test_splits()

    if args.image_type == "ms":
        dataset = EuroSATMSDataset(
            root=data_root / "EuroSATallBands",
            csv_path=train_csv,
            transform=normalize_MS_img,
        )
        band_names = MS_BAND_NAMES
    else:
        dataset = EuroSATRGBDataset(
            root=data_root / "EuroSAT", csv_path=train_csv
        )
        band_names = RGB_BAND_NAMES

    n = (
        len(dataset)
        if args.max_samples is None
        else min(args.max_samples, len(dataset))
    )
    print(f"Image type: {args.image_type.upper()}")
    print(f"Split:      train  ({n} images)")
    print(f"n_steps:    {args.n_steps}")
    if args.target_class is not None:
        print(
            f"Target class: {args.target_class} (only images with this label)"
        )

    results = band_attribution_totals(
        model=model,
        dataset=dataset,
        n_steps=args.n_steps,
        target_class=args.target_class,
        max_samples=args.max_samples,
        device=device,
        verbose=True,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_suffix = (
        f"_class{args.target_class}" if args.target_class is not None else ""
    )
    npz_path = (
        out_dir / f"{args.image_type}_train_attribution{class_suffix}.npz"
    )
    np.savez(
        npz_path,
        positive=results["positive"],
        negative=results["negative"],
        count=np.array(results["count"]),
        band_names=np.array(band_names),
    )
    print(f"Saved data: {npz_path.resolve()}")

    class_label = (
        f" / class {args.target_class}"
        if args.target_class is not None
        else ""
    )
    plot_band_attribution(
        positive=results["positive"],
        negative=results["negative"],
        band_names=band_names,
        output_path=out_dir
        / f"{args.image_type}_train_attribution{class_suffix}.png",
        title=(
            f"Band attribution — {args.image_type.upper()} / train{class_label} "
            f"(n={results['count']}, steps={args.n_steps})"
        ),
    )


if __name__ == "__main__":
    main()
