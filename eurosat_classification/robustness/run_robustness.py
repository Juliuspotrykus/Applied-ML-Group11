"""
Robustness evaluation for EuroSAT RGB and MS classifiers.

Usage:
    python -m eurosat_classification.robustness.run_robustness rgb
    python -m eurosat_classification.robustness.run_robustness ms

Options:
    --max_samples N   Limit test-set size (default 1000; -1 for all).
    --output_dir DIR  Output directory (default: results/robustness).
    --device DEVICE   cuda / mps / cpu (auto-detected).
    --batch_size N    Default: 64.
    --seed N          Default: 42.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import torch

from ..data.band_names import MS_BAND_NAMES
from ..data.datasets import EuroSATMSDataset, EuroSATRGBDataset
from ..data.download import get_dataset_path
from ..data.preprocessors import normalize_MS_img
from ..data.split import get_train_val_test_splits
from . import perturbations as P
from .evaluate import evaluate


class PerturbSpec:
    """Describes a family of perturbations parameterised by severity level."""

    def __init__(
        self,
        name: str,
        unit: str,
        severities: list[Any],
        fn: Callable[[Any], Callable[[torch.Tensor], torch.Tensor]],
        labels: list[str] | None = None,
    ) -> None:
        """
        Initializes the perturbation specification template.

        Args:
            name (str): Short identifier used in CSV/plot output.
            unit (str): Unit string appended to numeric severity values
            in labels
            severities (list[Any]): Ordered list of severity values
            passed to fn.
            fn (Callable[[Any], Callable[[torch.Tensor], torch.Tensor]]):
                Factory that takes a severity value and returns a
                (tensor → tensor) callable.
            labels (list[str] | None, optional): Optional per-severity display
                    labels. Overrides auto-generated labels.. Defaults to None.
        """
        self.name = name
        self.unit = unit
        self.severities = severities
        self.fn = fn
        self.labels = labels

    def label(self, i: int) -> str:
        """
        Return the display label for severity index i.

        Args:
            i (int): Target element position index within the severity array.

        Returns:
            str: Descriptive label for given severity level.
        """
        if self.labels:
            return self.labels[i]
        return (
            f"{self.severities[i]}{self.unit}"
            if self.unit
            else str(self.severities[i])
        )


RGB_SPECS: list[PerturbSpec] = [
    PerturbSpec(
        "rotation",
        "°",
        [0, 15, 30, 45, 90, 135, 180],
        fn=lambda degrees: lambda img: P.rotate_rgb(img, degrees),
    ),
    PerturbSpec(
        "gaussian_noise",
        "",
        [0.0, 0.01, 0.05, 0.1, 0.2, 0.5],
        labels=["0", "0.01", "0.05", "0.1", "0.2", "0.5"],
        fn=lambda sigma: lambda img: P.noise_rgb(img, sigma),
    ),
    PerturbSpec(
        "brightness",
        "×",
        [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        fn=lambda factor: lambda img: P.brightness_rgb(img, factor),
    ),
    PerturbSpec(
        "salt_and_pepper",
        "",
        [0.0, 0.01, 0.05, 0.1, 0.2, 0.4],
        labels=["0%", "1%", "5%", "10%", "20%", "40%"],
        fn=lambda density: lambda img: P.salt_pepper_rgb(img, density),
    ),
    PerturbSpec(
        "contrast",
        "×",
        [0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
        fn=lambda factor: lambda img: P.contrast_rgb(img, factor),
    ),
    PerturbSpec("hflip", "", [1], labels=["flipped"], fn=lambda _: P.hflip),
    PerturbSpec("vflip", "", [1], labels=["flipped"], fn=lambda _: P.vflip),
]

MS_SPECS: list[PerturbSpec] = [
    PerturbSpec(
        "rotation",
        "°",
        [0, 15, 30, 45, 90, 135, 180],
        fn=lambda degrees: lambda img: P.rotate_ms(img, degrees),
    ),
    PerturbSpec(
        "gaussian_noise",
        "",
        [0.0, 0.1, 0.25, 0.5, 1.0, 2.0],
        labels=["0", "0.1", "0.25", "0.5", "1.0", "2.0"],
        fn=lambda sigma: lambda img: P.noise_ms(img, sigma),
    ),
    PerturbSpec(
        "brightness",
        "×",
        [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        fn=lambda factor: lambda img: P.brightness_ms(img, factor),
    ),
    PerturbSpec(
        "salt_and_pepper",
        "",
        [0.0, 0.01, 0.05, 0.1, 0.2, 0.4],
        labels=["0%", "1%", "5%", "10%", "20%", "40%"],
        fn=lambda density: lambda img: P.salt_pepper_ms(img, density),
    ),
    PerturbSpec(
        "contrast",
        "×",
        [0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
        fn=lambda factor: lambda img: P.contrast_ms(img, factor),
    ),
    PerturbSpec(
        "band_each",
        "",
        list(range(13)),
        labels=MS_BAND_NAMES,
        fn=lambda band_idx: lambda img: P.band_dropout_ms(img, band_idx),
    ),
    PerturbSpec(
        "band_dropout",
        " bands",
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        fn=lambda n_bands: lambda img: P.band_n_dropout_ms(img, n_bands),
    ),
    PerturbSpec("hflip", "", [1], labels=["flipped"], fn=lambda _: P.hflip),
    PerturbSpec("vflip", "", [1], labels=["flipped"], fn=lambda _: P.vflip),
]


def run_suite(
    model: torch.nn.Module,
    dataset: Any,
    specs: list[PerturbSpec],
    device: torch.device,
    batch_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """
    Evaluate model on the clean dataset and on each perturbation level.

    Args:
        model (torch.nn.Module): Neural network classifier evaluating
        the samples.
        dataset (Dataset): Evaluation testing set base data structures.
        specs (list[PerturbSpec]): Specific transformation configurations
        to verify.
        device (torch.device): target execution processing pipeline
        (e.g., CUDA).
        batch_size (int): Data loader single inference windows.
        seed (int): Absolute random seed initializer.

    Returns:
        list[dict[str, Any]]: Array of results dictionaries containing keys:
            - "perturbation" (str): Name of transformation technique
            applied.
            - "severity_label" (str): Printable formatted axis value
            representation.
            - "severity_value" (Any): Metric quantity indicating
            degradation step.
            - "f1" (float): Metric F1 classification score performance
            result.
            - "mean_confidence" (float): Mean probability score
            magnitude tracker.
    """
    model.eval()

    print("  [clean] …", flush=True)
    clean = evaluate(model, dataset, None, device, batch_size=batch_size)
    print(
        f"  [clean] f1={clean['f1']:.4f}  conf={clean['mean_confidence']:.4f}"
    )

    rows = [
        {
            "perturbation": "clean",
            "severity_label": "—",
            "severity_value": "",
            "f1": round(clean["f1"], 6),
            "mean_confidence": round(clean["mean_confidence"], 6),
        }
    ]

    for spec in specs:
        print(f"  [{spec.name}] …", flush=True)
        for i, sev in enumerate(spec.severities):
            result = evaluate(
                model, dataset, spec.fn(sev), device, seed, batch_size
            )
            lbl = spec.label(i)
            rows.append(
                {
                    "perturbation": spec.name,
                    "severity_label": lbl,
                    "severity_value": sev,
                    "f1": round(result["f1"], 6),
                    "mean_confidence": round(result["mean_confidence"], 6),
                }
            )
            print(f"    {lbl:<16}  f1={result['f1']:.4f}", flush=True)

    return rows


def plot_results(
    rows: list[dict], clean_f1: float, modality: str, path: Path
) -> None:
    """
    Save a multi-panel figure with macro-F1 curves per perturbation.

    Args:
        rows (list[dict]): Perturbed rows (clean row excluded).
        clean_f1 (float): Baseline macro-F1 drawn as a reference
        line in each subplot.
        modality (str): "rgb" or "ms" — used for the figure title.
        path (Path): Output PNG path.
    """
    by_perturb: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_perturb[r["perturbation"]].append(r)

    n = len(by_perturb)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes_flat = axes.flatten() if n > 1 else [axes]

    for ax, (name, data) in zip(axes_flat, by_perturb.items()):
        x = list(range(len(data)))
        ax.plot(
            x,
            [d["f1"] for d in data],
            marker="o",
            color="steelblue",
            label="Macro F1",
        )
        ax.axhline(
            clean_f1,
            color="seagreen",
            linestyle=":",
            linewidth=1.5,
            label=f"Clean F1 ({clean_f1:.3f})",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [d["severity_label"] for d in data],
            rotation=40,
            ha="right",
            fontsize=7,
        )
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_ylabel("Score")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(f"Robustness — {modality.upper()}", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {path.resolve()}")


def run_for_modality(
    modality: str,
    data_root: Path,
    test_csv: str,
    max_samples: int,
    device: torch.device,
    batch_size: int,
    seed: int,
    out_dir: Path,
) -> None:
    """
    Load the saved model and test dataset for one modality and run the full
    perturbation suite.

    Args:
        modality (str): Sensory variant designator flag ("rgb" or "ms").
        data_root (Path): Root folder path target hosting standard file
        structural dirs.
        test_csv (str): File metadata localization indexing path for evaluation
        tracking rows.
        max_samples (int): Max sample index ceiling threshold count. If `-1`,
        checks full set.

        device (torch.device): Compute context execution environment wrapper.
        batch_size (int): Image indexing step stride configuration grouping.
        seed (int): Global generator initialization initialization integer.
        out_dir (Path): Output serialization targets base structure folder.
    """
    print(f"\n{'=' * 50}\n  {modality.upper()}\n{'=' * 50}")

    model = torch.load(
        f"models/{modality}_model_final.pkl",
        map_location=device,
        weights_only=False,
    )
    model.to(device).eval()

    if modality == "rgb":
        dataset = EuroSATRGBDataset(
            root=data_root / "EuroSAT", csv_path=test_csv
        )
        specs = RGB_SPECS
    else:
        dataset = EuroSATMSDataset(
            root=data_root / "EuroSATallBands",
            csv_path=test_csv,
            transform=normalize_MS_img,
        )
        specs = MS_SPECS

    if 0 < max_samples < len(dataset):
        from torch.utils.data import Subset

        idx = torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(seed)
        )[:max_samples].tolist()
        dataset = Subset(dataset, idx)

    print(f"  Samples: {len(dataset)}  Device: {device}")
    rows = run_suite(model, dataset, specs, device, batch_size, seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"robustness_{modality}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "perturbation",
                "severity_label",
                "severity_value",
                "f1",
                "mean_confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV:  {csv_path.resolve()}")

    clean_f1 = next(r["f1"] for r in rows if r["perturbation"] == "clean")
    plot_results(
        [r for r in rows if r["perturbation"] != "clean"],
        clean_f1,
        modality,
        out_dir / f"robustness_{modality}.png",
    )


def _auto_device() -> str:
    """
    Return best available torch device.

    Returns:
        str: Best available torch device.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    """
    Runs robustness evaluation for user-defined model.

    Arguments for running from terminal:
        --max_samples (int)
            Limit test-set size (default 1000; -1 for all).
        --output_dir (str)
            Output directory (default: results/robustness).
        --device (str)
            cuda / mps / cpu (default: auto-detected)
        --batch_size (int)
            Default: 64.
        --seed (int)
            Default: 42.

    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("modality", choices=["rgb", "ms"])
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--output_dir", default="results/robustness")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device or _auto_device())
    out_dir = Path(args.output_dir)
    data_root = Path(get_dataset_path())
    _, _, test_csv = get_train_val_test_splits()

    run_for_modality(
        args.modality,
        data_root,
        test_csv,
        args.max_samples,
        device,
        args.batch_size,
        args.seed,
        out_dir,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
