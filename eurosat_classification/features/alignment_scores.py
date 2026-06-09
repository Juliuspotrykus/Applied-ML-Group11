"""
To run: uv run python -m eurosat_classification.features.alignment_scores

IMPORTANT: band_attribution_runner.py needs to have been run before this!!

Main and secondary bands were chosen based on literature.
Main bands were those identified by both sources,
secondary bands are those identified by only one of the sources.

AnnualCrop
- B5, B6, B7, B8A [1]
- B5, B6, B7, B8, B8A [2] -> vegetation classification
- B11, B12 [2] -> moisture
- B4 [2] -> soil contrast

Forest
- B5, B6, B7, B8 [1]
- B5, B6, B7, B8, B8A [2] -> vegetation classification
- B11, B12 [2] -> moisture
- B4 [2] -> soil contrast

HerbaceousVegetation
- B5, B6, B7, B8A [1]
- B5, B6, B7, B8, B8A [2] -> vegetation classification
- B11, B12 [2] -> moisture

Highway
- B11, B12, B8 [1]
- B2 [2] -> man-made object detection
- B4 [2] -> urban and soil separation
- B11, B12 [2] -> moisture contrast

Industrial
- B11, B12, B8 [1]
- B2 [2] -> man-made object detection
- B4 [2] -> urban and soil separation
- B11, B12 [2] -> moisture contrast

Pasture
- B5, B6, B7, B8A [1]
- B5, B6, B7, B8, B8A [2] -> vegetation classification
- B11, B12 [2] -> moisture

PermanentCrop
- B5, B6, B7, B8A [1]
- B5, B6, B7, B8, B8A [2] -> vegetation classification
- B11, B12 [2] -> moisture
- B4 [2] -> soil contrast

Residential
- B11, B12, B8 [1]
- B2 [2] -> man-made object detection
- B4 [2] -> urban and soil separation
- B11, B12 [2] -> moisture contrast

River
- B3, B8, B11 [1]
- B3 [2] -> water contrast
- B8 [2] -> shoreline mapping
- B4 [2] -> land water separation

SeaLake
- B3, B8, B11 [1]
- B3 [2] -> water contrast
- B8 [2] -> shoreline mapping
- B4 [2] -> land water separation

[1] https://www.mdpi.com/2071-1050/17/22/10324
[2] https://custom-scripts.sentinel-hub.com/sentinel-2/bands/

"""

import argparse
from pathlib import Path

import numpy as np
import torch

from ..data.band_names import MS_BAND_NAMES
from ..data.label_map import label_map


def _auto_device() -> str:
    """
    Automatically selects the best available PyTorch device.

    Returns:
            str: "cuda" if a CUDA-capable GPU is available, otherwise "cpu".
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_class_attribution(class_idx: int) -> np.lib.npyio.NpzFile:
    """
    Loads pre-calculated class attribution scores for specific class.
    These are calculated by `band_attribution_runner.py` file.

    Args:
            class_idx (int): Integer index for class.

    Returns:
            np.lib.npyio.NpzFile: Numpy file with class attribution
                                scores including:
               - positive: Mean positive attribution per band.
               - negative: Mean negative attribution per band.
               - count: Number of images used.
               - band_names: Sentinel-2 band names.
    """
    path = (
        Path("results/band_attribution")
        / f"ms_train_attribution_class{class_idx}.npz"
    )
    return np.load(path, allow_pickle=True)


def main() -> dict[int, float]:
    """
    Computes band alignment scores with expected important bands per class.

    Score is computed as follows:
    - importance measure is calculated per band based on positive and negative
        attributions
    - proportion of importance of main bands relative to all bands is summed to
        weighted proportion of importance of secondary bands relative to all
        bands

    These scores are printed and saved as npz file.

    Argument parser arguments when running in terminal:
        --alpha (float):
            Weight assigned to secondary bands when computing alignment.
            Default to 0.75.
        --output_dir (str):
            Directory where alignment results are stored.
            Default to "results/alignment".

    Returns:
        dict[int, float]: Alignment score for each class.
    """
    parser = argparse.ArgumentParser(
        description="Alignment of attribution scores with literature.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.75,
        help="Weighting given to secondary bands.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/alignment",
        help="Directory where alignment scores will be saved.",
    )
    args = parser.parse_args()
    band_names = MS_BAND_NAMES

    expect_main_class_to_band = {
        0: [
            "B5 - Red Edge 1",
            "B6 - Red Edge 2",
            "B7 - Red Edge 3",
            "B8A - Narrow NIR",
        ],
        1: [
            "B5 - Red Edge 1",
            "B6 - Red Edge 2",
            "B7 - Red Edge 3",
            "B8 - NIR",
        ],
        2: [
            "B5 - Red Edge 1",
            "B6 - Red Edge 2",
            "B7 - Red Edge 3",
            "B8A - Narrow NIR",
        ],
        3: ["B11 - SWIR 1", "B12 - SWIR 2"],
        4: ["B11 - SWIR 1", "B12 - SWIR 2"],
        5: [
            "B5 - Red Edge 1",
            "B6 - Red Edge 2",
            "B7 - Red Edge 3",
            "B8A - Narrow NIR",
        ],
        6: [
            "B5 - Red Edge 1",
            "B6 - Red Edge 2",
            "B7 - Red Edge 3",
            "B8A - Narrow NIR",
        ],
        7: ["B11 - SWIR 1", "B12 - SWIR 2"],
        8: ["B3 - Green", "B8 - NIR"],
        9: ["B3 - Green", "B8 - NIR"],
    }

    expect_secondary_class_to_band = {
        0: ["B8 - NIR", "B11 - SWIR 1", "B12 - SWIR 2", "B4 - Red"],
        1: ["B8A - Narrow NIR", "B11 - SWIR 1", "B12 - SWIR 2", "B4 - Red"],
        2: ["B8 - NIR", "B11 - SWIR 1", "B12 - SWIR 2"],
        3: ["B8 - NIR", "B2 - Blue", "B4 - Red"],
        4: ["B8 - NIR", "B2 - Blue", "B4 - Red"],
        5: ["B8 - NIR", "B11 - SWIR 1", "B12 - SWIR 2"],
        6: ["B8 - NIR", "B11 - SWIR 1", "B12 - SWIR 2", "B4 - Red"],
        7: ["B8 - NIR", "B2 - Blue", "B4 - Red"],
        8: ["B11 - SWIR 1", "B4 - Red"],
        9: ["B11 - SWIR 1", "B4 - Red"],
    }

    alignment = {}

    for class_idx, _ in label_map.items():
        results = load_class_attribution(class_idx)

        positive = results["positive"]
        negative = results["negative"]

        importance = np.abs(positive) + np.abs(negative)
        total = importance.sum()

        if total == 0:
            alignment[class_idx] = 0.0
            continue

        # Identify bands belonging to main and secondary band sets
        main_mask = np.isin(
            band_names,
            expect_main_class_to_band[class_idx],
        )
        secondary_mask = np.isin(
            band_names,
            expect_secondary_class_to_band[class_idx],
        )

        # Compute fraction of total attribution mass assigned
        # to main and secondary expected band groups
        attr_main = importance[main_mask].sum() / total
        attr_secondary = importance[secondary_mask].sum() / total

        alignment[class_idx] = attr_main + args.alpha * attr_secondary

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "ms_alignment_scores.npz"
    np.savez(
        npz_path,
        alignment=np.array([alignment[i] for i in sorted(alignment)]),
        class_indices=np.array(sorted(alignment)),
        class_names=np.array(
            [label_map[i] for i in sorted(alignment)],
            dtype=object,
        ),
        alpha=args.alpha,
    )

    print("\nAlignment Scores")
    for class_idx in sorted(alignment):
        print(f"{label_map[class_idx]:<22}{alignment[class_idx]:.3f}")
    mean_alignment = np.mean(list(alignment.values()))
    print(f"{'Mean alignment':<22}{mean_alignment:.3f}")

    return alignment


if __name__ == "__main__":
    main()
