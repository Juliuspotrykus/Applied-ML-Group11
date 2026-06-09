"""This script implements ablation studies for the different bands in the ms dataset. 
For each requested band combination it drops those bands from the input, 
retrains the model from scratch on the combined train+val set, evaluates
on the test set, and reports the mean and standard error of the test macro F1
across several runs.

Usage:
    # All bands:
    python -m eurosat_classification.train.ablation

    # Drop a single band (B01) and a pair of bands (B09, B10) as separate experiments, 
    # with 5 runs of 30 epochs each:
    python -m eurosat_classification.train.ablation \\
        --drop B01 --drop B09 B10 --n-runs 5 --epochs 30 --batch-size 64
"""

import argparse
import math

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from ..data.band_names import MS_BAND_NAMES
from ..data.datasets import create_dataloaders
from .run_training import BEST_PARAMS, build_config_from_params
from .train import evaluate, train_model

BAND_INDEX = {name.split()[0]: i for i, name in enumerate(MS_BAND_NAMES)}


class BandSubset(Dataset):
    """Dataset wrapper that keeps only a subset of the input channels (bands) used for the ablation studies."""

    def __init__(self, base: Dataset, keep_idx: list[int]) -> None:
        """Initializes the BandSubset dataset.
        Args:
            base (Dataset): The base dataset class.
            keep_idx (list[int]): The indices of the channels to keep.
        """
        self.base = base
        self.keep_idx = keep_idx

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Returns the item at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple[torch.Tensor, int]: The image and label at the specified index.
        """
        img, label = self.base[idx]
        return img[self.keep_idx], label


def make_loaders(keep_idx: list[int], batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Creates DataLoaders for the combined train+val dataset and the test set, keeping only the specified channels.

    Args:
        keep_idx (list[int]): The indices of the channels to keep.
        batch_size (int): The batch size for the data loaders.

    Returns:
        tuple[DataLoader, DataLoader]: The train+val and test data loaders.
    """
    train_loader, val_loader, test_loader = create_dataloaders("ms", batch_size)
    trainval_ds = BandSubset(
        ConcatDataset([train_loader.dataset, val_loader.dataset]), keep_idx
    )
    test_ds = BandSubset(test_loader.dataset, keep_idx)

    trainval_loader = DataLoader(
        trainval_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
    )
    return trainval_loader, test_loader


def run(
    keep_idx: list[int], n_runs: int, epochs: int, batch_size: int
) -> tuple[float, float]:
    """Runs the training and evaluation for the specified channel subset, number of runs, epochs, and batch size,
        and returns the mean and standard error of the mean of the test F1 scores.

    Args:
        keep_idx (list[int]): The indices of the channels to keep.
        n_runs (int): The number of runs to perform.
        epochs (int): The number of epochs to train for each run.
        batch_size (int): The batch size for the data loaders.

    Returns:
        tuple[float, float]: The mean and standard error of the mean of the test F1 scores.
    """
    params = BEST_PARAMS["ms"]
    trainval_loader, test_loader = make_loaders(keep_idx, batch_size)

    scores = []
    for i in range(1, n_runs + 1):
        config = build_config_from_params("ms", params)
        config.in_channels = len(keep_idx)
        model, _ = train_model(
            config, trainval_loader, val_loader=None, lr=params["lr"], epochs=epochs
        )
        _, test_f1 = evaluate(model, test_loader, nn.CrossEntropyLoss())
        scores.append(test_f1)
        print(f"  run {i}/{n_runs}: test F1 = {test_f1:.4f}")

    n = len(scores)
    mean = sum(scores) / n
    sem = math.sqrt(sum((s - mean) ** 2 for s in scores) / (n - 1)) / math.sqrt(n)
    return mean, sem


def main() -> None:
    """Main function to run the MS band ablation study. Parses command-line arguments for the bands to drop, number of runs,
    epochs, and batch size. When --drop is not added it trains on all bands, otherwise it runs each specified combination of dropped bands. In both cases it reports the mean and
    standard error of the mean of the test F1 scores.
    """
    parser = argparse.ArgumentParser(description="MS band ablation")
    parser.add_argument(
        "--drop",
        action="append",
        nargs="+",
    )
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if args.n_runs <= 1:
        parser.error(
            "--n-runs must be at least 2 to compute a standard error of the mean"
        )

    if not args.drop:
        print("=== all bands ===")
        mean, sem = run(
            list(BAND_INDEX.values()), args.n_runs, args.epochs, args.batch_size
        )
        print("\n=== results (test macro F1) ===")
        print(f"all bands: {mean:.4f} +/- {sem:.4f}")
        return

    unknown = [b for combo in args.drop for b in combo if b not in BAND_INDEX]
    if unknown:
        parser.error(f"Unknown bands: {unknown}")

    results = []
    for combo in args.drop:
        keep_idx = [i for tok, i in BAND_INDEX.items() if tok not in set(combo)]
        print(f"\n=== drop {combo} ({len(keep_idx)} bands) ===")
        mean, sem = run(keep_idx, args.n_runs, args.epochs, args.batch_size)
        results.append((combo, mean, sem))

    print("\n=== results (test macro F1) ===")
    for combo, mean, sem in results:
        print(f"drop {combo}: {mean:.4f} +/- {sem:.4f}")


if __name__ == "__main__":
    main()
