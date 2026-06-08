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
    """Wraps a dataset and keeps only the given channel indices of each image."""

    def __init__(self, base: Dataset, keep_idx: list[int]) -> None:
        self.base = base
        self.keep_idx = keep_idx

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img, label = self.base[idx]
        return img[self.keep_idx], label


def make_loaders(keep_idx: list[int], batch_size: int):
    """Build train+val and test loaders that give only the kept bands."""
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
    """Train n_runs times on the kept bands, return (mean test F1, sem)."""
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
    parser = argparse.ArgumentParser(description="MS band ablation")
    parser.add_argument(
        "--drop",
        action="append",
        nargs="+",
        required=True,
    )
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if args.n_runs <= 1:
        parser.error(
            "--n-runs must be at least 2 to compute a standard error of the mean"
        )

    unknown = [b for combo in args.drop for b in combo if b not in BAND_INDEX]
    if unknown:
        parser.error(f"Unknown bands: {unknown}")

    print("=== all bands ===")
    all_mean, all_sem = run(
        list(BAND_INDEX.values()), args.n_runs, args.epochs, args.batch_size
    )

    results = []
    for combo in args.drop:
        keep_idx = [i for tok, i in BAND_INDEX.items() if tok not in set(combo)]
        print(f"\n=== drop {combo} ({len(keep_idx)} bands) ===")
        mean, sem = run(keep_idx, args.n_runs, args.epochs, args.batch_size)
        results.append((combo, mean, sem))

    print("\n=== results (test macro F1) ===")
    print(f"all bands: {all_mean:.4f} +/- {all_sem:.4f}")
    for combo, mean, sem in results:
        print(f"drop {combo}: {mean:.4f} +/- {sem:.4f}  (diff {mean - all_mean:+.4f})")


if __name__ == "__main__":
    main()
