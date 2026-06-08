import argparse
import math

import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from ..data.datasets import create_dataloaders
from .run_training import BEST_PARAMS, build_config_from_params
from .train import evaluate, train_model


def make_trainval_loader(image_type: str, batch_size: int):
    """Build train+validation and test loader."""
    train_loader, val_loader, test_loader = create_dataloaders(
        image_type, batch_size
    )
    trainval_ds = ConcatDataset([train_loader.dataset, val_loader.dataset])
    trainval_loader = DataLoader(
        trainval_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    return trainval_loader, test_loader


def train_once(
    image_type: str, params: dict, trainval_loader, test_loader, epochs: int
) -> float:
    """Train one model on train+val for a fixed number of epochs, return test macro F1."""
    config = build_config_from_params(image_type, params)
    model, _ = train_model(
        config,
        trainval_loader,
        val_loader=None,
        lr=params["lr"],
        epochs=epochs,
    )

    _, test_f1 = evaluate(model, test_loader, nn.CrossEntropyLoss())
    return test_f1


def run_comparison(
    image_type: str, params: dict, n_runs: int, epochs: int, batch_size: int
) -> None:
    if n_runs <= 1:
        raise ValueError(
            "n_runs must be at least 2 to compute a standard error of the mean"
        )

    trainval_loader, test_loader = make_trainval_loader(image_type, batch_size)

    scores = []
    for run in range(1, n_runs + 1):
        f1 = train_once(
            image_type, params, trainval_loader, test_loader, epochs
        )
        scores.append(f1)
        print(f"[{image_type}] run {run}/{n_runs}: test F1 = {f1:.4f}")

    n = len(scores)
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / (n - 1)
    sem = math.sqrt(variance) / math.sqrt(n)

    print(f"\n=== {image_type} over {n} runs ===")
    print(f"Mean test macro F1: {mean:.4f}")
    print(f"Std error of mean:  {sem:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare RGB and MS models over repeated training runs."
    )
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if args.n_runs <= 1:
        parser.error(
            "--n-runs must be at least 2 to compute a standard error of the mean"
        )

    run_comparison(
        "rgb",
        BEST_PARAMS["rgb"],
        n_runs=args.n_runs,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    run_comparison(
        "ms",
        BEST_PARAMS["ms"],
        n_runs=args.n_runs,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
