"""Evaluation loop used by the robustness runner."""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset


class _PerturbedDataset(Dataset):
    """Wraps an existing dataset and applies a perturbation on-the-fly.

    Each sample is perturbed with a deterministic seed (base_seed + index)
    so that stochastic perturbations (noise, salt-and-pepper) are reproducible
    across repeated calls with the same seed.
    """

    def __init__(self, base: Dataset, fn: Callable, seed: int = 42) -> None:
        self.base = base
        self.fn = fn
        self.seed = seed

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, label = self.base[idx]
        torch.manual_seed(self.seed + idx)
        return self.fn(img), label


def evaluate(
    model: torch.nn.Module,
    dataset: Dataset,
    perturb_fn: Callable | None,
    device: torch.device,
    seed: int = 42,
    batch_size: int = 64,
) -> dict:
    """Run a single forward pass over the dataset and return metrics.

    Args:
        model: Trained CNN already in eval mode.
        dataset: Test dataset (un-perturbed raw version).
        perturb_fn: Applied to each image tensor before inference. None = clean run.
        device: Torch device.
        seed: Base random seed for stochastic perturbations.
        batch_size: DataLoader batch size (num_workers=0 to respect per-sample seeds).

    Returns:
        dict with keys:
            f1 (float), mean_confidence (float),
    """
    eval_ds = _PerturbedDataset(dataset, perturb_fn, seed) if perturb_fn is not None else dataset
    loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds: list[int] = []
    all_labels: list[int] = []
    all_confs: list[float] = []

    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device))
            preds = logits.argmax(dim=1)
            confs = F.softmax(logits, dim=1).max(dim=1).values
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_confs.extend(confs.cpu().tolist())

    n = len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    mean_confidence = sum(all_confs) / n
    return {
        "f1": macro_f1,
        "mean_confidence": mean_confidence,
        "preds": all_preds,
    }
