from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from clean import clean_sealake_folder
from download import get_dataset_path
from label_map import label_map
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class EuroSATDataset(Dataset, ABC):
    """Abstract class for RGB and MS dataset"""

    def __init__(self, root: str | Path, transform: Optional[Callable] = None) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        for idx, class_name in label_map.items():
            class_dir = self.root / class_name
            for f in class_dir.iterdir():
                self.samples.append((f.name, f, idx))

    def __len__(self) -> int:
        return len(self.samples)

    @abstractmethod
    def _load_image(self, path: Path):
        pass

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = self._load_image(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class EuroSATRGBDataset(EuroSATDataset):
    """Dataset for RGB jpg images"""

    def _load_image(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")


class EuroSATMSDataset(EuroSATDataset):
    """Dataset for MS tif files, return torch tensor of dimension [13, H, W], so for us [13, 64, 64]"""

    def _load_image(self, path: Path) -> torch.Tensor:
        import tifffile  # placed here so package does not need to be imported when not necessary

        arr = tifffile.imread(path)
        arr = arr.astype(np.float32)
        tensor = torch.from_numpy(arr).permute(
            2, 0, 1
        )  # Reorders axes to match PyTorch's conventions → [13, H, W]
        return tensor


def create_dataloaders(
    type: str, batch_size: int = 64
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Method to create the dataloaders for either the RGB or the MS data

    Args:
        type (str): Possible values are "rgb" or "ms"
        batch_size (int, optional): Defaults to 64.

    Raises:
        ValueError: If wrong type is passed it will raise a value error

    Returns:
        type (DataLoader): The three data loaders for train, test, and validation split.
    """
    path = get_dataset_path()
    clean_sealake_folder(path)

    train_path, val_path, test_path = get_train_val_test_splits()

    if type == "rgb":
        train_ds = EuroSATRGBDataset(root=Path(path) / "EuroSAT", csv_path=train_path)
        val_ds = EuroSATRGBDataset(root=Path(path) / "EuroSAT", csv_path=val_path)
        test_ds = EuroSATRGBDataset(root=Path(path) / "EuroSAT", csv_path=test_path)
    elif type == "ms":
        train_ds = EuroSATMSDataset(root=Path(path) / "EuroSAT", csv_path=train_path)
        val_ds = EuroSATMSDataset(root=Path(path) / "EuroSAT", csv_path=val_path)
        test_ds = EuroSATMSDataset(root=Path(path) / "EuroSAT", csv_path=test_path)
    else:
        raise ValueError()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
