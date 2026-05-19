from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from label_map import label_map
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class EuroSATDataset(Dataset, ABC):
    """Abstract class for RGB and MS dataset"""

    def __init__(self, root: str | Path, split_csv_path: str, transform: Optional[Callable] = None) -> None:
        self.root = Path(root)
        self.split_csv = pd.read_csv(split_csv_path)
        self.transform = transform
        self.samples = []

        # Get the file names for the files in the split (without folder name or extension)
        self.split_filenames = set(self.split_csv["Filename"].apply(lambda path: Path(path).stem))

        for idx, class_name in label_map.items():
            class_dir = self.root / class_name
            for f in class_dir.iterdir():
                if f.stem in self.split_filenames:
                    self.samples.append((f.name, f, idx))

    def __len__(self) -> int:
        return len(self.samples)

    @abstractmethod
    def _load_image(self, path: Path) -> torch.Tensor:
        pass

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        _, path, label = self.samples[idx]
        img = self._load_image(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class EuroSATRGBDataset(EuroSATDataset):
    """Dataset for RGB jpg images"""

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return transforms.ToTensor()(img)


class EuroSATMSDataset(EuroSATDataset):
    """Dataset for MS tif files, return torch tensor of dimension [13, H, W], so for us [13, 64, 64]"""

    def _load_image(self, path: Path) -> torch.Tensor:
        import tifffile  # placed here so package does not need to be imported when not necessary

        arr = tifffile.imread(path)
        arr = arr.astype(np.float32)
        tensor = torch.from_numpy(arr).permute(
            2, 0, 1
        )  # Reorders axes to match PyTorch's conventions -> [13, H, W]
        return tensor
