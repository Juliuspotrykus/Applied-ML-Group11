from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from data.download import get_dataset_path
from data.clean import clean_sealake_folder


class EuroSATDataset(Dataset):
    """Super class for RGB and MS dataset"""

    FILE_TYPE: str = ""

    def __init__(self, root: str | Path, transform: Optional[Callable] = None) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        for idx, class_name in label_map.items():
            class_dir = self.root / class_name
            for f in class_dir.iterdir():
                self.samples.append((f, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path):
        raise NotImplementedError

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = self._load_image(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class EuroSATRGBDataset(EuroSATDataset):
    """Dataset for RGB jpg images"""

    FILE_TYPE = ".jpg"

    def _load_image(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")


class EuroSATMSDataset(EuroSATBase):
    """Dataset for MS tif files, return torch tensor of dimension [13, H, W], so for us [13, 64, 64]"""

    FILE_TYPE = ".tif"

    def _load_image(self, path: Path) -> torch.Tensor:
        import tifffile  # placed here so package does not need to be imported when not necessary

        arr = tifffile.imread(path)
        arr = arr.astype(np.float32)
        tensor = torch.from_numpy(arr).permute(
            2, 0, 1
        )  # Reorders axes to match PyTorch's conventions → [13, H, W]
        return tensor
