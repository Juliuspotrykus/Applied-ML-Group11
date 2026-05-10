import os
from torchvision import datasets, transforms
import numpy as np

# original path used -> TODO change to path within folder that is added to gitignore
path = "/Users/luciaadan/.cache/kagglehub/datasets/apollo2506/eurosat-dataset/versions/6"

# Use ImageLoader to extract files from folders using folder name as class labels
rgb_path = os.path.join(path, "EuroSAT")
# transform rgb images to tensors
rgb_dataset = datasets.ImageFolder(root=rgb_path, transform=transforms.ToTensor())
print(rgb_dataset.classes)
print(len(rgb_dataset))

# testing -> 10 classes, 27000 files
image_rgb, label_rgb = rgb_dataset[0]
# shape of images is 3, 64, 64
print(image_rgb.shape)    
print(label_rgb)          

# 13 band SeaLake folder contains 597 incorrect files that need to be removed
ms_path = os.path.join(path, "EuroSATallBands")
sealake_path_ms = os.path.join(ms_path, "SeaLake")

bombay_files = [f for f in os.listdir(sealake_path_ms) if f.startswith("Bombay")]
jakarta_files = [f for f in os.listdir(sealake_path_ms) if f.startswith("Jakarta")]

print("Bombay files:", len(bombay_files))
print("Jakarta files:", len(jakarta_files))
print("Total to remove:", len(bombay_files) + len(jakarta_files))

# TODO: uncomment code to delete files
# for f in bombay_files + jakarta_files:
#     os.remove(os.path.join(sealake_path_ms, f))