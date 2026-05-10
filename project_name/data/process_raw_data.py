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
