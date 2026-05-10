from datasets import load_dataset
import torch
import numpy as np

# Load RGB dataset from HF: Has 27,000 rows and 2 cols (image, label)
eurosat_rgb = load_dataset("jonathan-roberts1/EuroSAT")
eurosat_rgb = eurosat_rgb["train"]
print(eurosat_rgb)

sample = eurosat_rgb[0]
# images are shape (64, 64, 3)
image = np.array(sample["image"]) 
# labels are integers from 0-9
label = np.array(sample["label"])

print(image)
print(label)