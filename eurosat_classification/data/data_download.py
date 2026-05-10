import kagglehub

# Download latest version
path = kagglehub.dataset_download(
    "apollo2506/eurosat-dataset"
    path="./data"
)

print("Path to dataset files:", path)