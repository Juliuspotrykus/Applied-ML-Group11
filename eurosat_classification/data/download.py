import kagglehub


def get_dataset_path() -> str:
    return kagglehub.dataset_download("apollo2506/eurosat-dataset")
