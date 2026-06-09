import kagglehub


def get_dataset_path() -> str:
    """
    Gets path that EuroSAT dataset from Kaggle was saved to.

    Returns:
        str: Path to dataset.
    """
    return kagglehub.dataset_download("apollo2506/eurosat-dataset")
