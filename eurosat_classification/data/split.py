import os

from .download import get_dataset_path


def get_train_val_test_splits() -> tuple[str, str, str]:
    """
    Finds the train, val, test splits from RGB version of Kaggle dataset as
    CSV files. Contain filenames (which are unique) of which files are in each
    dataset.

    Raises:
        ValueError: Train set path not found.
        ValueError: Validation  set path not found.
        ValueError: Test  set path not found.

    Returns:
        tuple[str, str, str]: Paths to each of the data splits.
    """
    rgb_path = os.path.join(get_dataset_path(), "EuroSAT")
    train_path = None
    val_path = None
    test_path = None

    for f in os.listdir(rgb_path):
        if f.startswith("train"):
            train_path = os.path.join(rgb_path, f)

        elif f.startswith("validation"):
            val_path = os.path.join(rgb_path, f)

        elif f.startswith("test"):
            test_path = os.path.join(rgb_path, f)

    if train_path is None:
        raise ValueError("Train path not found.")
    if val_path is None:
        raise ValueError("Validation path not found.")
    if test_path is None:
        raise ValueError("Test path not found.")

    return train_path, val_path, test_path
