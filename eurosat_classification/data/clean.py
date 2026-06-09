import os

from .download import get_dataset_path


def clean_sealake_folder() -> None:
    """
    Removes corrupt files from MS SeaLake dataset. Detected by file names
    beggining with "Bombay" and "Jakarta" instead of "SeaLake".
    """
    ms_sealake_path = os.path.join(get_dataset_path(), "EuroSATallBands", "SeaLake")

    bombay_files = []
    jakarta_files = []

    for f in os.listdir(ms_sealake_path):
        if f.startswith("Bombay"):
            bombay_files.append(f)
        elif f.startswith("Jakarta"):
            jakarta_files.append(f)

    for f in bombay_files + jakarta_files:
        os.remove(os.path.join(ms_sealake_path, f))

    print(
        f"Removed {len(bombay_files)} bombay files and {len(jakarta_files)} "
        f"jakarta files from SeaLake"
    )
