import os

from data.download import get_dataset_path


def clean_sealake_folder() -> None:
    ms_sealake_path = os.path.join(get_dataset_path(), "EuroSATallBands", "SeaLake")

    bombay_files = [f for f in os.listdir(ms_sealake_path) if f.startswith("Bombay")]
    jakarta_files = [f for f in os.listdir(ms_sealake_path) if f.startswith("Jakarta")]

    for f in bombay_files + jakarta_files:
        os.remove(os.path.join(ms_sealake_path, f))

    print(
        f"Removed {len(bombay_files)} bombay files and {len(jakarta_files)} jakarta files from SeaLake"
    )
