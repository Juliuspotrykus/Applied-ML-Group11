# EuroSAT Land-Use Classification — Group 11

A configurable convolutional neural network trained on the [EuroSAT](https://github.com/phelber/EuroSAT) satellite imagery dataset, with hyperparameter search via [Optuna](https://optuna.org/). Supports both the **RGB** (3-channel JPEG) and **multispectral** (13-band TIFF) variants of the dataset.

## Project structure

```
.
├── eurosat_classification/
│   ├── data/
│   │   ├── datasets.py          # PyTorch Datasets + DataLoaders
│   │   ├── download.py          # Pulls EuroSAT via kagglehub
│   │   ├── split.py             # Train/val/test splitting
│   │   ├── clean.py             # SeaLake folder cleanup
│   │   ├── label_map.py
│   │   ├── band_names.py        # MS band names/indices
│   │   └── preprocessors.py     # MS-band normalisation
│   ├── models/
│   │   └── cnn.py               # CNN, CNNConfig, ConvBlockConfig
│   ├── features/                # Explainability
│   │   ├── gradcam.py           # Grad-CAM
│   │   ├── integrated_gradients.py
│   │   └── retrieve_model.py    # Loads a saved model from .pkl
│   ├── notebooks/               # Preprocessing experiments
│   └── train/
│       ├── train.py             # train_model() + evaluate()
│       ├── tune.py              # Optuna search
│       ├── run_training.py      # Trains best config + saves the model
│       └── hyperparameters.sh   # SLURM submission script
├── tests/                       # unittests
├── models/                      # trained models land here
├── logs/slurm/                  # SLURM job outputs (.out)
├── pyproject.toml               # project + dependencies
├── uv.lock                      # locked dependency versions
└── main.py                      # FastAPI app
```

## Prerequisites

- **Python ≥ 3.12** (`pyproject.toml`)
- uv for environment and dependency management

Install uv: 

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation

```bash
git clone https://github.com/Juliuspotrykus/Applied-ML-Group11.git Applied-ML-Group11
cd Applied-ML-Group11
uv sync
source .venv/bin/activate
```

## Data

The EuroSAT dataset is pulled automatically
from Kaggle via `kagglehub` the first time you build
the dataloaders, you don't need to download it manually. It's cached locally, so
subsequent runs reuse it.

The multispectral contains a handful of spurious files in the
`EuroSATallBands/SeaLake` folder. These are removed automatically by
`clean_sealake_folder()` before the datasets are constructed.

To download and clean the data independently, e.g. to inspect it before training, start a Python session and run::

```python
from eurosat_classification.data.download import get_dataset_path
from eurosat_classification.data.clean import clean_sealake_folder

path = get_dataset_path()   
clean_sealake_folder()      
```

## Usage

### Hyperparameter tuning

`tune.py` runs an Optuna study (defaults to 30 trials, MedianPruner) for one image modality at a time, maximising validation macro-F1. It prints the best F1 and parameters.

Run locally:

```bash
uv run python -m eurosat_classification.train.tune rgb
uv run python -m eurosat_classification.train.tune ms
```

### Training the final model

`run_training.py` holds the best hyperparameters found by tuning (`BEST_PARAMS`), retrains on them, and saves the trained model plus a loss/F1 plot to `models/<modality>_model_final.pkl` (and `.png`):

```bash
uv run python -m eurosat_classification.train.run_training rgb
uv run python -m eurosat_classification.train.run_training ms
```

### Running on a SLURM cluster 

`hyperparameters.sh` (eurosat_classification/train/hyperparameters.sh) submits a single-modality run to a GPU partition. Submit one job per modality so each gets its own wall-time budget and runs in parallel:

```bash
sbatch --job-name=tune_rgb eurosat_classification/train/hyperparameters.sh rgb
sbatch --job-name=tune_ms  eurosat_classification/train/hyperparameters.sh ms
```

Logs land in `logs/slurm/<job-name>-<job-id>.out`. Check status with `squeue -u $USER`.

Once tuning finishes, copy the best parameters it printed into the matching
entry of `BEST_PARAMS` in `run_training.py`, then train the final model on the
cluster. Training is short (a single 30-epoch run), so you can run it directly
on a GPU node rather than submitting a separate batch job:

```bash
srun --partition=gpu --gpus-per-node=rtx_pro_6000:1 --cpus-per-task=8 --mem=32GB --time=1:00:00 \
    bash -c "module load Python/3.13.5-GCCcore-14.3.0 && source .venv/bin/activate && \
    python -u -m eurosat_classification.train.run_training rgb"
```

Swap `rgb` for `ms` to train the multispectral model. The saved model and plot
land in `models/` as described above.

## API

TODO
