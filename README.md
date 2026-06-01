# EuroSAT Land-Use Classification — Group 11

A configurable convolutional neural network trained on the [EuroSAT](https://github.com/phelber/EuroSAT) satellite imagery dataset, with hyperparameter search via [Optuna](https://optuna.org/). Supports both the **RGB** (3-channel JPEG) and **multispectral** (13-band TIFF) variants of the dataset.

## Project structure

```
.
├── eurosat_classification/
│   ├── data/
│   │   ├── datasets.py        # PyTorch Datasets + DataLoaders 
│   │   ├── download.py        # Pulls EuroSAT via kagglehub
│   │   ├── split.py           
│   │   ├── clean.py           # SeaLake folder cleanup
│   │   ├── label_map.py       
│   │   └── preprocessors.py   # MS-band normalisation
│   ├── models/
│   │   └── cnn.py             # CNN, CNNConfig, ConvBlockConfig
│   ├── features/              
│   └── train/
│       ├── train.py           # train_model() + evaluate(); used by tune.py
│       ├── tune.py            # Optuna search
│       └── hyperparameters.sh # SLURM submission script
├── tests/                     # unittests 
├── models/                    # trained weights land here 
├── pyproject.toml             # project + dependencies
├── uv.lock                    # locked dependency versions
└── main.py
```

## Prerequisites

- **Python ≥ 3.12** (`pyproject.toml`)
- uv for environment and dependency management

Install uv (one-time, from the project's docs):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation

```bash
git clone <repo-url> Applied-ML-Group11
cd Applied-ML-Group11
uv sync
```

```bash
uv run python -m eurosat_classification.train.tune rgb
```

## Data

The EuroSAT dataset is pulled automatically
from Kaggle via `kagglehub` the first time you build
the dataloaders, you don't need to download it manually. It's cached locally, so
subsequent runs reuse it.

The multispectral contains a handful of spurious files in the
`EuroSATallBands/SeaLake` folder. These are removed automatically by
`clean_sealake_folder()` before the datasets are constructed.

To download and clean the data independently, e.g. to inspect it before training:

```python
from eurosat_classification.data.download import get_dataset_path
from eurosat_classification.data.clean import clean_sealake_folder

path = get_dataset_path()   
clean_sealake_folder()      
```

## Usage

### Hyperparameter tuning

`tune.py` runs an Optuna study (defaults to 30 trials, MedianPruner) for one image modality at a time, then retrains the best configuration and saves the weights to `models/best_<modality>.pt`.

Run locally:

```bash
uv run python -m eurosat_classification.train.tune rgb
uv run python -m eurosat_classification.train.tune ms
```

### Running on a SLURM cluster 

`hyperparameters.sh` (eurosat_classification/train/hyperparameters.sh) submits a single-modality run to a GPU partition. Submit one job per modality so each gets its own wall-time budget and runs in parallel:

```bash
sbatch --job-name=tune_rgb eurosat_classification/train/hyperparameters.sh rgb
sbatch --job-name=tune_ms  eurosat_classification/train/hyperparameters.sh ms
```

Logs land in `logs/slurm/<job-name>-<job-id>.out`. Check status with `squeue -u $USER`.

### Loading a trained model

`tune.py` saves `state_dict`s, so you reconstruct the architecture before loading the weights:

```python
import torch
from eurosat_classification.models.cnn import CNN, CNNConfig

config = CNNConfig(...)              # same config as training
model = CNN(config)
model.load_state_dict(torch.load("models/best_rgb.pt", weights_only=True))
model.eval()
```

## API

TODO
