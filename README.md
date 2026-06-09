# EuroSAT Land-Use Classification — Applied Machine Learning Project - Group 11

A configurable convolutional neural network trained on the [EuroSAT](https://github.com/phelber/EuroSAT) satellite imagery dataset, with hyperparameter search via [Optuna](https://optuna.org/). Supports both the **RGB** (3-channel JPEG) and **multispectral** (13-band TIFF) variants of the dataset, taken as baseline and final models, respectively.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [Reproducing the Project Locally](#reproducing-the-project-locally)
5. [Running the API](#running-the-api)
6. [Running the Streamlit Demo](#running-the-streamlit-demo)
7. [Results & Evaluation](#results--evaluation)

---

## Project overview

- **Input**: 64x64 pixel satellite image with 10 meter ground sampling distance
- **Output**: Land type classification prediction from 10 possible classes:
    - AnnualCrop
    - Forest
    - HerbaceousVegetation
    - Highway
    - Industrial
    - Pasture
    - PermanentCrop
    - Residential
    - River
    - SeaLake

---

## Project structure

```
.
├── eurosat_classification/
│   ├── data/
│   │   ├── datasets.py          # PyTorch Datasets + DataLoaders
│   │   ├── download.py          # Pulls EuroSAT via kagglehub
│   │   ├── split.py             # Train/val/test splitting
│   │   ├── clean.py             # SeaLake folder cleanup
│   │   ├── label_map.py         # Label map for class names
│   │   ├── band_names.py        # MS band names/indices
│   │   └── preprocessors.py     # MS-band normalisation
│   ├── models/
│   │   └── cnn.py               # CNN, CNNConfig, ConvBlockConfig
│   ├── features/                # Explainability & Band attributions
│   │   ├── gradcam.py           # Grad-CAM
│   │   ├── integrated_gradients.py     # Integrated gradients
│   │   ├── band_attribution_runner.py  # Calculated attribution per band
│   │   ├── alignment_scores.py  # Calculates alignment of band attribution with literature
│   │   ├── band_attribution.sh  # SLURM submission script
│   │   ├── test_files/          # Satellite images for testing
│   │   └── retrieve_model.py    # Loads a saved model from .pkl
│   ├── notebooks/               # Preprocessing experiments
│   ├── robustness/              # Preprocessing experiments
│   │   ├── evaluate.py          # Evaluation of robustness experiments
│   │   ├── perturbations.py     # Perturbation functions to test robustness
│   │   └── run_robustness.py    # Robustness evaluation + plot creation
│   └── train/
│       ├── train.py             # train_model() + evaluate()
│       ├── tune.py              # Optuna search
│       ├── run_training.py      # Trains best config + saves the model
│       ├── compare_models.py    # Trains model several runs to obtain mean & SEM
│       ├── ablation.py          # Trains model without certain bands removed
│       └── hyperparameters.sh   # SLURM submission script
├── tests/                       # unittests
├── models/                      # Trained models land here
├── docker/                      # Dockerfile for containerization
├── logs/slurm/                  # SLURM job outputs (.out)
├── pyproject.toml               # Project + Dependencies
├── uv.lock                      # Locked dependency versions
├── app.py                       # Streamlit demo 
└── main.py                      # FastAPI app
```

----

## Prerequisites

- **Python ≥ 3.12** (`pyproject.toml`)
- uv for environment and dependency management
- Docker 

Install uv: 

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

```bash
git clone https://github.com/Juliuspotrykus/Applied-ML-Group11.git 
cd Applied-ML-Group11
uv sync
source .venv/bin/activate
```

---

## Reproducing the Project Locally

### Data

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

### Usage

#### Hyperparameter tuning

`tune.py` runs an Optuna study (defaults to 30 trials, MedianPruner) for one image modality at a time, maximising validation macro-F1. It prints the best F1 and parameters.

Run locally:

```bash
uv run python -m eurosat_classification.train.tune rgb
uv run python -m eurosat_classification.train.tune ms
```

#### Training the final model

`run_training.py` holds the best hyperparameters found by tuning (`BEST_PARAMS`), retrains on them, and saves the trained model plus a loss/F1 plot to `models/<modality>_model_final.pkl` (and `.png`):

```bash
uv run python -m eurosat_classification.train.run_training rgb
uv run python -m eurosat_classification.train.run_training ms
```

#### Running on a SLURM cluster 

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


**TODO**: Add ablation studies, robustness, band attribution, final evaluation, model comparison, etc

---

## Running the API

### Option A: Using FastAPI
From the project root, run:
```bash
uvicorn main:app --reload
```
#### Access the API
Open the following link in your browser: `http://127.0.0.1:8000`

### Option B: Using Docker
From the project root, run:
```bash
docker build -f docker/Dockerfile -t eurosat-api .
docker run -p 8000:8000 eurosat-api
```
#### Access the API
Open the following link in your browser: `http://localhost:8000`

---

## Running the Streamlit Demo
Interactive, user-friendly, tool for uploading satellite images, running predictions, and visualizing explainability.

### Make sure the API is running
Start it via FastAPI or Docker (see above).

### Start the Streamlit App
From the project root, run:
```bash
streamlit run app.py
```

### Use the app through the website opened
1. Select "RGB" or "Multispectral" mode
2. Upload satellite image
3. Click Submit
4. View prediction and confidence
5. (Optionally) Select target class to explain
6. View explainability outputs


---

## Results & Evaluation

**TODO**: Add statistical comparison of models, loss curves, band attributions, ablation studies, alignment scores, etc

