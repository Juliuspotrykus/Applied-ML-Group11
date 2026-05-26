#!/bin/bash
#SBATCH --job-name=hyperparameter_tuning
#SBATCH --time=4:00:00
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/slurm/%x-%j.out

IMAGE_TYPE=$1
if [[ "$IMAGE_TYPE" != "rgb" && "$IMAGE_TYPE" != "ms" ]]; then
    echo "Usage: sbatch --job-name=tune_<type> hyperparameters.sh <rgb|ms>"
    exit 1
fi

echo "=== Job started: $(date) ==="
echo "Node:       $(hostname)"
echo "Image type: $IMAGE_TYPE"

module load Python/3.13.5-GCCcore-14.3.0

source .venv/bin/activate

echo "=== Python: $(python --version) | venv active ==="
echo "=== Starting hyperparameter tuning ($IMAGE_TYPE) ==="

python -u -m eurosat_classification.train.tune "$IMAGE_TYPE"

echo "=== Job finished: $(date) ==="
