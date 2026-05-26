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

echo "=== Job started: $(date) ==="
echo "Node:      $(hostname)"

module load Python/3.13.5-GCCcore-14.3.0

source .venv/bin/activate

echo "=== Python: $(python --version) | venv active ==="
echo "=== Starting hyperparameter tuning ==="

python /eurosat_classification/train/tune.py

echo "=== Job finished: $(date) ==="
