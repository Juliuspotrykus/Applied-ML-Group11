#!/bin/bash
#SBATCH --job-name=robustness
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/slurm/%x-%j.out

MODALITY=$1

if [[ "$MODALITY" != "rgb" && "$MODALITY" != "ms" ]]; then
    echo "Usage: sbatch run_robustness.sh <rgb|ms>"
    exit 1
fi

echo "=== Job started: $(date) ==="
echo "Node:      $(hostname)"
echo "Modality:  $MODALITY"

module load Python/3.13.5-GCCcore-14.3.0

source .venv/bin/activate

echo "=== Python: $(python --version) | venv active ==="
echo "=== Starting robustness evaluation ==="

python -u -m eurosat_classification.robustness.run_robustness "$MODALITY" \
    --max_samples -1 \
    --output_dir results/robustness \
    --batch_size 64

echo "=== Job finished: $(date) ==="
