#!/bin/bash
#SBATCH --job-name=band_attribution
#SBATCH --time=01:0:00
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/slurm/%x-%j.out

IMAGE_TYPE=$1
MODEL_PATH=$2

if [[ "$IMAGE_TYPE" != "rgb" && "$IMAGE_TYPE" != "ms" ]]; then
    echo "Usage: sbatch band_attribution.sh <rgb|ms> <model_path>"
    exit 1
fi
if [[ -z "$MODEL_PATH" ]]; then
    echo "Usage: sbatch band_attribution.sh <rgb|ms> <model_path>"
    exit 1
fi

echo "=== Job started: $(date) ==="
echo "Node:        $(hostname)"
echo "Image type:  $IMAGE_TYPE"
echo "Model:       $MODEL_PATH"

module load Python/3.13.5-GCCcore-14.3.0

source .venv/bin/activate

echo "=== Python: $(python --version) | venv active ==="
echo "=== Starting band attribution ==="

for CLASS in 0 1 2 3 4 5 6 7 8 9; do
    echo "=== Class $CLASS ==="
    python -u -m eurosat_classification.features.band_attribution_runner \
        --model_path "$MODEL_PATH" \
        --image_type "$IMAGE_TYPE" \
        --output_dir results/band_attribution \
        --device cuda \
        --target_class "$CLASS"
done

echo "=== Job finished: $(date) ==="
