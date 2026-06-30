#!/bin/bash
#SBATCH -J run6_gpu
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --gres=gpu:1
#SBATCH -t 08:00:00
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -o $FRONT/run6_gpu.log

# Run 6 — 12-channel TFP labels (same 12 inputs as Run 5, but 4-class TFP
# labels from era5_training.nc instead of hybrid). Isolates the label-quality
# effect: Run 4 (4ch TFP) -> Run 6 (12ch TFP) -> Run 5 (12ch hybrid).
# Output prefix: unet_v4_tfp_2019-2024_e30_b32

module load python/GEOSpyD/24.11.3-0/3.12

FRONT=$FRONT

echo "[$(date)] Run 6 GPU starting on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 $FRONT/scripts/train_unet_classifier.py \
    --train 2019 2024 \
    --val   2025 2025 \
    --epochs 30 \
    --batch 32 \
    --tfp-labels \
    --data-root $FRONT/data

echo "[$(date)] exit=$?"
