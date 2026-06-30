#!/bin/bash
#SBATCH -J run7_gpu
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --gres=gpu:1
#SBATCH -t 08:00:00
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -o $FRONT/run7_gpu.log

# Run 7 — 11-channel ERA5-only REGRESSION. Predicts continuous frontal
# diagnostics (tfp_850/tadv_850/grad_mag_850) instead of discrete classes.
# Threshold-free, no WPC dependency, climate-change robust.
# Output prefix: unet_reg_2019-2024_e30_b32

module load python/GEOSpyD/24.11.3-0/3.12

FRONT=$FRONT

echo "[$(date)] Run 7 GPU starting on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 $FRONT/scripts/train_unet_regression.py \
    --train 2019 2024 \
    --val   2025 2025 \
    --epochs 30 \
    --batch 32 \
    --data-root $FRONT/data

echo "[$(date)] exit=$?"
