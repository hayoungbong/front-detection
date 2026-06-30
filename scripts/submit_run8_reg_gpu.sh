#!/bin/bash
#SBATCH -J run8_reg
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --gres=gpu:1
#SBATCH -t 06:00:00
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -o $FRONT/logs/run8_reg_%j.log

# Run 8 (regression) — 12-channel ERA5-only regression, 2010-2024
# Extends Run 7 (r=0.993, 2019-2024) with 15 years of training data.
# Predicts TFP/tadv/grad_mag at 850 hPa — no WPC labels, climate-robust.

module load python/GEOSpyD/24.11.3-0/3.12

FRONT=$FRONT
mkdir -p $FRONT/logs

echo "[$(date)] Run 8 regression starting on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 $FRONT/scripts/train_unet_regression.py \
    --train 2010 2024 \
    --val   2025 2025 \
    --epochs 30 \
    --batch  32 \
    --run-name run8_reg \
    --data-root $FRONT/data

echo "[$(date)] exit=$?"
