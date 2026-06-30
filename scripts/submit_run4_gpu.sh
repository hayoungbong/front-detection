#!/bin/bash
#SBATCH -J run4_gpu
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --gres=gpu:1
#SBATCH -t 08:00:00
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -o $FRONT/run4_gpu.log

# Run 4 — 4-channel TFP baseline (t850/u850/v850/tfp_850), 4-class (BG/CF/WF/SF).
# Clean GPU re-run of the TFP ceiling for direct comparison vs Run 5/6.
# Output prefix: unet_2019-2024_e30_b32

module load python/GEOSpyD/24.11.3-0/3.12

FRONT=$FRONT

echo "[$(date)] Run 4 GPU starting on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 $FRONT/scripts/train_unet_classifier_legacy.py \
    --train 2019 2024 \
    --val   2025 2025 \
    --epochs 30 \
    --batch 32 \
    --data-dir  $FRONT/data/training \
    --model-dir $FRONT/data/models

echo "[$(date)] exit=$?"
