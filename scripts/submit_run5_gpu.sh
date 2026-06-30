#!/bin/bash
#SBATCH -J run5_gpu
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --gres=gpu:1
#SBATCH -t 08:00:00
#SBATCH -A s1001
#SBATCH -o /discover/nobackup/projects/giss/paleofun/hbong/front/run5_gpu.log

# Run 5 — 12-channel hybrid labels (ERA5 TFP position x WPC expert type),
# 5-class (BG/CF/WF/SF/OF). The full system.
# Output prefix: unet_v4_hybrid_2019-2024_e30_b32

module load python/GEOSpyD/24.11.3-0/3.12

FRONT=/discover/nobackup/projects/giss/paleofun/hbong/front

echo "[$(date)] Run 5 GPU starting on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 $FRONT/train_unet_v4.py \
    --train 2019 2024 \
    --val   2025 2025 \
    --epochs 30 \
    --batch 32 \
    --data-root $FRONT/data

echo "[$(date)] exit=$?"
