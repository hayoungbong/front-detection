#!/bin/bash
#SBATCH -J run8_4gpu
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH -t 10:00:00
#SBATCH -A <YOUR_ACCOUNT>
#SBATCH -o $FRONT/logs/run8_%j.log

# Run 8 — 12-channel Hybrid, 2010-2024, 4×A100 DDP
# Tests whether 2.5× more training data (15 yr vs 6 yr) recovers CF/WF/SF
# performance lost in Run 5b (F1=0.255) while retaining OF detection.
# Expected: ~25 hrs on 4×A100 (vs ~90 hrs single GPU).

module load python/GEOSpyD/24.11.3-0/3.12

FRONT=$FRONT
mkdir -p $FRONT/logs

echo "[$(date)] Run 8 starting on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

torchrun \
    --standalone \
    --nproc_per_node=4 \
    $FRONT/scripts/train_unet_classifier.py \
    --train 2010 2024 \
    --val   2025 2025 \
    --epochs 30 \
    --batch  32 \
    --workers 8 \
    --amp \
    --run-name run8 \
    --data-root $FRONT/data

echo "[$(date)] exit=$?"
