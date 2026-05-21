#!/bin/bash
#SBATCH --job-name=drnd
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=07:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/drnd_%j.log
#SBATCH --error=logs/drnd_%j.err

# CSL GPU job: DRND training with 7-hour timeout
# Usage: sbatch commands/run_drnd_csl.sbatch

set -e
module purge
module load cuda/12.4
source ~/.bashrc
conda activate base

cd /Users/minjae/Library/CloudStorage/OneDrive-UniversityofIllinois-Urbana/Minjae/01_Research/01_UIUC/2025_year/IRPO/asdf

ENV_NAME=${ENV_NAME:-pointmaze-v4}
SEED=${SEED:-0}
PROJECT=${PROJECT:-DRND-CSL}

echo "========================================="
echo "DRND Training on CSL"
echo "Environment: $ENV_NAME"
echo "Seed: $SEED"
echo "Project: $PROJECT"
echo "Time limit: 7 hours"
echo "========================================="

python main.py \
  --env-name "$ENV_NAME" \
  --algo-name drnd \
  --seed "$SEED" \
  --project "$PROJECT" \
  --wandb-mode online
