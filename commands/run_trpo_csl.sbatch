#!/bin/bash
#SBATCH --job-name=trpo
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=07:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/trpo_%j.log
#SBATCH --error=logs/trpo_%j.err

# CSL GPU job: TRPO training with 7-hour timeout
# Usage: sbatch commands/run_trpo_csl.sbatch

set -e
module purge
module load cuda/12.4
source ~/.bashrc
conda activate base

cd /Users/minjae/Library/CloudStorage/OneDrive-UniversityofIllinois-Urbana/Minjae/01_Research/01_UIUC/2025_year/IRPO/asdf

ENV_NAME=${ENV_NAME:-pointmaze-v4}
SEED=${SEED:-0}
PROJECT=${PROJECT:-TRPO-CSL}

echo "========================================="
echo "TRPO Training on CSL"
echo "Environment: $ENV_NAME"
echo "Seed: $SEED"
echo "Project: $PROJECT"
echo "Time limit: 7 hours"
echo "========================================="

python main.py \
  --env-name "$ENV_NAME" \
  --algo-name trpo \
  --seed "$SEED" \
  --project "$PROJECT" \
  --wandb-mode online
