#!/bin/bash
#SBATCH --job-name=irpo
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=07:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/irpo_%j.log
#SBATCH --error=logs/irpo_%j.err

# CSL GPU job: IRPO training with 7-hour timeout
# Usage: sbatch commands/run_irpo_csl.sbatch

set -e
module purge
module load cuda/12.4
source ~/.bashrc
conda activate base

cd /Users/minjae/Library/CloudStorage/OneDrive-UniversityofIllinois-Urbana/Minjae/01_Research/01_UIUC/2025_year/IRPO/asdf

# Default environment: pointmaze-v4
# Override with: sbatch --export=ENV_NAME=maze-v1 commands/run_irpo_csl.sbatch
ENV_NAME=${ENV_NAME:-pointmaze-v4}
SEED=${SEED:-0}
PROJECT=${PROJECT:-IRPO-CSL}

echo "========================================="
echo "IRPO Training on CSL"
echo "Environment: $ENV_NAME"
echo "Seed: $SEED"
echo "Project: $PROJECT"
echo "Time limit: 7 hours"
echo "========================================="

python main.py \
  --env-name "$ENV_NAME" \
  --algo-name irpo \
  --seed "$SEED" \
  --project "$PROJECT" \
  --wandb-mode online
