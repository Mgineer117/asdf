#!/bin/bash
#SBATCH --job-name=multi-seed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=07:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/multi_seed_%j.log
#SBATCH --error=logs/multi_seed_%j.err

# CSL GPU job: Multi-seed run with 7-hour timeout
# This is useful for running multiple seeds sequentially in a single job
#
# Usage: sbatch --export=ALGO_NAME=irpo,ENV_NAME=pointmaze-v4,SEEDS="0 1 2" commands/run_multi_seed_csl.sbatch
#
# Default values:
#   ALGO_NAME: irpo
#   ENV_NAME: pointmaze-v4
#   SEEDS: 0 1

set -e
module purge
module load cuda/12.4
source ~/.bashrc
conda activate base

cd /Users/minjae/Library/CloudStorage/OneDrive-UniversityofIllinois-Urbana/Minjae/01_Research/01_UIUC/2025_year/IRPO/asdf

ALGO_NAME=${ALGO_NAME:-irpo}
ENV_NAME=${ENV_NAME:-pointmaze-v4}
SEEDS=${SEEDS:-"0 1"}
PROJECT=${PROJECT:-${ALGO_NAME}-CSL-MultiSeed}

echo "========================================="
echo "Multi-Seed Training on CSL"
echo "Algorithm: $ALGO_NAME"
echo "Environment: $ENV_NAME"
echo "Seeds: $SEEDS"
echo "Project: $PROJECT"
echo "Time limit: 7 hours"
echo "========================================="

for SEED in $SEEDS; do
  echo ""
  echo "Running seed: $SEED"
  echo "---"
  
  python main.py \
    --env-name "$ENV_NAME" \
    --algo-name "$ALGO_NAME" \
    --seed "$SEED" \
    --project "$PROJECT" \
    --wandb-mode online
  
  echo "Seed $SEED completed"
done

echo ""
echo "All seeds completed!"
