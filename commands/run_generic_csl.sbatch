#!/bin/bash
#SBATCH --job-name=generic-rl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=07:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/rl_job_%j.log
#SBATCH --error=logs/rl_job_%j.err

# CSL GPU job: Generic RL training with 7-hour timeout
# This is a flexible template for any algorithm + environment combination
#
# Usage: sbatch --export=ALGO_NAME=irpo,ENV_NAME=pointmaze-v4,SEED=0 commands/run_generic_csl.sbatch
#
# Required exports:
#   ALGO_NAME: Algorithm name (irpo, ppo, trpo, drnd, hrl, maml, psne, htrpo)
#   ENV_NAME:  Environment name (default: pointmaze-v4)
#   SEED:      Random seed (default: 0)
#   PROJECT:   WandB project name (default: {ALGO_NAME}-CSL)

set -e

# Validate required parameter
if [ -z "$ALGO_NAME" ]; then
  echo "ERROR: ALGO_NAME not set"
  echo "Usage: sbatch --export=ALGO_NAME=irpo,ENV_NAME=pointmaze-v4 commands/run_generic_csl.sbatch"
  exit 1
fi

module purge
module load cuda/12.4
source ~/.bashrc
conda activate base

cd /Users/minjae/Library/CloudStorage/OneDrive-UniversityofIllinois-Urbana/Minjae/01_Research/01_UIUC/2025_year/IRPO/asdf

# Set defaults
ENV_NAME=${ENV_NAME:-pointmaze-v4}
SEED=${SEED:-0}
PROJECT=${PROJECT:-${ALGO_NAME}-CSL}

# Validate algorithm name
valid_algos="irpo ppo trpo drnd hrl maml psne htrpo"
if ! echo "$valid_algos" | grep -w "$ALGO_NAME" > /dev/null; then
  echo "ERROR: Invalid ALGO_NAME='$ALGO_NAME'"
  echo "Valid algorithms: $valid_algos"
  exit 1
fi

echo "========================================="
echo "RL Training on CSL"
echo "Algorithm: $ALGO_NAME"
echo "Environment: $ENV_NAME"
echo "Seed: $SEED"
echo "Project: $PROJECT"
echo "Time limit: 7 hours"
echo "========================================="

python main.py \
  --env-name "$ENV_NAME" \
  --algo-name "$ALGO_NAME" \
  --seed "$SEED" \
  --project "$PROJECT" \
  --wandb-mode online

echo "Job completed successfully!"
