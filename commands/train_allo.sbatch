#!/bin/bash
#SBATCH --job-name=train_allo
#SBATCH --account=huytran1-ic
#SBATCH --partition=eng-research-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:2
#SBATCH --time=2-00:00:00
#SBATCH --output=cac.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

ulimit -n 4096  # raise file descriptor limit

# Load conda
source ~/.bashrc
# Or: source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh  # if ~/.bashrc doesn't source conda

# Activate your conda environment
conda activate irpo

# === Run CACeriments in Parallel ===
CUDA_VISIBLE_DEVICES=0 python3 train_allo.py --env-name fetchreach-v1 &
CUDA_VISIBLE_DEVICES=1 python3 train_allo.py --env-name antmaze-v3 &


# === Wait for all background jobs to finish ===
wait