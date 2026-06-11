#!/bin/bash
#SBATCH --job-name=pacman_hrl
#SBATCH --account=huytran1-ic
#SBATCH --partition=csl
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --gres=gpu:2
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/pacman_hrl.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

ulimit -n 4096  # raise file descriptor limit

# 1. Block Python from reading ~/.local/lib to prevent cross-contamination (CRITICAL)
export PYTHONNOUSERSITE=1

# 2. Hook conda into the non-interactive shell
eval "$(conda shell.bash hook)"

# 3. Add MuJoCo and NVIDIA to path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/minjae5/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# 4. Safely activate your environment
conda activate irpo

# Create logs directory if missing
mkdir -p logs

# === Run 5 seeds across 2 GPUs === #
# First three seeds (0, 1, 2) on GPU 0
for SEED in 0 1 2; do
    python3 main.py --project Atari --env pacman --algo hrl --seed $SEED --gpu-idx 0 --actor-activation elu &
    sleep 3
done

# Next two seeds (3, 4) on GPU 1
for SEED in 3 4; do
    python3 main.py --project Atari --env pacman --algo hrl --seed $SEED --gpu-idx 1 --actor-activation elu &
    sleep 3
done

# === Wait for all 5 runs to finish ===
wait
