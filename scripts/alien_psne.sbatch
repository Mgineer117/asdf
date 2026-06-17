#!/bin/bash
#SBATCH --job-name=alien_psne
#SBATCH --account=huytran1-ic
#SBATCH --partition=eng-research-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/alien_psne.o%j
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

# === Run 5 seeds (0-4) in parallel on 1 GPU / Node === #
for SEED in {0..4}; do
    python3 main.py --project Atari --env alien --algo psne --seed $SEED --gpu-idx 0 &
    sleep 3
done

# === Wait for all 5 runs to finish ===
wait
