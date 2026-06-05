#!/bin/bash
#SBATCH --job-name=atari_irpo_random
#SBATCH --account=huytran1-ic
#SBATCH --partition=csl
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/atari_irpo_random.o%j
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

# === Run all 4 Atari envs x 5 seeds in parallel (20 tasks total, 1 GPU) === #
for ENV in pacman amidar bankheist alien; do
    for SEED in {0..4}; do
        python3 main.py --project IRPO_ATARI_RANDOM --env $ENV --algo irpo \
            --int-reward-type random --seed $SEED --gpu-idx 0 &
        sleep 3
    done
done

# === Wait for all 20 runs to finish ===
wait
