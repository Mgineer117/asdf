#!/bin/bash
#SBATCH --job-name=trpo
#SBATCH --account=huytran1-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=trpo.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

ulimit -n 4096  # raise file descriptor limit

# 1. Block Python from reading ~/.local/lib to prevent cross-contamination (CRITICAL)
export PYTHONNOUSERSITE=1

# 2. Hook conda into the non-interactive shell
eval "$(conda shell.bash hook)"

# 3. Add MuJoCo to the path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/minjae5/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# 4. Safely activate your environment (DO NOT use source ~/.bashrc)
conda activate irpo

# === Run Experiments in Parallel === # 
# Removed the hardcoded CUDA variables. Slurm handles this for you.
# Added a 3-second sleep between launches to prevent driver timeout during PyTorch initialization.
python3 main.py --project NEW_IRPO --env-name pacman --algo-name drnd --seed 0 --gpu-idx 0 &
sleep 3
python3 main.py --project NEW_IRPO --env-name pacman --algo-name drnd --seed 1 --gpu-idx 0 &
sleep 3
python3 main.py --project NEW_IRPO --env-name pacman --algo-name drnd --seed 2 --gpu-idx 0 &
sleep 3
python3 main.py --project NEW_IRPO --env-name pacman --algo-name drnd --seed 3 --gpu-idx 0 &
sleep 3
python3 main.py --project NEW_IRPO --env-name pacman --algo-name drnd --seed 4 --gpu-idx 0 &
sleep 3
python3 main.py --project NEW_IRPO --env-name pacman --algo-name drnd --seed 5 --gpu-idx 0 &
sleep 3
python3 main.py --project NEW_IRPO --env-name pacman --algo-name drnd --seed 6 --gpu-idx 0 &
sleep 3
python3 main.py --project NEW_IRPO --env-name pacman --algo-name drnd --seed 7 --gpu-idx 0 &
sleep 3
python3 main.py --project NEW_IRPO --env-name pacman --algo-name drnd --seed 8 --gpu-idx 0 &
sleep 3
python3 main.py --project NEW_IRPO --env-name pacman --algo-name drnd --seed 9 --gpu-idx 0 &


# === Wait for all background jobs to finish ===
wait
