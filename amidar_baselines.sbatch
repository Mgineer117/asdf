#!/bin/bash
#SBATCH --job-name=amidar-baselines
#SBATCH --account=huytran1-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=15
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=3-00:00:00
#SBATCH --output=amidar_baselines.o%j
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

# === GPU 0: drnd, psne, trpo (3 algos x 3 seeds = 9 runs) === #
python3 main.py --project NEW_IRPO --env-name amidar --algo-name drnd --seed 0 --gpu-idx 0 &
sleep 3
python3 main.py --project NEW_IRPO --env-name amidar --algo-name drnd --seed 1 --gpu-idx 0 &
sleep 3
python3 main.py --project NEW_IRPO --env-name amidar --algo-name drnd --seed 2 --gpu-idx 0 &
sleep 3
python3 main.py --project NEW_IRPO --env-name amidar --algo-name psne --seed 0 --gpu-idx 0 &
sleep 3
python3 main.py --project NEW_IRPO --env-name amidar --algo-name psne --seed 1 --gpu-idx 0 &
sleep 3
python3 main.py --project NEW_IRPO --env-name amidar --algo-name psne --seed 2 --gpu-idx 0 &
sleep 3
python3 main.py --project NEW_IRPO --env-name amidar --algo-name trpo --seed 0 --gpu-idx 0 &
sleep 3
python3 main.py --project NEW_IRPO --env-name amidar --algo-name trpo --seed 1 --gpu-idx 0 &
sleep 3
python3 main.py --project NEW_IRPO --env-name amidar --algo-name trpo --seed 2 --gpu-idx 0 &
sleep 3

# === GPU 1: hrl, maml (2 algos x 3 seeds = 6 runs) === #
python3 main.py --project NEW_IRPO --env-name amidar --algo-name hrl --seed 0 --gpu-idx 1 &
sleep 3
python3 main.py --project NEW_IRPO --env-name amidar --algo-name hrl --seed 1 --gpu-idx 1 &
sleep 3
python3 main.py --project NEW_IRPO --env-name amidar --algo-name hrl --seed 2 --gpu-idx 1 &
sleep 3
python3 main.py --project NEW_IRPO --env-name amidar --algo-name maml --seed 0 --gpu-idx 1 &
sleep 3
python3 main.py --project NEW_IRPO --env-name amidar --algo-name maml --seed 1 --gpu-idx 1 &
sleep 3
python3 main.py --project NEW_IRPO --env-name amidar --algo-name maml --seed 2 --gpu-idx 1 &

# === Wait for all background jobs to finish ===
wait
