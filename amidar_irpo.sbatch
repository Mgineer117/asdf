#!/bin/bash
#SBATCH --job-name=amidar-irpo
#SBATCH --account=huytran1-ic
#SBATCH --partition=eng-research-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=amidar_irpo.o%j
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
# 3-second sleep between launches to avoid driver-init contention.
python3 main.py --project NEW_IRPO --env-name amidar --algo-name irpo --seed 0 --gpu-idx 0 &
sleep 3
python3 main.py --project NEW_IRPO --env-name amidar --algo-name irpo --seed 1 --gpu-idx 0 &
sleep 3
python3 main.py --project NEW_IRPO --env-name amidar --algo-name irpo --seed 2 --gpu-idx 0 &
sleep 3
python3 main.py --project NEW_IRPO --env-name amidar --algo-name irpo --int-reward-type random --seed 0 --gpu-idx 0 &
sleep 3
python3 main.py --project NEW_IRPO --env-name amidar --algo-name irpo --int-reward-type random --seed 1 --gpu-idx 0 &
sleep 3
python3 main.py --project NEW_IRPO --env-name amidar --algo-name irpo --int-reward-type random --seed 2 --gpu-idx 0 &

# === Wait for all background jobs to finish ===
wait
