#!/bin/bash
#SBATCH --job-name=atari_allo_eng
#SBATCH --account=huytran1-ic
#SBATCH --partition=eng-research-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=2
#SBATCH --mem=512G
#SBATCH --gres=gpu:2
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/atari_allo_eng.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

ulimit -n 4096  # raise file descriptor limit

# 1. Block Python from reading ~/.local/lib to prevent cross-contamination
export PYTHONNOUSERSITE=1

# 2. Hook conda into the non-interactive shell
eval "$(conda shell.bash hook)"

# 3. Add MuJoCo and NVIDIA to path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/minjae5/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# 4. Safely activate environment
conda activate irpo

mkdir -p logs

echo "Starting training for ALLO models on Atari environments..."
echo "Distributing alien and amidar to GPU 0, bankheist and pacman to GPU 1"

# === GPU 0: Alien and Amidar ===
for SEED in {0..4}; do
    python3 main.py --project Atari --env alien --algo irpo --int-reward-type allo --seed $SEED --gpu-idx 0 &
    sleep 3
    python3 main.py --project Atari --env amidar --algo irpo --int-reward-type allo --seed $SEED --gpu-idx 0 &
    sleep 3
done

# === GPU 1: Bankheist and Pacman ===
for SEED in {0..4}; do
    python3 main.py --project Atari --env bankheist --algo irpo --int-reward-type allo --seed $SEED --gpu-idx 1 &
    sleep 3
    python3 main.py --project Atari --env pacman --algo irpo --int-reward-type allo --seed $SEED --gpu-idx 1 &
    sleep 3
done

# Wait for all background jobs to finish
wait
echo "All jobs finished!"
