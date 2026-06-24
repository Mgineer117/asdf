#!/bin/bash
#SBATCH --job-name=alien_ppo
#SBATCH --account=huytran1-ic
#SBATCH --partition=eng-research-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/alien_ppo.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

ulimit -n 4096  # raise file descriptor limit

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook)"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/minjae5/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

conda activate irpo

mkdir -p logs

# === Run 5 seeds (0-4) in parallel on 1 GPU / Node === #
for SEED in {0..4}; do
    python3 main.py --project Atari --env alien --algo ppo  --seed $SEED --gpu-idx 0 &
    sleep 3
done

wait
