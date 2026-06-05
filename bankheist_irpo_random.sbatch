#!/bin/bash
#SBATCH --job-name=bankheist_irpo_random
#SBATCH --account=huytran1-ic
#SBATCH --partition=csl
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/bankheist_irpo_random.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

ulimit -n 4096  # raise file descriptor limit

export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook)"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/minjae5/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

conda activate irpo
mkdir -p logs

for SEED in {0..4}; do
    python3 main.py --project IRPO_ATARI_RANDOM --env bankheist --algo irpo \
        --int-reward-type random --seed $SEED --gpu-idx 0 &
    sleep 3
done

wait
