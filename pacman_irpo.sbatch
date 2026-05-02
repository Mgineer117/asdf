#!/bin/bash
#SBATCH --job-name=pacman-irpo
#SBATCH --account=huytran1-ic
#SBATCH --partition=eng-research-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

set -u
cd "$SLURM_SUBMIT_DIR"
mkdir -p log
ulimit -n 4096

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
eval "$(conda shell.bash hook)"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/minjae5/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
conda activate irpo

PROJECT="NEW_IRPO"
ENV_NAME="pacman"
SEEDS=(0 1 2)

for seed in "${SEEDS[@]}"; do
    tag="${PROJECT}_${ENV_NAME}_irpo_allo_seed${seed}"
    python3 main.py \
        --project "${PROJECT}" \
        --env-name "${ENV_NAME}" \
        --algo-name irpo \
        --int-reward-type allo \
        --seed "${seed}" \
        --gpu-idx 0 \
        > "log/${tag}.out" 2>&1 &
    sleep 3
done

for seed in "${SEEDS[@]}"; do
    tag="${PROJECT}_${ENV_NAME}_irpo_random_seed${seed}"
    python3 main.py \
        --project "${PROJECT}" \
        --env-name "${ENV_NAME}" \
        --algo-name irpo \
        --int-reward-type random \
        --seed "${seed}" \
        --gpu-idx 1 \
        > "log/${tag}.out" 2>&1 &
    sleep 3
done

wait
