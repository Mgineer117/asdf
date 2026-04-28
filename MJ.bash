#!/usr/bin/env bash
# MJ — single-GPU HRL on amidar and pacman with 5 parallel seeds per env.
# 2 envs × 5 seeds = 10 processes running concurrently on the same GPU.
#
# Children are nohup'd + disowned so the terminal returns immediately and
# jobs survive logout. Monitor with: tail -f log/baselines_*_hrl_seed*.out

set -u
mkdir -p log
GPU=${GPU:-0}
PROJECT="baselines"
ALGO="hrl"
ENVS=(amidar)
SEEDS=(0 1 2 3 4)

for env in "${ENVS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        tag="${PROJECT}_${env}_${ALGO}_seed${seed}"
        nohup python3 main.py \
            --project "${PROJECT}" \
            --env-name "${env}" \
            --algo-name "${ALGO}" \
            --seed "${seed}" \
            --gpu-idx "${GPU}" \
            > "log/${tag}.out" 2>&1 &
        sleep 3
    done
done

disown -a
echo "Launched ${#ENVS[@]} envs × ${#SEEDS[@]} seeds = $((${#ENVS[@]} * ${#SEEDS[@]})) hrl jobs on GPU ${GPU}. PIDs:"
jobs -p
