#!/usr/bin/env bash
# A4000 — NEW_IRPO pacman experiments across 4 GPUs.
# Each algorithm gets its own GPU and fans out 5 seeds concurrently on it.
#   GPU 0 → ppo
#   GPU 1 → drnd
#   GPU 2 → trpo
#   GPU 3 → psne
# 4 algos × 5 seeds = 20 processes, 5 per GPU concurrent.
#
# Children are nohup'd + disowned so the terminal returns immediately and
# jobs survive logout. Monitor with: tail -f log/NEW_IRPO_*.out

set -u
mkdir -p log
PROJECT="NEW_IRPO"
ENV="pacman"
SEEDS=(0 1 2 3 4)

# (gpu, algo)
ALLOCATIONS=(
    "0|hrl"
    "1|maml"
)

for alloc in "${ALLOCATIONS[@]}"; do
    IFS='|' read -r gpu algo <<< "${alloc}"
    for seed in "${SEEDS[@]}"; do
        tag="${PROJECT}_${ENV}_${algo}_seed${seed}"
        nohup python3 main.py \
            --project "${PROJECT}" \
            --env-name "${ENV}" \
            --algo-name "${algo}" \
            --seed "${seed}" \
            --gpu-idx "${gpu}" \
            > "log/${tag}.out" 2>&1 &
        sleep 3
    done
done

disown -a
echo "Launched 10 NEW_IRPO jobs (2 algos × 5 seeds) across 2 GPUs. PIDs:"
jobs -p