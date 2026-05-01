#!/usr/bin/env bash
# A4000 — pacman + amidar baseline experiments across 4 GPUs.
# Each algorithm gets its own GPU and runs both envs concurrently on it.
#   GPU 0 → ppo   (amidar, pacman)
#   GPU 1 → drnd  (amidar, pacman)
#   GPU 2 → trpo  (amidar, pacman)
#   GPU 3 → psne  (amidar, pacman)
# 4 algos × 2 envs = 8 processes, 2 per GPU concurrent. --num-runs 5 each.
#
# Children are nohup'd + disowned so the terminal returns immediately and
# jobs survive logout. Monitor with: tail -f log/baselines_*_*.out

set -u
mkdir -p log
PROJECT="baselines"
ENVS=(amidar pacman)

# (gpu, algo)
ALLOCATIONS=(
    "0|ppo"
    "1|drnd"
    "2|trpo"
    "3|psne"
)

for alloc in "${ALLOCATIONS[@]}"; do
    IFS='|' read -r gpu algo <<< "${alloc}"
    for env in "${ENVS[@]}"; do
        tag="${PROJECT}_${env}_${algo}"
        nohup python3 main.py \
            --project "${PROJECT}" \
            --env-name "${env}" \
            --algo-name "${algo}" \
            --num-runs 5 \
            --gpu-idx "${gpu}" \
            > "log/${tag}.out" 2>&1 &
        sleep 3
    done
done

disown -a
echo "Launched 8 baseline jobs (4 algos × 2 envs) across 4 GPUs. PIDs:"
jobs -p
