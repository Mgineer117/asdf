#!/usr/bin/env bash
# EUROPA — Baseline algorithms on fourrooms-v2, all on GPU 0.
#   Algorithms: ppo, hrl, drnd, psne, trpo (5 jobs × 10 internal runs each)
#
# Children are nohup'd + disowned so the terminal returns immediately and
# jobs survive logout. Monitor with: tail -f log/baselines_fourrooms-v2_*.out

set -u
mkdir -p log
ENV="fourrooms-v2"
GPU=0
PROJECT="baselines"
ALGOS=(ppo hrl drnd psne trpo)

for algo in "${ALGOS[@]}"; do
    tag="${PROJECT}_${ENV}_${algo}"
    nohup python3 main.py \
        --project "${PROJECT}" \
        --env-name "${ENV}" \
        --algo-name "${algo}" \
        --num-runs 10 \
        --gpu-idx "${GPU}" \
        > "log/${tag}.out" 2>&1 &
    sleep 3
done

disown -a
echo "Launched ${#ALGOS[@]} fourrooms-v2 baseline jobs in background on GPU ${GPU}. PIDs:"
jobs -p
