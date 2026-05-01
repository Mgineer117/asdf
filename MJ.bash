#!/usr/bin/env bash
# MJ — single-GPU runs of HRL on amidar and pacman (5 seeds each).
# Both envs run concurrently on the same GPU.
#
# Children are nohup'd + disowned so the terminal returns immediately and
# jobs survive logout. Monitor with: tail -f log/baselines_*_hrl.out

set -u
mkdir -p log
GPU=${GPU:-0}
PROJECT="baselines"
ALGO="hrl"
ENVS=(amidar pacman fourrooms-v2)

for env in "${ENVS[@]}"; do
    tag="${PROJECT}_${env}_${ALGO}"
    nohup python3 main.py \
        --project "${PROJECT}" \
        --env-name "${env}" \
        --algo-name "${ALGO}" \
        --num-runs 5 \
        --gpu-idx "${GPU}" \
        > "log/${tag}.out" 2>&1 &
    sleep 3
done

disown -a
echo "Launched 3 hrl jobs (amidar, pacman, fourrooms-v2) on GPU ${GPU}. PIDs:"
jobs -p
