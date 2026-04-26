#!/usr/bin/env bash
# Ablation: --min-base-updates (lock duration) for Thompson IRPO on
# pointmaze-v4. Main IRPO is now Thompson, so we sweep the lock value.
#
# 10 runs × {1, 5, 10, 20} settings. Set GPU=<idx> to pin GPU.

set -u
PROJECT="ablation_thompson"
ENV="pointmaze-v4"
GPU=${GPU:-0}
MIN_UPDATES=(1 5 10 20)

# Self-detach so the script survives terminal exit (SIGHUP).
if [ -z "${DETACHED:-}" ]; then
    mkdir -p log
    DETACHED=1 nohup bash "$0" "$@" > "log/${PROJECT}.master.out" 2>&1 &
    disown
    echo "Detached ${0} as PID $! — tail log/${PROJECT}.master.out"
    exit 0
fi

for k in "${MIN_UPDATES[@]}"; do
    nohup python3 main.py \
        --project "${PROJECT}" \
        --env-name "${ENV}" \
        --algo-name irpo \
        --num-runs 10 \
        --min-base-updates "${k}" \
        --gpu-idx "${GPU}" \
        > "log/${PROJECT}_k${k}.out" 2>&1 &
    sleep 3
done
wait
