#!/usr/bin/env bash
# Ablation: TRPO vs SGD for the IRPO base-policy update on pointmaze-v4.
# 10 runs per setting.

set -u
PROJECT="ablation_base_update"
ENV="pointmaze-v4"
GPU=${GPU:-0}

# Self-detach so the script survives terminal exit (SIGHUP).
if [ -z "${DETACHED:-}" ]; then
    mkdir -p log
    DETACHED=1 nohup bash "$0" "$@" > "log/${PROJECT}.master.out" 2>&1 &
    disown
    echo "Detached ${0} as PID $! — tail log/${PROJECT}.master.out"
    exit 0
fi

for update_type in trpo sgd; do
    nohup python3 main.py \
        --project "${PROJECT}" \
        --env-name "${ENV}" \
        --algo-name irpo \
        --base-policy-update-type "${update_type}" \
        --num-runs 10 \
        --gpu-idx "${GPU}" \
        > "log/${PROJECT}_${update_type}.out" 2>&1 &
    sleep 3
done
wait
