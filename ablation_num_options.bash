#!/usr/bin/env bash
# Ablation: --num-options ∈ {2, 4, 6, 8} on pointmaze-v4 with
# --num-exp-updates 5. 10 runs per setting.

set -u
PROJECT="ablation_num_options"
ENV="pointmaze-v4"
GPU=${GPU:-0}
NUM_OPTIONS=(2 4 6 8)

# Self-detach so the script survives terminal exit (SIGHUP).
if [ -z "${DETACHED:-}" ]; then
    mkdir -p log
    DETACHED=1 nohup bash "$0" "$@" > "log/${PROJECT}.master.out" 2>&1 &
    disown
    echo "Detached ${0} as PID $! — tail log/${PROJECT}.master.out"
    exit 0
fi

for n in "${NUM_OPTIONS[@]}"; do
    nohup python3 main.py \
        --project "${PROJECT}" \
        --env-name "${ENV}" \
        --algo-name irpo \
        --num-options "${n}" \
        --num-exp-updates 5 \
        --num-runs 10 \
        --gpu-idx "${GPU}" \
        > "log/${PROJECT}_n${n}.out" 2>&1 &
    sleep 3
done
wait
