#!/usr/bin/env bash
# Ablation: --num-exp-updates ∈ {2, 5, 10} on pointmaze-v4 with the default
# --num-options from config/envs/pointmaze.json. 10 runs per setting.

set -u
PROJECT="ablation_num_exp_updates"
ENV="pointmaze-v4"
GPU=${GPU:-0}
NUM_EXP=(2 5 10)

# Self-detach so the script survives terminal exit (SIGHUP).
if [ -z "${DETACHED:-}" ]; then
    mkdir -p log
    DETACHED=1 nohup bash "$0" "$@" > "log/${PROJECT}.master.out" 2>&1 &
    disown
    echo "Detached ${0} as PID $! — tail log/${PROJECT}.master.out"
    exit 0
fi

for k in "${NUM_EXP[@]}"; do
    nohup python3 main.py \
        --project "${PROJECT}" \
        --env-name "${ENV}" \
        --algo-name irpo \
        --num-exp-updates "${k}" \
        --num-runs 10 \
        --gpu-idx "${GPU}" \
        > "log/${PROJECT}_k${k}.out" 2>&1 &
    sleep 3
done
wait
