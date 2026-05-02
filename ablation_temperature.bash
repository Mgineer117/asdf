#!/usr/bin/env bash
# Ablation: --temperature ∈ {0.2, 0.5, 0.8, 1.0} for IRPO with softmax
# aggregation (temperature only affects softmax weighting). Env is
# parameterized via ENV (default pointmaze-v2). 10 runs per setting.

set -u
PROJECT="ablation_temperature"
ENV=${ENV:-pointmaze-v2}
GPU=${GPU:-0}
TEMPS=(0.2 0.5 0.8 1.0)
TAG="${PROJECT}_${ENV}"

# Self-detach so the script survives terminal exit (SIGHUP).
if [ -z "${DETACHED:-}" ]; then
    mkdir -p log
    DETACHED=1 nohup bash "$0" "$@" > "log/${TAG}.master.out" 2>&1 &
    disown
    echo "Detached ${0} (ENV=${ENV}, GPU=${GPU}) as PID $! — tail log/${TAG}.master.out"
    exit 0
fi

for t in "${TEMPS[@]}"; do
    nohup python3 main.py \
        --project "${PROJECT}" \
        --env-name "${ENV}" \
        --algo-name irpo \
        --aggregation-method softmax \
        --temperature "${t}" \
        --num-runs 10 \
        --gpu-idx "${GPU}" \
        > "log/${TAG}_t${t}.out" 2>&1 &
    sleep 3
done
wait
