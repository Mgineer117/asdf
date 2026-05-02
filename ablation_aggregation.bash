#!/usr/bin/env bash
# Ablation: --aggregation-method ∈ {uniform, argmax, softmax} for IRPO.
# Env is parameterized via ENV (default pointmaze-v2). 10 runs per setting.

set -u
PROJECT="ablation_aggregation"
ENV=${ENV:-pointmaze-v2}
GPU=${GPU:-0}
METHODS=(uniform argmax softmax)
TAG="${PROJECT}_${ENV}"

# Self-detach so the script survives terminal exit (SIGHUP).
if [ -z "${DETACHED:-}" ]; then
    mkdir -p log
    DETACHED=1 nohup bash "$0" "$@" > "log/${TAG}.master.out" 2>&1 &
    disown
    echo "Detached ${0} (ENV=${ENV}, GPU=${GPU}) as PID $! — tail log/${TAG}.master.out"
    exit 0
fi

for m in "${METHODS[@]}"; do
    nohup python3 main.py \
        --project "${PROJECT}" \
        --env-name "${ENV}" \
        --algo-name irpo \
        --aggregation-method "${m}" \
        --num-runs 10 \
        --gpu-idx "${GPU}" \
        > "log/${TAG}_${m}.out" 2>&1 &
    sleep 3
done
wait
