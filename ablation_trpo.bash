#!/usr/bin/env bash
# Ablation: IRPO_TRPO switch threshold p ∈ {0.1, 0.5, 0.8} on pointmaze-v4.
# Pre-switch the policy uses Thompson IRPO; once learning_progress > p we copy
# the best final exploratory policy into the base actor and switch to a
# vanilla TRPO policy-gradient update for the rest of training.
#
# Includes a baseline `irpo` row (no switch) for comparison.
# 10 runs × (1 baseline + 3 thresholds).

set -u
PROJECT="ablation_trpo"
ENV="pointmaze-v4"
GPU=${GPU:-0}
SWITCHES=(0.1 0.5 0.8)

# Baseline (no switch — default Thompson IRPO)
nohup python3 main.py \
    --project "${PROJECT}" \
    --env-name "${ENV}" \
    --algo-name irpo \
    --irpo-type irpo \
    --num-runs 10 \
    --gpu-idx "${GPU}" \
    > "log/${PROJECT}_baseline.out" 2>&1 &
sleep 3

for p in "${SWITCHES[@]}"; do
    nohup python3 main.py \
        --project "${PROJECT}" \
        --env-name "${ENV}" \
        --algo-name irpo \
        --irpo-type irpo_trpo \
        --trpo-switch-progress "${p}" \
        --num-runs 10 \
        --gpu-idx "${GPU}" \
        > "log/${PROJECT}_p${p}.out" 2>&1 &
    sleep 3
done
wait
