#!/usr/bin/env bash
# Ablation: IRPO (default = Thompson + IRPO backprop gradient, base policy is
# inferred from best final-exploratory policy) vs IRPO_IS (Thompson option
# selection + IS-corrected gradient on exploratory rollouts, base policy is
# the primary evaluation policy) on pointmaze-v4.
#
# 10 runs × 2 settings.

set -u
PROJECT="ablation_is"
ENV="pointmaze-v4"
GPU=${GPU:-0}

for irpo_type in irpo irpo_is; do
    nohup python3 main.py \
        --project "${PROJECT}" \
        --env-name "${ENV}" \
        --algo-name irpo \
        --irpo-type "${irpo_type}" \
        --num-runs 10 \
        --gpu-idx "${GPU}" \
        > "log/${PROJECT}_${irpo_type}.out" 2>&1 &
    sleep 3
done
wait
