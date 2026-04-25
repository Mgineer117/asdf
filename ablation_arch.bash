#!/usr/bin/env bash
# Ablation: actor/critic architecture × activation for IRPO on pointmaze-v4.
# Sizes:        [32,32], [512,512], [256,128,64]
# Activations:  relu, tanh
# 10 runs × 3 sizes × 2 activations.
#
# Critic mirrors actor: --critic-fc-dim is passed the same dims as actor.

set -u
PROJECT="ablation_arch"
ENV="pointmaze-v4"
GPU=${GPU:-0}

ARCHS=(
    "32 32"
    "512 512"
    "256 128 64"
)
ACTIVATIONS=(relu tanh)

for arch in "${ARCHS[@]}"; do
    tag=$(echo "${arch}" | tr ' ' '-')
    for act in "${ACTIVATIONS[@]}"; do
        # shellcheck disable=SC2086
        nohup python3 main.py \
            --project "${PROJECT}" \
            --env-name "${ENV}" \
            --algo-name irpo \
            --actor-fc-dim ${arch} \
            --critic-fc-dim ${arch} \
            --actor-activation "${act}" \
            --num-runs 10 \
            --gpu-idx "${GPU}" \
            > "log/${PROJECT}_${tag}_${act}.out" 2>&1 &
        sleep 3
    done
done
wait
