#!/usr/bin/env bash
# EUROPA — Actor architecture ablation using PPO on pointmaze-v4.
# Sizes:        [32,32], [512,512], [256,128,64]
# Activations:  relu, tanh
# 6 (size × activation) configs × 10 internal runs each.
# Europa has 2 GPUs: 3 configs on GPU 0, 3 configs on GPU 1.
#
# Critic mirrors actor (--critic-fc-dim = --actor-fc-dim).
# Each child uses --num-runs 10 so all 10 seeds run sequentially in-process.
# Children are nohup'd + disowned so the terminal returns immediately and
# jobs survive your logout.

set -u
mkdir -p log
PROJECT="ablation_arch_ppo"
ENV="pointmaze-v4"

# (gpu, "arch dims", activation)
CONFIGS=(
    "0|32 32|relu"
    "0|32 32|tanh"
    "0|512 512|relu"
    "1|512 512|tanh"
    "1|256 128 64|relu"
    "1|256 128 64|tanh"
)

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r gpu arch act <<< "${cfg}"
    tag=$(echo "${arch}" | tr ' ' '-')
    # shellcheck disable=SC2086
    nohup python3 main.py \
        --project "${PROJECT}" \
        --env-name "${ENV}" \
        --algo-name ppo \
        --actor-fc-dim ${arch} \
        --critic-fc-dim ${arch} \
        --actor-activation "${act}" \
        --num-runs 10 \
        --gpu-idx "${gpu}" \
        > "log/${PROJECT}_${tag}_${act}.out" 2>&1 &
    sleep 3
done

disown -a
echo "Launched ${#CONFIGS[@]} PPO arch configs in background. PIDs:"
jobs -p
