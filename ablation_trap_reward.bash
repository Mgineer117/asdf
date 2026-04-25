#!/usr/bin/env bash
# Ablation: trap-reward environment (fourrooms-v2) — compares baseline
# algorithms against IRPO with varying --num-exp-updates.
#
# Setup:
#   - Env:  fourrooms-v2
#   - IRPO: --num-exp-updates ∈ {2, 5, 10, 20}, default Thompson IRPO
#   - Baselines: ppo, trpo, drnd, hrl, htrpo, maml, psne, irpo_random
#   - 10 runs per setting → (4 + 8) settings.

set -u
PROJECT="ablation_trap_reward"
ENV="fourrooms-v2"
GPU=${GPU:-0}
NUM_EXP=(2 5 10 20)
BASELINES=(ppo trpo drnd hrl htrpo maml psne irpo_random)

# IRPO (default = Thompson) with varying num_exp_updates
for k in "${NUM_EXP[@]}"; do
    nohup python3 main.py \
        --project "${PROJECT}" \
        --env-name "${ENV}" \
        --algo-name irpo \
        --num-exp-updates "${k}" \
        --num-runs 10 \
        --gpu-idx "${GPU}" \
        > "log/${PROJECT}_irpo_k${k}.out" 2>&1 &
    sleep 3
done

# Baseline algorithms
for algo in "${BASELINES[@]}"; do
    # irpo_random shares the irpo entry-point with --int-reward-type random
    if [ "${algo}" = "irpo_random" ]; then
        algo_name="irpo"
        extra_args=(--int-reward-type random)
    else
        algo_name="${algo}"
        extra_args=()
    fi
    nohup python3 main.py \
        --project "${PROJECT}" \
        --env-name "${ENV}" \
        --algo-name "${algo_name}" \
        "${extra_args[@]}" \
        --num-runs 10 \
        --gpu-idx "${GPU}" \
        > "log/${PROJECT}_${algo}.out" 2>&1 &
    sleep 3
done
wait
