#!/usr/bin/env bash
# A4000 — IRPO ablations + baselines on fourrooms-v2 across 4 GPUs.
#   Aggregation method ∈ {uniform, argmax, softmax}     → 3 IRPO jobs
#   Temperature        ∈ {0.2, 0.5, 0.8, 1.0} (softmax) → 4 IRPO jobs
#   Baselines          ∈ {ppo, trpo, psne, hrl, drnd}   → 5 baseline jobs
# Total 12 configs × 10 internal runs each, evenly split (3 per GPU).
#
# GPU placement:
#   GPU 0 → agg: uniform,  temp: 0.2,  baseline: ppo
#   GPU 1 → agg: argmax,   temp: 0.5,  baseline: trpo
#   GPU 2 → agg: softmax,  temp: 0.8,  baseline: psne
#   GPU 3 → temp: 1.0,     baseline: hrl, baseline: drnd
#
# Children are nohup'd + disowned so the terminal returns immediately and
# jobs survive logout. Monitor with:
#   tail -f log/ablation_*_fourrooms-v2_*.out
#   tail -f log/baselines_fourrooms-v2_*.out

set -u
mkdir -p log
ENV="fourrooms-v2"

# (gpu, kind, value)  kind ∈ {agg, temp, base}
CONFIGS=(
    "0|agg|uniform"
    "0|agg|argmax"
    "0|agg|softmax"
    "0|temp|0.2"
    "0|temp|0.5"
    "1|temp|0.8"
    "1|temp|1.0"
    "1|base|ppo"
    "1|base|trpo"
    "1|base|psne"
    "1|base|hrl"
    "1|base|drnd"
)

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r gpu kind val <<< "${cfg}"
    case "${kind}" in
        agg)
            project="ablation_aggregation"
            tag="${project}_${ENV}_${val}"
            nohup python3 main.py \
                --project "${project}" \
                --env-name "${ENV}" \
                --algo-name irpo \
                --aggregation-method "${val}" \
                --num-runs 10 \
                --gpu-idx "${gpu}" \
                > "log/${tag}.out" 2>&1 &
            ;;
        temp)
            project="ablation_temperature"
            tag="${project}_${ENV}_t${val}"
            nohup python3 main.py \
                --project "${project}" \
                --env-name "${ENV}" \
                --algo-name irpo \
                --aggregation-method softmax \
                --temperature "${val}" \
                --num-runs 10 \
                --gpu-idx "${gpu}" \
                > "log/${tag}.out" 2>&1 &
            ;;
        base)
            project="baselines"
            tag="${project}_${ENV}_${val}"
            nohup python3 main.py \
                --project "${project}" \
                --env-name "${ENV}" \
                --algo-name "${val}" \
                --num-runs 10 \
                --gpu-idx "${gpu}" \
                > "log/${tag}.out" 2>&1 &
            ;;
    esac
    sleep 3
done

disown -a
echo "Launched ${#CONFIGS[@]} fourrooms-v2 configs in background across 4 GPUs. PIDs:"
jobs -p
