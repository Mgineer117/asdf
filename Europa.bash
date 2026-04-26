#!/usr/bin/env bash
# EUROPA — IRPO ablations on fourrooms-v2 across 2 GPUs.
#   Aggregation method ∈ {uniform, argmax, softmax}     → 3 jobs
#   Temperature        ∈ {0.2, 0.5, 0.8, 1.0} (softmax) → 4 jobs
# Total 7 configs × 10 internal runs each.
#
# GPU placement (split roughly evenly across the 2 GPUs):
#   GPU 0 → agg: uniform, agg: softmax, temp: 0.5, temp: 1.0
#   GPU 1 → agg: argmax,  temp: 0.2,    temp: 0.8
#
# Children are nohup'd + disowned so the terminal returns immediately and
# jobs survive logout. Monitor with: tail -f log/ablation_*_fourrooms-v2_*.out

set -u
mkdir -p log
ENV="fourrooms-v2"

# (gpu, kind, value)  kind ∈ {agg, temp}
CONFIGS=(
    "0|agg|uniform"
    "1|agg|argmax"
    "0|agg|softmax"
    "1|temp|0.2"
    "0|temp|0.5"
    "1|temp|0.8"
    "0|temp|1.0"
)

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r gpu kind val <<< "${cfg}"
    if [ "${kind}" = "agg" ]; then
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
    else
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
    fi
    sleep 3
done

disown -a
echo "Launched ${#CONFIGS[@]} fourrooms-v2 ablation configs in background. PIDs:"
jobs -p
