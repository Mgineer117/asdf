#!/usr/bin/env bash
# 104 — wall-clock benchmark: irpo and ppo on pointmaze-v1 and antmaze-v1.
# Runs SERIALLY on a single GPU (no concurrency, no GPU contention) so the
# wall-clock time of each run is uncontaminated. --num-runs 3 per config so
# each measurement covers three full training runs.
#
# Output: log/104_<algo>_<env>.out and stderr-printed wall-clock from `time`.

PROJECT="wallclock_104"

if [[ "${DETACHED:-0}" != "1" ]]; then
    mkdir -p log
    DETACHED=1 nohup bash "$0" "$@" > "log/${PROJECT}.master.out" 2>&1 &
    disown
    echo "Launched detached ${PROJECT} benchmark. Monitor log/${PROJECT}.master.out"
    exit 0
fi

set -u
mkdir -p log
GPU=${GPU:-0}

CONFIGS=(
    "irpo|pointmaze-v1"
    "irpo|antmaze-v1"
    "ppo|pointmaze-v1"
    "ppo|antmaze-v1"
)

for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r algo env <<< "${cfg}"
    tag="${PROJECT}_${algo}_${env}"
    echo "=== running ${tag} ==="
    start=$(date +%s)
    python3 main.py \
        --project "${PROJECT}" \
        --env-name "${env}" \
        --algo-name "${algo}" \
        --num-runs 3 \
        --gpu-idx "${GPU}" \
        > "log/${tag}.out" 2>&1
    rc=$?
    end=$(date +%s)
    secs=$((end - start))
    echo "wall_clock_seconds=${secs}  exit=${rc}" | tee -a "log/${tag}.out"
done

echo "=== done; per-run wall-clock summary ==="
for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r algo env <<< "${cfg}"
    tag="${PROJECT}_${algo}_${env}"
    grep -h "wall_clock_seconds" "log/${tag}.out" | tail -1 | sed "s/^/${tag}: /"
done
