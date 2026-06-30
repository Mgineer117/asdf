#!/bin/bash
#
# Runs IRPO and MAML only, sequentially, on a single environment.
#
# Sequential (not parallel) on one GPU so a single second-order run does not
# OOM the way the 4-way parallel ALLO script does. Override any of the vars
# below from the command line, e.g.:
#
#   ENV=pacman GPU=1 ./run_irpo_maml_local.sh
#   ENV=antmaze-v1 INT_REWARD=allo ./run_irpo_maml_local.sh
#
set -euo pipefail

ENV="${ENV:-pacman}"
GPU="${GPU:-0}"
INT_REWARD="${INT_REWARD:-allo}"
NUM_RUNS="${NUM_RUNS:-1}"
PROJECT="${PROJECT:-Exp}"

mkdir -p logs

echo "=========================================================="
echo " IRPO + MAML (sequential) on env=$ENV"
echo " int_reward=$INT_REWARD | gpu=$GPU | num_runs=$NUM_RUNS"
echo "=========================================================="

for ALGO in irpo maml; do
    echo ""
    echo ">>> Starting $ALGO on $ENV ..."
    python3 main.py \
        --project "$PROJECT" \
        --env "$ENV" \
        --algo "$ALGO" \
        --int-reward-type "$INT_REWARD" \
        --num-runs "$NUM_RUNS" \
        --gpu-idx "$GPU" \
        2>&1 | tee "logs/${ENV}_${ALGO}.log"
    echo ">>> Finished $ALGO on $ENV"
done

echo ""
echo "All runs (irpo, maml) finished."
