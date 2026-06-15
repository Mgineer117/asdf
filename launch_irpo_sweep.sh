#!/bin/bash

# Number of agents to run per GPU (3 agents * 4 GPUs = 12 agents total)
AGENTS_PER_GPU=3

echo "=========================================================="
echo "    Launching IRPO Random Hyperparameter Sweep in Background  "
echo "=========================================================="

# ----------------------------------------------------------------------
# 1. IRPO Random Pacman (GPUs 0, 1, 2, 3)
# ----------------------------------------------------------------------
echo "Initializing WandB Sweep for 'irpo-random-pacman'..."
# Create the sweep and capture the ID
IRPO_OUT=$(python3 search_irpo_random.py --count 0 --project IRPO-RANDOM-PACMAN-SWEEP 2>&1)
SWEEP_ID_IRPO=$(echo "$IRPO_OUT" | grep "Created NEW wandb sweep with ID:" | awk '{print $NF}')

if [ -z "$SWEEP_ID_IRPO" ]; then
    echo "❌ Failed to create sweep for IRPO. Output:"
    echo "$IRPO_OUT"
    exit 1
fi
echo "✅ IRPO Sweep created with ID: $SWEEP_ID_IRPO"

echo "🚀 Launching 12 agents for IRPO Random (3 per GPU on GPUs 0, 1, 2, 3)..."
for ((i=1; i<=$AGENTS_PER_GPU; i++)); do
    CUDA_VISIBLE_DEVICES=0 python3 search_irpo_random.py --sweep_id $SWEEP_ID_IRPO --project IRPO-RANDOM-PACMAN-SWEEP > log_sweep_irpo_gpu0_$i.txt 2>&1 &
    CUDA_VISIBLE_DEVICES=1 python3 search_irpo_random.py --sweep_id $SWEEP_ID_IRPO --project IRPO-RANDOM-PACMAN-SWEEP > log_sweep_irpo_gpu1_$i.txt 2>&1 &
    CUDA_VISIBLE_DEVICES=2 python3 search_irpo_random.py --sweep_id $SWEEP_ID_IRPO --project IRPO-RANDOM-PACMAN-SWEEP > log_sweep_irpo_gpu2_$i.txt 2>&1 &
    CUDA_VISIBLE_DEVICES=3 python3 search_irpo_random.py --sweep_id $SWEEP_ID_IRPO --project IRPO-RANDOM-PACMAN-SWEEP > log_sweep_irpo_gpu3_$i.txt 2>&1 &
done

echo ""
echo "=========================================================="
echo "🎉 All 12 agents successfully deployed to the background! "
echo "   - IRPO logs : log_sweep_irpo_gpuX_Y.txt"
echo "=========================================================="
