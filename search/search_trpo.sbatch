#!/bin/bash
#SBATCH --job-name=search_trpo
#SBATCH --account=huytran1-ic
#SBATCH --partition=ic-express
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --gres=gpu:h100.1g.20gb:1
#SBATCH --time=08:00:00
#SBATCH --output=search/logs/search_trpo.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

cd "$SLURM_SUBMIT_DIR"

ulimit -n 4096
export PYTHONNOUSERSITE=1
eval "$(conda shell.bash hook)"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/minjae5/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
conda activate irpo

mkdir -p search/logs

SWEEP_ID_FILE="search/logs/trpo_sweep_id.txt"

if [ -f "$SWEEP_ID_FILE" ]; then
    SWEEP_ID=$(cat "$SWEEP_ID_FILE")
    echo ">>> Resuming existing TRPO sweep: $SWEEP_ID"
else
    echo ">>> Creating new TRPO sweep..."
    SWEEP_OUTPUT=$(python3 search/files/search_trpo.py --env pacman --count 0 2>&1)
    echo "$SWEEP_OUTPUT"
    SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep "Created NEW wandb sweep with ID:" | awk '{print $NF}')
    if [ -z "$SWEEP_ID" ]; then
        echo "ERROR: Failed to create wandb sweep. Exiting."
        exit 1
    fi
    echo "$SWEEP_ID" > "$SWEEP_ID_FILE"
    echo ">>> Created TRPO sweep: $SWEEP_ID"
fi

echo ">>> Launching 5 agents for TRPO sweep $SWEEP_ID on pacman"
for i in {1..5}; do
    python3 search/files/search_trpo.py \
        --env pacman \
        --sweep_id "$SWEEP_ID" \
        --count 50 \
        --gpu-idx 0 &
    sleep 2
done

wait
echo ">>> [$(date)] Done"
