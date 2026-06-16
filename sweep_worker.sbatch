#!/bin/bash
#SBATCH --job-name=irpo_sweep
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --account=huytran1-ic
#SBATCH --gpus=h200:1
#SBATCH --array=1-10
#SBATCH --nice=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G

# The sweep ID is provided as the first argument to sbatch
SWEEP_ID=$1

if [ -z "$SWEEP_ID" ]; then
    echo "Error: No sweep ID provided!"
    exit 1
fi

echo "Starting WandB agent for Sweep ID: $SWEEP_ID [Array Task $SLURM_ARRAY_TASK_ID]"

# Run the agent (each parallel worker will execute 20 trials before exiting)
python search_irpo_random.py --sweep_id "$SWEEP_ID" --count 20
